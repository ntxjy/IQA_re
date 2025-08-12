#!/usr/bin/env python

"""
DepthAnyCamera training script including conversion of any camera input to equirectangular projection (ERP) with pitch awareness and ERP augmentation.
This version trained new idisc model with attention adaptation, multi-resolution data preparation, and mixed datasets.
"""

import argparse
import json
import os
import random
import uuid
from datetime import datetime as dt
from time import time
from typing import Any, Dict

import numpy as np
import shutil
import torch
import torch.utils.data.distributed
from torch import distributed as dist
from torch import nn, optim
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import (DataLoader, DistributedSampler, RandomSampler,
                              SequentialSampler)
import wandb
os.environ["WANDB_START_METHOD"] = "thread"

import dac.dataloders as custom_dataset
from dac.models.idisc_erp import IDiscERP
from dac.models.idisc import IDisc
from dac.models.idisc_equi import IDiscEqui
from dac.models.cnn_depth import CNNDepth
from dac.utils import (DICT_METRICS_DEPTH, DICT_METRICS_NORMALS,
                         RunningMetric, format_seconds, is_main_process,
                         setup_multi_processes, setup_slurm, validate)

def main_worker(gpu, config: Dict[str, Any], args: argparse.Namespace, ngpus_per_node, run_id: str):
    if not args.distributed:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        args.rank = 0
        args.world_size = 1
    else:
        # initializes the distributed backend which will take care of synchronizing nodes/GPUs
        # setup_multi_processes(config)
        # setup_slurm("nccl", port=args.master_port)
        # args.rank = int(os.environ["RANK"])
        # args.world_size = int(os.environ["WORLD_SIZE"])
        # args.local_rank = int(os.environ["LOCAL_RANK"])

        args.rank = args.rank * ngpus_per_node + gpu
        # backend = "nccl" sometimes got stuck at dist.barrier(). Changing to "gloo" can make the training slow
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

        print(f"Start running DDP on: {args.rank}.")
        config["training"]["batch_size"] = int(
            config["training"]["batch_size"] / args.world_size
        )
        # create model and move it to GPU with id rank
        device = args.rank
        torch.cuda.set_device(device)
        if is_main_process():
            print("BatchSize per GPU: ", config["training"]["batch_size"])
        dist.barrier()

    ##############################
    ########## DATASET ###########
    ##############################
    # Datasets loading
    is_normals = config["model"]["output_dim"] > 1
    
    train_datasets = []
    for i_d, dataset_name in enumerate(config["data"]["train_dataset"]):
        save_dir_train = os.path.join(args.base_path, config["data"]["train_data_root"][i_d])
        assert hasattr(
            custom_dataset, dataset_name
        ), f"{dataset_name} not a custom dataset"
    
        if "ERPOnline" in dataset_name:
            train_dataset_single = getattr(custom_dataset, dataset_name)(
                test_mode=False,
                base_path=save_dir_train,
                crop=config["data"]["crop"],
                augmentations_db=config["data"]["augmentations"],
                crop_size=config["data"]["crop_size"],
                erp_height=config["data"]["erp_height"],
                theta_aug_deg=config["data"]["theta_aug_deg"],
                phi_aug_deg=config["data"]["phi_aug_deg"],
                roll_aug_deg=config["data"]["roll_aug_deg"],
                fov_align=config["data"].get("fov_align", True),
            )
        else:
            train_dataset_single = getattr(custom_dataset, dataset_name)(
                test_mode=False,
                base_path=save_dir_train,
                crop=config["data"]["crop"],
                augmentations_db=config["data"]["augmentations"],
            )
        train_datasets.append(train_dataset_single)
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
            
    # creating validation dataset
    val_dataset_name = config["data"]["val_dataset"]
    val_save_dir = os.path.join(args.base_path, config["data"]["val_data_root"])
    if val_dataset_name == "KITTI360Dataset":
        valid_dataset = getattr(custom_dataset, val_dataset_name)(
                test_mode=True, base_path=val_save_dir, crop=config["data"]["crop"],
                tgt_f=config["data"]["tgt_f"],
                undistort_f=config["data"]["undistort_f"],
                fwd_sz=config["data"]["fwd_sz"],
                erp=config["data"]["erp"],
            )
    elif "ERPOnline" in val_dataset_name:
        valid_dataset = getattr(custom_dataset, val_dataset_name)(
                test_mode=True, base_path=val_save_dir, crop=config["data"]["crop"],
                crop_size=config["data"]["crop_size"],
                erp_height=config["data"]["erp_height"],
                theta_aug_deg=config["data"]["theta_aug_deg"],
                phi_aug_deg=config["data"]["phi_aug_deg"],
                roll_aug_deg=config["data"]["roll_aug_deg"],
            )
    else:
        valid_dataset = getattr(custom_dataset, val_dataset_name)(
                test_mode=True, base_path=val_save_dir, crop=config["data"]["crop"]
            )

    if is_normals:
        metrics_tracker = RunningMetric(list(DICT_METRICS_NORMALS.keys()))
    else:
        metrics_tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))

    # Dataset samplers, create distributed sampler pinned to rank
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
    else:
        print("\t-> Local random sampler")
        train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)

    # Dataset loader
    val_batch_size = 2 * config["training"]["batch_size"]
    num_workers = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=1,  # using more sometimes causes issues
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
    )

    ##############################
    ########### MODEL ############
    ##############################
    # Build model
    # model = IDiscERP.build(config).to(device)
    model = eval(args.model_name).build(config).to(device)
    if args.checkpoint_pretrained is not None:
        print(f'    -> Loading pretrained checkpoint from: {args.checkpoint_pretrained}')
        model.load_pretrained(args.checkpoint_pretrained)
    erp_mode = True if "ERP" in args.model_name else False
    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=False,
            broadcast_buffers=True # setting to False to handle the in-place operation error in swinL training, may affect training stability
        )

    ##############################
    ######### OPTIMIZER ##########
    ##############################
    f16 = config["training"].get("f16", False)
    nsteps_accumulation_gradient = config["training"]["nsteps_accumulation_gradient"]
    gen_model = model.module if args.distributed else model
    params, lrs = gen_model.get_params(config)
    optimizer = optim.AdamW(
        params,
        weight_decay=config["training"]["wd"],
    )

    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lrs,
        total_steps=config["training"]["n_iters"],
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        last_epoch=-1,
        pct_start=0.3,
        div_factor=config["training"]["div_factor"],
        final_div_factor=config["training"]["final_div_factor"],
    )

    scaler = torch.cuda.amp.GradScaler()
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)
    optimizer.zero_grad()

    ##############################
    ########## TRAINING ##########
    ##############################
    true_step, step = 0, 0
    start = time()
    n_steps = config["training"]["n_iters"]
    validate.best_loss = np.inf

    if is_main_process():
        # writer = SummaryWriter(os.path.join(args.checkpoint_path, run_id))
        wandb.init(project="depth_any_cam", config=config, name=os.path.join(args.checkpoint_path[2:], run_id))
        wandb.watch(model)
        print("Start training:")

    while True:
        for batch in train_loader:
            # Regress from multiple resolutions and accumlate losses
            loss = 0
            _, _, h_orig, w_orig = batch["image"].shape
            multi_reso_ratios = config["data"].get("multi_reso_ratios", [1.0])
            for resize_ratio in multi_reso_ratios:
                batch_resize = {}
                if resize_ratio != 1.0:
                    h = int(h_orig * resize_ratio)
                    w = int(w_orig * resize_ratio)
                    for k, v in batch.items():
                        if k in ["image"]:
                            batch_resize[k] = nn.functional.interpolate(
                                v, size=(h, w), mode="bilinear", align_corners=True
                            )
                        elif k in ["gt", "mask", "attn_mask"]:
                            batch_resize[k] = nn.functional.interpolate(
                                v, size=(h, w), mode="nearest"
                            )
                            # rescale depth with the same ratio
                            if k == "gt":
                                batch_resize[k] /= resize_ratio
                        else:
                            batch_resize[k] = v
                else:
                    batch_resize = batch
                
                if (step + 1) % nsteps_accumulation_gradient:
                    with context as fp, model.no_sync() as no_sync:
                        batch_resize = {k: v.to(model.device) for k, v in batch_resize.items()}
                        if erp_mode:
                            preds, losses, _ = model(**batch_resize)
                        else:
                            preds, losses, _ = model(batch_resize["image"], batch_resize["gt"], batch_resize["mask"])
                        loss = loss + (
                            sum([v for k, v in losses["opt"].items()])
                            / nsteps_accumulation_gradient
                        )
                else:
                    with context:
                        batch_resize = {k: v.to(model.device) for k, v in batch_resize.items()}
                        if erp_mode:
                            preds, losses, _ = model(**batch_resize)
                        else:
                            preds, losses, _ = model(batch_resize["image"], batch_resize["gt"], batch_resize["mask"])
                        loss = loss + (
                            sum([v for k, v in losses["opt"].items()])
                            / nsteps_accumulation_gradient
                        )        
            loss = loss / len(multi_reso_ratios)

            # Backward pass
            if (step + 1) % nsteps_accumulation_gradient:
                if f16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            # Gradient accumulation (if any), now sync gpus
            else:
                if f16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler is not None and (step // nsteps_accumulation_gradient) < scheduler.total_steps:
                    scheduler.step()
                optimizer.zero_grad()

            step += 1
            true_step = step // nsteps_accumulation_gradient

            if is_main_process():
                # Train loss logging
                if step % (100 * nsteps_accumulation_gradient) == 0:
                    log_loss_dict = {
                        f"Train/{k}": v.detach().cpu().item()
                        for loss in losses.values()
                        for k, v in loss.items()
                    }
                    elapsed = int(time() - start)
                    eta = int(elapsed * (n_steps - true_step) / max(1, true_step))
                    print(
                        f"Loss at {true_step}/{n_steps} [{format_seconds(elapsed)}<{format_seconds(eta)}]:"
                    )
                    print(
                        ", ".join([f"{k}: {v:.5f}" for k, v in log_loss_dict.items()])
                    )
                    for k, v in log_loss_dict.items():
                        # writer.add_scalar(k, v, true_step)
                        wandb.log({k: v}, step=true_step)

            #  Validation
            is_last_step = true_step == config["training"]["n_iters"]
            is_validation = (
                    step
                    % (
                            nsteps_accumulation_gradient
                            * config["training"]["validation_interval"]
                    )
                    == 0
            )
            if is_last_step or is_validation:
                del preds, losses, loss, batch
                torch.cuda.empty_cache()

                if is_main_process():
                    print(f"Validation at {true_step}th step...")
                model.eval()
                start_validation = time()
                with torch.no_grad():
                    metrics_all = validate(
                        model,
                        test_loader=valid_loader,
                        step=true_step,
                        config=config,
                        run_id=run_id,
                        out_dir=args.checkpoint_path,
                        metrics_tracker=metrics_tracker,
                        context=context,
                        erp_mode=erp_mode,
                        max_num_samples=200,
                        is_last_step=is_last_step,
                    )
                if is_main_process():
                    for k, v in metrics_all.items():
                        # writer.add_scalar(k, v, true_step)
                        wandb.log({'Validation/'+k: v}, step=true_step)
                    print(f"Elapsed: {format_seconds(int(time() - start_validation))}")
                model.train()

            # Exit
            if true_step == config["training"]["n_iters"]:
                if args.distributed:
                    dist.destroy_process_group()
                return 0


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description="Training script", conflict_handler="resolve"
    )
    wandb.login()

    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="IDiscERP", help="Model name: IDiscERP, IDisc, IDiscEqui, CNNDepth")
    parser.add_argument("--master-port", type=str, required=False)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--base-path", default=os.environ.get("TMPDIR", ""))
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint-pretrained", type=str, default=None)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)
    parser.add_argument('--dist_url', type=str, help='url used to set up distributed training',
                        default='tcp://127.0.0.1:1234')

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    # make checkpoint path
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{uuid.uuid4()}"
    args.checkpoint_path = os.path.join(args.checkpoint_path, os.path.basename(args.config_file[:-5]))
    if config["data"].get("fov_align", True) == True:
        args.checkpoint_path += "_fov_align"
    if config["data"].get("multi_reso_ratios", [1.0]) != [1.0]:
        args.checkpoint_path += "_multi_reso"
    args.checkpoint_path += "_"+args.model_name
    os.makedirs(os.path.join(args.checkpoint_path, run_id), exist_ok=True)

    # copy config_file to checkpoint path
    shutil.copy(args.config_file, os.path.join(args.checkpoint_path, run_id))

    # fix seeds
    seed = config["generic"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(config, args, ngpus_per_node, run_id))
    else:
        main_worker(0, config, args, ngpus_per_node, run_id)
