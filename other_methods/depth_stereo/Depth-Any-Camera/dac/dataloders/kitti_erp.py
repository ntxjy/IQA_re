import os
import json
import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset, resize_for_input
from dac.utils.erp_geometry import cam_to_erp_patch_fast


class KITTIERPDataset(BaseDataset):
    CAM_INTRINSIC = {
        "2011_09_26": torch.tensor(
            [
                [7.215377e02, 0.000000e00, 6.095593e02, 4.485728e01],
                [0.000000e00, 7.215377e02, 1.728540e02, 2.163791e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.745884e-03],
            ]
        ),
        "2011_09_28": torch.tensor(
            [
                [7.070493e02, 0.000000e00, 6.040814e02, 4.575831e01],
                [0.000000e00, 7.070493e02, 1.805066e02, -3.454157e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 4.981016e-03],
            ]
        ),
        "2011_09_29": torch.tensor(
            [
                [7.183351e02, 0.000000e00, 6.003891e02, 4.450382e01],
                [0.000000e00, 7.183351e02, 1.815122e02, -5.951107e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 2.616315e-03],
            ]
        ),
        "2011_09_30": torch.tensor(
            [
                [7.070912e02, 0.000000e00, 6.018873e02, 4.688783e01],
                [0.000000e00, 7.070912e02, 1.831104e02, 1.178601e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 6.203223e-03],
            ]
        ),
        "2011_10_03": torch.tensor(
            [
                [7.188560e02, 0.000000e00, 6.071928e02, 4.538225e01],
                [0.000000e00, 7.188560e02, 1.852157e02, -1.130887e-01],
                [0.000000e00, 0.000000e00, 1.000000e00, 3.779761e-03],
            ]
        ),
    }
    min_depth = 0.01
    max_depth = 80
    test_split = "kitti_eigen_test.txt"
    train_split = "kitti_eigen_train.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=256,
        crop=None,
        is_dense=False,
        benchmark=False,
        augmentations_db={},
        # masked=True,
        normalize=True,
        erp=True, # indicate whether the dataset is treated as erp (originally erp dataset can be treated as perspective for evaluation matching virtual f to tgt_f)
        tgt_f = 0, # focal length of perspective training data
        cano_sz=(1400, 1400), # half erp size of erp training data
        fwd_sz = (375, 1242),
        visual_debug=False,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        # self.tgt_f = tgt_f
        self.cano_sz = cano_sz
        self.fwd_sz = fwd_sz
        self.visual_debug = visual_debug

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join('splits/kitti', self.split_file)) as f:
            for line in f:
                img_info = dict()
                if not self.benchmark:  # benchmark test
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == "None" or not os.path.exists(
                        os.path.join(self.base_path, depth_map)
                    ):
                        self.invalid_depth_num += 1
                        continue
                    img_info["annotation_filename_depth"] = os.path.join(
                        self.base_path, depth_map
                    )
                    img_info["annotation_filename_erp_range"] = os.path.join(
                        self.base_path, depth_map.replace('.png', '.json')
                    )
                    
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                img_info["pred_scale_factor"] = 1.0
                self.dataset.append(img_info)

        print(
            f"Loaded {len(self.dataset)} images. Totally {self.invalid_depth_num} invalid pairs are filtered"
        )

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        image = np.asarray(
            Image.open(self.dataset[idx]["image_filename"])
        )
        depth = (
            np.asarray(
                Image.open(self.dataset[idx]["annotation_filename_depth"])
            ).astype(np.float32)
            / self.depth_scale
        )
        info = self.dataset[idx].copy()
            
        # prepare kitti image camera intrinsics
        cam_intrinsics = self.CAM_INTRINSIC[self.dataset[idx]["image_filename"].split("/")[2]][:, :3]
        info['camera_intrinsics'] = cam_intrinsics
        
        # convert depth from zbuffer to euclid
        x, y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        depth = depth * np.sqrt((x - cam_intrinsics[0, 2].item())**2 + (y - cam_intrinsics[1, 2].item())**2 + cam_intrinsics[0, 0].item()**2) / cam_intrinsics[0, 0].item()
        depth = depth.astype(np.float32)    
        
        im_width, im_height = image.shape[1], image.shape[0]
        cam_params = {
            'dataset': 'kitti',
            'wFOV': np.arctan(im_width / 2 / cam_intrinsics[0, 0]) * 2,
            'hFOV': np.arctan(im_height / 2 / cam_intrinsics[1, 1]) * 2,
            'width': im_width,
            'height': im_height,
            'fx': cam_intrinsics[0, 0],
            'fy': cam_intrinsics[1, 1],
            'cx': cam_intrinsics[0, 2],
            'cy': cam_intrinsics[1, 2],
        }
            
        theta = 0
        phi = 0
        roll = 0
        crop_w = int(self.cano_sz[0] * (cam_params['wFOV'] + 0.628) / np.pi) # determine the crop size in ERP based on FOV with some padding
        # make crop aspect ratio same as fwd_sz aspect ratio. This make later visualization can directly change erp size for equivalent process
        crop_h = int(crop_w * self.fwd_sz[0] / self.fwd_sz[1])
        
        # convert image to erp patch
        image = image.astype(np.float32) / 255.0
        depth = np.expand_dims(depth, axis=2)
        mask_valid_depth = depth > self.min_depth
    
        image, depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            image, depth, (mask_valid_depth * 1.0).astype(np.float32), theta, phi,
            crop_h, crop_w, self.cano_sz[0], self.cano_sz[0]*2, cam_params, roll, scale_fac=None
        )
        lat_range = np.array([float(np.min(latitude)), float(np.max(latitude))])
        long_range = np.array([float(np.min(longitude)), float(np.max(longitude))])        
    
        # resizing process to fwd_sz.
        image, depth, pad, pred_scale_factor, attn_mask = resize_for_input(image, depth, self.fwd_sz, cam_intrinsics, [image.shape[0], image.shape[1]], 1.0, padding_rgb=[0, 0, 0], mask=erp_mask)
        info['pred_scale_factor'] = info['pred_scale_factor'] * pred_scale_factor
        info['pad'] = pad
        info['phi'] = phi
        if not self.test_mode:
            depth /= info['pred_scale_factor']
            
        image, gts, info = self.transform(image=(image * 255.).astype(np.uint8), gts={"depth": depth, "attn_mask": (attn_mask>0).astype(np.float32)}, info=info)

        if self.visual_debug:
            # visualize image, gts[gt], gts[attn_mask]
            import matplotlib.pyplot as plt
            print(f'phi: {np.rad2deg(phi)} deg, theta: {np.rad2deg(theta)} deg, roll: {np.rad2deg(roll)} deg')
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow((image.permute(1, 2, 0) - image.min()) / (image.max() - image.min()))
            plt.title("Image")
            plt.subplot(2, 2, 2)
            plt.imshow(gts["gt"].squeeze())
            plt.title("Ground Truth")
            plt.subplot(2, 2, 3)
            plt.imshow(gts["mask"].squeeze().bool())
            plt.title("valid Mask")
            plt.subplot(2, 2, 4)
            plt.imshow(gts["attn_mask"].squeeze().bool())
            plt.title("Attn Mask")
            plt.show()
            
        if self.test_mode:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_range": lat_range, "long_range":long_range, "info": info}
        else:
            return {"image": image, "gt": gts["gt"], "mask": gts["mask"], "attn_mask": gts["attn_mask"], "lat_range": lat_range, "long_range":long_range}

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, height_end = 0, self.fwd_sz[0]
        width_start, width_end = 0, self.fwd_sz[1]
        height_start, width_start = int(image.shape[0] - self.fwd_sz[0]), int((image.shape[1] - self.fwd_sz[1]) / 2)
        height_end, width_end = height_start + self.fwd_sz[0], width_start + self.fwd_sz[1]
        
        image = image[height_start:height_end, width_start:width_end]
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start
        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"]
            depth = depth[height_start:height_end, width_start:width_end]
            mask = depth > self.min_depth
            if self.test_mode:
                mask = np.logical_and(mask, depth < self.max_depth)
                mask = np.logical_and(mask, self.eval_mask(mask))
            mask = mask.astype(np.uint8)
            new_gts["gt"] = depth
            new_gts["mask"] = mask
                
        if "attn_mask" in gts:
            attn_mask = gts["attn_mask"]
            if attn_mask is not None:
                attn_mask = attn_mask[height_start:height_end, width_start:width_end]
                new_gts["attn_mask"] = attn_mask

        return image, new_gts, info

    def eval_mask(self, valid_mask):
        """Do grag_crop or eigen_crop for testing"""
        if self.test_mode:
            if self.crop is not None:
                mask_height, mask_width = valid_mask.shape[-2:]
                eval_mask = np.zeros_like(valid_mask)
                if "garg" in self.crop:
                    eval_mask[
                        int(0.40810811 * mask_height) : int(0.99189189 * mask_height),
                        int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
                    ] = 1
                elif "eigen" in self.crop:
                    eval_mask[
                        int(0.3324324 * mask_height) : int(0.91351351 * mask_height),
                        int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
                    ] = 1
                else:
                    eval_mask = np.ones_like(valid_mask)
            valid_mask = np.logical_and(valid_mask, eval_mask)
        return valid_mask
