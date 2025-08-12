"""
The dataset provide multiple options to evaluate metric depth model: direct raw fisheye, undistorted fisheye, and fisheye converted to ERP.
In testing, the model is assumed being trained from perspective images. The testing images' focal length will be matched to the training camera model in focal length.
"""

import os
import numpy as np
import cv2
import torch
import random
from PIL import Image

from .dataset import BaseDataset, resize_for_input
from dac.utils.erp_geometry import fisheye_mei_to_erp


class KITTI360Dataset(BaseDataset):
    CAM_INTRINSIC = {
        "02": torch.tensor(
            [
                [1.33632e+03, 0.000000e00, 7.16943e+02],
                [0.000000e00, 1.33578e+03, 7.05764e+02],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        ),
        "03": torch.tensor(
            [
                [1.48543e+03, 0.000000e00, 6.98883e+02],
                [0.000000e00, 1.48494e+03, 6.98145e+02],
                [0.000000e00, 0.000000e00, 1.000000e00],
            ]
        )
    }
    
    camera_params_02 = {
        "dataset": "kitti360",
        "camera_model": "MEI",
        "camera_name": "image_02",
        "image_width": 1400,
        "image_height": 1400,
        "xi": 2.2134047507854890e+00,
        "k1": 1.6798235660113681e-02,
        "k2": 1.6548773243373522e+00,
        "p1": 4.2223943394772046e-04,
        "p2": 4.2462134260997584e-04,
        "fx": 1.3363220825849971e+03,
        "fy": 1.3357883350012958e+03,
        "cx": 7.1694323510126321e+02,
        "cy": 7.0576498308221585e+02
    }
    
    camera_params_03 = {
        "dataset": "kitti360",
        "camera_model": "MEI",
        "camera_name": "image_03",
        "image_width": 1400,
        "image_height": 1400,
        "xi": 2.5535139132482758e+00,
        "k1": 4.9370396274089505e-02,
        "k2": 4.5068455478645308e+00,
        "p1": 1.3477698472982495e-03,
        "p2": -7.0340482615055284e-04,
        "fx": 1.4854388981875156e+03,
        "fy": 1.4849477411748708e+03,
        "cx": 6.9888316784030962e+02,
        "cy": 6.9814541887723055e+02
    }
    
    min_depth = 0.01
    max_depth = 80
    test_split = "kitti360_val_fisheye.txt"
    train_split = "kitti360_train_fisheye.txt"

    def __init__(
        self,
        test_mode,
        base_path,
        depth_scale=256,
        crop=None,
        is_dense=False,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        tgt_f=7.215377e02,
        undistort_f=0,
        fwd_sz=(1400, 1400),
        cano_sz=(1400, 1400),
        erp=True,
        rand_crop=False,
        load_attn_mask=False,
        visual_debug=False,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        # self.crop = crop
        self.is_dense = is_dense
        self.tgt_f = tgt_f
        self.undistort_f = undistort_f
        self.fwd_sz = fwd_sz
        # cano_sz (h, w) is the size when model is being trained. At training time, fwd_sz (h, w) should be the same as cano_sz
        self.cano_sz = cano_sz
        self.erp = erp
        self.rand_crop = rand_crop
        self.load_attn_mask = load_attn_mask
        self.visual_debug = visual_debug

        if undistort_f > 0:
            self.split_file = self.split_file[:-4] + f"_undistort_f{self.undistort_f}.txt"
        if erp:
            self.split_file = self.split_file[:-4] + "_erp.txt"

        if self.load_attn_mask:
            # Prepare the mask for fisheye images to remove the ego-car border
            self.mask_fisheye02 = (np.load(os.path.join('splits', 'kitti360', 'mask_left_fisheye.npy'))==0).astype(np.uint8)
            self.mask_fisheye03 = (np.load(os.path.join('splits', 'kitti360', 'mask_right_fisheye.npy'))==0).astype(np.uint8)
            # Convert fisheye masks to erp masks
            if self.erp:
                self.mask_fisheye02 = fisheye_mei_to_erp(self.mask_fisheye02, self.camera_params_02, self.fwd_sz)
                self.mask_fisheye03 = fisheye_mei_to_erp(self.mask_fisheye03, self.camera_params_03, self.fwd_sz)
            self.mask_fisheye02 = cv2.resize(self.mask_fisheye02, (self.fwd_sz[1], self.fwd_sz[0]), interpolation=cv2.INTER_NEAREST)
            self.mask_fisheye03 = cv2.resize(self.mask_fisheye03, (self.fwd_sz[1], self.fwd_sz[0]), interpolation=cv2.INTER_NEAREST)
        
        # if resize_im:
        #     # the dataset will be resized to match the focal length of the kitti dataset
        #     if undistort_f >0:
        #         self.height = int(1400 * tgt_f / undistort_f)
        #         self.width = int(1400 * tgt_f / undistort_f)
        #     else:
        #         self.height = int(1400 * tgt_f / self.CAM_INTRINSIC['02'][1, 1])
        #         self.width = int(1400 * tgt_f / self.CAM_INTRINSIC['02'][1, 1])
        # else:
        self.height = fwd_sz[0]
        self.width = fwd_sz[1]            

        # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

    def load_dataset(self):
        self.invalid_depth_num = 0
        with open(os.path.join('splits/kitti360', self.split_file)) as f:
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
                    
                # setup original intrinsics
                img_name = line.strip().split(" ")[0]
                img_info["image_filename"] = os.path.join(self.base_path, img_name)
                if 'image_02' in img_name:
                    img_info["camera_intrinsics"] = self.CAM_INTRINSIC['02'][:, :3].clone()
                elif 'image_03' in img_name:
                    img_info["camera_intrinsics"] = self.CAM_INTRINSIC['03'][:, :3].clone()
                
                if self.undistort_f > 0:
                    img_info["camera_intrinsics"] = torch.tensor(
                        [
                            [self.undistort_f, 0.000000e00, 700.],
                            [0.000000e00, self.undistort_f, 700.],
                            [0.000000e00, 0.000000e00, 1.000000e00],
                        ])
                elif self.erp:
                    img_info["camera_intrinsics"] = torch.tensor(
                        [
                            [1 / np.tan(np.pi/1400), 0.000000e00, 700.],
                            [0.000000e00, 1 / np.tan(np.pi/1400), 700.],
                            [0.000000e00, 0.000000e00, 1.000000e00],
                        ])
                
                # setup pred_scale_factor due to conversion to target focal length
                if self.tgt_f > 0:
                    img_info["pred_scale_factor"] = (img_info["camera_intrinsics"][0, 0] + img_info["camera_intrinsics"][1, 1]).item() / 2 / self.tgt_f
                    img_info["camera_intrinsics"][0, 0] /= img_info["pred_scale_factor"]
                    img_info["camera_intrinsics"][1, 1] /= img_info["pred_scale_factor"]
                else:
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
            Image.open(
                os.path.join(
                    # self.base_path, 
                    self.dataset[idx]["image_filename"])
            )
        ).astype(np.uint8)
        depth = None
        if not self.benchmark:
            depth = (
                np.asarray(
                    Image.open(
                        os.path.join(
                            # self.base_path,
                            self.dataset[idx]["annotation_filename_depth"],
                        )
                    )
                ).astype(np.float32)
                / self.depth_scale
            )
        info = self.dataset[idx].copy()

        info["camera_intrinsics"] = self.dataset[idx]["camera_intrinsics"].clone()
        # resizing process to fwd_sz.
        if self.erp:
            to_cano_ratio = self.cano_sz[0] / image.shape[0]
            image, depth, pad, pred_scale_factor = resize_for_input(image, depth, self.fwd_sz, info["camera_intrinsics"], self.cano_sz, to_cano_ratio)
        else:
            image, depth, pad, pred_scale_factor = resize_for_input(image, depth, self.fwd_sz, info["camera_intrinsics"], [image.shape[0], image.shape[1]], 1.0)
        info['f_align_factor'] = np.copy(info['pred_scale_factor'])
        info['pred_scale_factor'] = info['pred_scale_factor'] * pred_scale_factor
        info['pad'] = pad
        if not self.test_mode:
            depth /= info['pred_scale_factor']

        if self.load_attn_mask:
            if 'image_02' in self.dataset[idx]["image_filename"]:
                no_border_mask = self.mask_fisheye02.astype(np.float32)
            elif 'image_03' in self.dataset[idx]["image_filename"]:
                no_border_mask = self.mask_fisheye03.astype(np.float32)
            else:
                no_border_mask = None
        else:
            no_border_mask = np.ones_like(depth)

        image, gts, info = self.transform(image=image, gts={"depth": depth, 'attn_mask': no_border_mask}, info=info)

        if self.erp:
            # Attention: lat_range and long_range does not support rotation/translation augmentation in transform            
            if self.rand_crop:
                crop_height = self.fwd_sz[0] // 6
                crop_width = self.fwd_sz[1] // 2

                top = random.randint(self.fwd_sz[0] // 5 * 2, self.fwd_sz[0] // 3 * 2 - crop_height)
                left = random.randint(0, self.fwd_sz[1] - crop_width)

                image = image[:, top:top+crop_height, left:left+crop_width]       
                gts["gt"] = gts["gt"][:, top:top+crop_height, left:left+crop_width]
                gts["mask"] = gts["mask"][:, top:top+crop_height, left:left+crop_width]
                gts["attn_mask"] = gts["attn_mask"][:, top:top+crop_height, left:left+crop_width]
            
                lat_range = torch.tensor([top, top + crop_height], dtype=torch.float32) * np.pi / self.fwd_sz[0] - np.pi/2
                long_range = torch.tensor([left, left + crop_width], dtype=torch.float32) * np.pi / self.fwd_sz[1]
            else:
                lat_range = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32)
                long_range = torch.tensor([-np.pi/2, np.pi/2], dtype=torch.float32)
                
        if self.visual_debug:
            # visualize image, gts[gt], gts[attn_mask]
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow((image.permute(1, 2, 0) - image.min()) / (image.max() - image.min()))
            plt.title("Image")
            plt.subplot(2, 2, 2)
            plt.imshow(gts["gt"].squeeze())
            plt.title("Ground Truth")
            if self.erp:
                plt.subplot(2, 2, 3)
                plt.imshow(gts["attn_mask"].squeeze())
                plt.title("Attn Mask")
            plt.subplot(2, 2, 4)
            plt.imshow(gts["mask"].squeeze())
            plt.title("Valid Depth Mask")
            plt.show()
        
        output = {"image": image, "gt": gts["gt"], "mask": gts["mask"]}
        if self.test_mode:
            output["info"] = info
        if self.erp:
            output["lat_range"] = lat_range
            output["long_range"] = long_range
        # if self.load_attn_mask:
        output["attn_mask"] = gts["attn_mask"]
        
        return output

    # def get_pointcloud_mask(self, shape):
    #     if self.crop is None:
    #         return np.ones(shape)
    #     mask_height, mask_width = shape
    #     mask = np.zeros(shape)
    #     if "garg" in self.crop:
    #         mask[
    #             int(0.40810811 * mask_height) : int(0.99189189 * mask_height),
    #             int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
    #         ] = 1
    #     elif "eigen" in self.crop:
    #         mask[
    #             int(0.3324324 * mask_height) : int(0.91351351 * mask_height),
    #             int(0.0359477 * mask_width) : int(0.96405229 * mask_width),
    #         ] = 1
    #     return mask

    def preprocess_crop(self, image, gts=None, info=None):
        height_start, width_start = int(image.shape[0] - self.height), int(
            (image.shape[1] - self.width) / 2
        )
        height_end, width_end = height_start + self.height, width_start + self.width
        image = image[height_start:height_end, width_start:width_end]
        info["camera_intrinsics"][0, 2] = info["camera_intrinsics"][0, 2] - width_start
        info["camera_intrinsics"][1, 2] = info["camera_intrinsics"][1, 2] - height_start
        new_gts = {}
        if "depth" in gts:
            depth = gts["depth"]
            if depth is not None:
                height_start, width_start = int(depth.shape[0] - self.height), int(
                    (depth.shape[1] - self.width) / 2
                )
                height_end, width_end = (
                    height_start + self.height,
                    width_start + self.width,
                )
                depth = depth[height_start:height_end, width_start:width_end]
                mask = depth > self.min_depth
                if self.test_mode:
                    mask = np.logical_and(mask, depth < self.max_depth)
                    # mask = self.eval_mask(mask)
                mask = mask.astype(np.uint8)
                new_gts["gt"] = depth
                new_gts["mask"] = mask
                
        if "attn_mask" in gts:
            attn_mask = gts["attn_mask"]
            if attn_mask is not None:
                height_start, width_start = int(attn_mask.shape[0] - self.height), int(
                    (attn_mask.shape[1] - self.width) / 2
                )
                height_end, width_end = (
                    height_start + self.height,
                    width_start + self.width,
                )
                attn_mask = attn_mask[height_start:height_end, width_start:width_end]
                new_gts["attn_mask"] = attn_mask
        return image, new_gts, info

    # def eval_mask(self, valid_mask):
    #     """Do grag_crop or eigen_crop for testing"""
    #     if self.test_mode:
    #         if self.crop is not None:
    #             mask_height, mask_width = valid_mask.shape[-2:]
    #             eval_mask = np.zeros_like(valid_mask)
    #             if "garg" in self.crop:
    #                 eval_mask[
    #                     int(0.40810811 * mask_height) : int(0.99189189 * mask_height),
    #                     int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
    #                 ] = 1
    #             elif "eigen" in self.crop:
    #                 eval_mask[
    #                     int(0.3324324 * mask_height) : int(0.91351351 * mask_height),
    #                     int(0.03594771 * mask_width) : int(0.96405229 * mask_width),
    #                 ] = 1
    #         valid_mask = np.logical_and(valid_mask, eval_mask)
    #     return valid_mask
