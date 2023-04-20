# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import numpy as np
import torch
import cv2
from detectron2.modeling.meta_arch.MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from detectron2.modeling.meta_arch.MiDaS import utils as midas_utils
from torchvision.transforms import Compose
from typing import overload
import numpy as np
from functools import singledispatch

class SimpleMapper:
    def __init__(self, cfg): 
        self._input_size = (cfg.UTILS.TARGET_SHAPE_W, cfg.UTILS.TARGET_SHAPE_H)
        self._use_depth_module = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE
        if self._use_depth_module:
            self.transform = Compose(
                [
                    Resize(
                        cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_W,
                        cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_H,
                        resize_target=True,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method=cfg.ADDITIONAL_MODULES.DEPTH_MODULE.RESIZE_MODE,
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    PrepareForNet(),
                ]
            )

    def __call__(self, image):
        if type(image) == np.ndarray: image_copy = image.copy()
        elif type(image) == str: image_copy = cv2.imread(image)
        else: assert False
        dataset_dict = {}
        image_copy = cv2.resize(image_copy, self._input_size)
        dataset_dict["width"] = image_copy.shape[1]
        dataset_dict["height"] = image_copy.shape[0]
        dataset_dict["image"] = torch.from_numpy(image_copy.transpose(2, 0, 1).copy())
        
        if self._use_depth_module:
            img_midas = cv2.cvtColor(image_copy.copy(), cv2.COLOR_BGR2RGB) / 255.0
            img_input = self.transform({"image": img_midas})["image"]
            dataset_dict["image_for_depth_module"] = img_input

        return dataset_dict