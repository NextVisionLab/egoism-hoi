import copy
import numpy as np
import torch
import cv2
from abc import abstractmethod
from . import transforms as T

class BaseEhoiDatasetMapper:
    def __init__(self, cfg, data_anns_sup = None, is_train = True, **kwargs):
        self._data_anns_sup = data_anns_sup
        self._scale_factor = cfg.ADDITIONAL_MODULES.ASSOCIATION_VECTOR_SCALE_FACTOR
        self._cfg = cfg
        self._masks_gt = cfg.ADDITIONAL_MODULES.USE_MASK_GT
        self._keypoints_gt = cfg.ADDITIONAL_MODULES.KEYPOINTS_GT
        if self._keypoints_gt: self._num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        self.is_train = is_train

    @abstractmethod
    def __call__(self, dataset_dict):
        pass
    
    def inference(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = cv2.imread(dataset_dict["file_name"])
        ####DATA AUGMENTATION
        transform_list = []
        image, transforms = T.apply_transform_gens(transform_list, image)
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"] = image.shape[1]    
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy())
        dataset_dict["image"] = image_t
        return dataset_dict