import logging
from abc import abstractmethod
import torchvision

from detectron2.modeling.roi_heads.mask_head import build_mask_head
from .rcnn import GeneralizedRCNN
from .build import META_ARCH_REGISTRY
from .additional_modules import *
from ..roi_heads.keypoint_head import build_keypoint_head
from ..poolers import ROIPooler
from detectron2.layers import ShapeSpec
from .MiDaS.midas.midas_loss import ScaleAndShiftInvariantLoss
from .MiDaS.midas.midas_loss_simple import DepthLoss

__all__ = ["EhoiNet"]
logger = logging.getLogger(__name__)
@META_ARCH_REGISTRY.register()
class EhoiNet(GeneralizedRCNN):
    def __init__(self, cfg, metadata):
        GeneralizedRCNN_modules = GeneralizedRCNN.from_config(cfg)
        super().__init__(
            backbone = GeneralizedRCNN_modules["backbone"],
            proposal_generator = GeneralizedRCNN_modules["proposal_generator"],
            roi_heads = GeneralizedRCNN_modules["roi_heads"],
            pixel_mean = GeneralizedRCNN_modules["pixel_mean"],
            pixel_std = GeneralizedRCNN_modules["pixel_std"],
            input_format = GeneralizedRCNN_modules["input_format"],
            vis_period = GeneralizedRCNN_modules["vis_period"])

        ###ATTRIBUTI
        thing_classes = metadata.as_dict()["thing_classes"]
        self._thing_classes = thing_classes
        self._id_hand = thing_classes.index("hand") if "hand" in thing_classes else thing_classes.index("mano")
        self._num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self._contact_state_modality = cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY
        self._predict_mask = cfg.ADDITIONAL_MODULES.USE_MASK
        self._mask_gt = cfg.ADDITIONAL_MODULES.USE_MASK_GT
        self._predict_keypoints = cfg.ADDITIONAL_MODULES.KEYPOINTS
        self._keypoints_gt = cfg.ADDITIONAL_MODULES.KEYPOINTS_GT
        self._use_depth_module = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE

        ###UTILS
        self._last_extracted_features = {}
        self._last_inference_times = {}

        ###ADDITIONAL MODULES
        self.classification_hand_lr = SideLRClassificationModule(cfg)
        self.classification_contact_state = self.build_contact_state_head(cfg)
        if self._predict_mask: self.build_mask_module(cfg)

        ###DEPTH MODULE
        if self._use_depth_module: 
            self.depth_module = DepthModule(cfg)
            self._loss_depth_f = DepthLoss()

        if self._predict_keypoints: 
            raise NotImplementedError
            #self.build_keypoint_module(cfg)
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
        
    @abstractmethod
    def inference(self, *args, **kwargs):
        pass
    
    ###BUILD KEYPOINT MODULE
    def build_keypoint_module(self, cfg):
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        input_shape       = self.backbone.output_shape()
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        in_channels       = [input_shape[f].channels for f in in_features][0]
        self._keypoint_in_features = in_features
        self._keypoint_pooler = ROIPooler(output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio, pooler_type=pooler_type,) if pooler_type else None
        if pooler_type: shape = ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        else: shape = {f: input_shape[f] for f in in_features}
        self._keypoint_head = build_keypoint_head(cfg, shape)

    ###BUILD CONTACT STATE HEAD
    def build_contact_state_head(self, cfg):
        if self._contact_state_modality == "rgb": return ContactStateRGBClassificationModule(cfg)
        if self._contact_state_modality == "cnn_rgb": return ContactStateCNNClassificationModule(cfg, n_channels=3)
        if self._contact_state_modality == "depth": return ContactStateCNNClassificationModule(cfg, n_channels=1, use_pretrain_first_layer=False)
        if self._contact_state_modality == "mask": return ContactStateCNNClassificationModule(cfg, n_channels=1, use_pretrain_first_layer=False)
        if self._contact_state_modality == "rgb+depth": return ContactStateCNNClassificationModule(cfg, n_channels=4)
        if self._contact_state_modality == "mask+rgb": return ContactStateCNNClassificationModule(cfg, n_channels=4)
        if self._contact_state_modality == "mask+depth": return ContactStateCNNClassificationModule(cfg, n_channels=2, use_pretrain_first_layer=False)
        if self._contact_state_modality == "mask+rgb+depth": return ContactStateCNNClassificationModule(cfg, n_channels=5)
        if self._contact_state_modality == "mask+rgb+depth+fusion": return ContactStateFusionClassificationModule(cfg, n_channels=5)
        if self._contact_state_modality == "mask+rgb+fusion": return ContactStateFusionClassificationModule(cfg, n_channels=4)
        if self._contact_state_modality == "rgb+depth+fusion": return ContactStateFusionClassificationModule(cfg, n_channels=4)
        if self._contact_state_modality == "rgb+fusion": return ContactStateFusionClassificationModule(cfg, n_channels=3)

        assert False, "Unknown Modality."

    ###BUILD MASK MODULE
    def build_mask_module(self, cfg):
        in_features          = cfg.MODEL.ROI_HEADS.IN_FEATURES
        input_shape          = self.backbone.output_shape()
        pooler_resolution    = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_type          = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        pooler_scales        = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio       = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        in_channels          = [input_shape[f].channels for f in in_features][0]
        self._mask_in_features =  in_features            
        self._mask_pooler = ROIPooler(output_size=pooler_resolution, scales=pooler_scales, sampling_ratio=sampling_ratio, pooler_type=pooler_type,)
        if pooler_type: shape = ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        else: shape = {f: input_shape[f] for f in in_features}
        self._mask_rcnn_head = build_mask_head(cfg, shape)

    ###EXTRACT FEATURES MAPS
    def extract_features_maps(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        self._last_extracted_features["rgb"] = self.backbone(images.tensor)
        if self._use_depth_module: 
            self._last_extracted_features["depth"] = self.depth_module.extract_features_maps(batched_inputs)[0]

    ###PREPARE GT LABELS
    @abstractmethod
    def _prepare_gt_labels(self, proposals_match):
        pass

    ###PREPARE HANDS FEATURES
    @abstractmethod
    def _prepare_hands_features(self, *args, **kwargs):
        pass