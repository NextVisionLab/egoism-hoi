import logging
import torch
from typing import Dict, List, Optional
import numpy as np
import time
import kornia.augmentation as K
import kornia
import cv2

from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes
from detectron2.utils.custom_utils import expand_box, extract_masks_and_resize
from .rcnn import GeneralizedRCNN
from .additional_modules import *
from .ehoi_net import EhoiNet
from .build import META_ARCH_REGISTRY
from ..roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference_in_training, mask_rcnn_loss

__all__ = ["MMEhoiNetv1"]
logger = logging.getLogger(__name__)
@META_ARCH_REGISTRY.register()
class MMEhoiNetv1(EhoiNet):
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)

        ###PARAMS
        self._expand_hand_box_ratio = cfg.ADDITIONAL_MODULES.EXPAND_HAND_BOX_RATIO

        ###ADDITIONAL MODULES
        self.association_vector_regressor = AssociationVectorRegressor(cfg)

        ###ROI ALIGN CONTACT STATE CNN
        input_size_cnn_contact_state = cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE if "CONTACT_STATE_CNN_INPUT_SIZE" in cfg.ADDITIONAL_MODULES else 128
        self._roi_align_cnn_contact_state = torchvision.ops.RoIAlign((input_size_cnn_contact_state, input_size_cnn_contact_state), 1, -1)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training: return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] if "instances" in batched_inputs[0] else None
        
        ###BACKBONE
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features

        ###ROI HEAD
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposals_match, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        
        ###PREPARE GT LABELS
        self._prepare_gt_labels(proposals_match)

        ###DEPTH MODULE + LOSS
        if self._use_depth_module: 
            self._last_extracted_features["depth"], self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            if "depth_gt" in batched_inputs[0]:
                gt_depth_maps =  torch.tensor(np.array([e["depth_gt"] for e in batched_inputs])).to(self.device)
                loss_depth_estimation = self._loss_depth_f(self._depth_maps_predicted, gt_depth_maps)
            else:
                loss_depth_estimation = torch.tensor([0], dtype=torch.float32).to(self.device)

        ###MASK HEAD + LOSS
        if self._predict_mask:
            proposals_mask, _ = select_foreground_proposals(proposals_match, self._num_classes)
            features_mask = self._mask_pooler([features[f] for f in self._mask_in_features], [x.proposal_boxes for x in proposals_mask])
            pred_mask_logits = self._mask_rcnn_head.layers(features_mask)
            if self._mask_gt:
                mask_losses = {"loss_mask": mask_rcnn_loss(pred_mask_logits, proposals_mask) * self._mask_rcnn_head.loss_weight}
            proposals_match = mask_rcnn_inference_in_training(pred_mask_logits, proposals_mask)

        ###PREPARE HANDS FEATURES
        self._prepare_hands_features(batched_inputs, proposals_match)
        self._last_proposals_match = proposals_match

        ###LOSS ADDITIONAL MODULE CLASSIFICATION HAND LR
        _, loss_classification_hand_lr = self.classification_hand_lr(self._c_hands_features, self._c_gt_hands_lr) 

        ###LOSS ADDITIONAL MODULE CLASSIFICATION CONTACT STATE
        if self._contact_state_modality == "rgb":   
            _, loss_classification_contact_state = self.classification_contact_state(self._c_hands_features_padded, self._c_gt_hands_contact_state)
        elif "fusion" in self._contact_state_modality:
            _, loss_classification_contact_state = self.classification_contact_state(self._c_hands_features_cnn, self._c_hands_features_padded, self._c_gt_hands_contact_state)
        else:
            _, loss_classification_contact_state = self.classification_contact_state(self._c_hands_features_cnn, self._c_gt_hands_contact_state)

        ###LOSS ADDITIONAL MODULE REGRESSION DXDYMAGNITUDE
        indexes_contact = [i for i, x in enumerate(self._c_gt_hands_contact_state) if x == 1]
        _, loss_regression_vector = self.association_vector_regressor(self._c_hands_features_padded[indexes_contact], np.array(self._c_gt_hands_dxdymagnitude)[indexes_contact])            

        total_loss = {}
        total_loss.update(detector_losses)
        total_loss.update(proposal_losses)
        if isinstance(loss_classification_contact_state, dict): total_loss.update(loss_classification_contact_state)
        else: total_loss['loss_classification_contact_state'] =  loss_classification_contact_state
        total_loss['loss_classification_hand_lr'] =  loss_classification_hand_lr
        total_loss['loss_regression_dxdymagn'] =  loss_regression_vector
        if self._use_depth_module: total_loss['loss_depth'] = loss_depth_estimation
        if self._mask_gt: total_loss.update(mask_losses)

        return total_loss

    #INFERENZA
    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        self._last_inference_times ={k: 0 for k, v in self._last_inference_times.items()} 
        
        ###BACKBONE
        start_time = time.time()
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features
        self._last_inference_times["backbone"] = time.time() - start_time

        ###ROI HEAD
        tmp_time = time.time()
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        self._last_inference_times["roi_heads"] = time.time() - tmp_time

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        ###ADDITIONAL OUTPUT
        instances = results[0]["instances"]
        results[0]['additional_outputs'] = Instances(instances.image_size)

        ###DEPTH MODULE OUTPUT
        if self._use_depth_module: 
            tmp_time = time.time()
            self._last_extracted_features["depth"], self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            self._last_inference_times["depth.extract_features_maps"] = time.time() - tmp_time

        ###MASK HEAD
        if self._predict_mask:
            tmp_time = time.time()   
            features_mask = self._mask_pooler([features[f] for f in self._mask_in_features], [instances.pred_boxes])
            instances = self._mask_rcnn_head(features_mask, [instances])[0]
            self._last_inference_times["mask_rcnn_head"] = time.time() - tmp_time
        
        ###PREPARE HANDS FEATURES
        tmp_time = time.time()
        instances_hands = instances[instances.pred_classes == self._id_hand]
        self._prepare_hands_features_inference(batched_inputs, instances_hands)
        self._last_inference_times["data.prep.additional_modules"] = time.time() - tmp_time
    
        #### SIDE
        tmp_time = time.time()
        output_classification_side = torch.round(torch.sigmoid(self.classification_hand_lr(self._c_hands_features))).int()
        self._last_inference_times["classification_hand_lr"] = time.time() - tmp_time
        
        #### DXDYMAGN VECTOR
        tmp_time = time.time()
        output_dxdymagn = self.association_vector_regressor(self._c_hands_features_padded)
        self._last_inference_times["association_vector_regressor"] = time.time() - tmp_time

        #### CONTACT STATE
        tmp_time = time.time()   
        if self._contact_state_modality == "rgb":
            output_classification_contact = torch.round(torch.sigmoid(self.classification_contact_state(self._c_hands_features_padded))).int()
        elif "fusion" in self._contact_state_modality:
            self.scores_contact = self.classification_contact_state(self._c_hands_features_cnn, self._c_hands_features_padded)
            output_classification_contact = torch.round(self.scores_contact).int()
        else:
            self.scores_contact = torch.sigmoid(self.classification_contact_state(self._c_hands_features_cnn))
            output_classification_contact = torch.round(self.scores_contact).int()
        self._last_inference_times["classification_contact_state"] = time.time() - tmp_time

        results[0]['additional_outputs'].set("boxes", instances_hands.pred_boxes.tensor.reshape(-1, 4).detach().cpu())
        results[0]['additional_outputs'].set("sides", output_classification_side.detach().cpu())
        results[0]['additional_outputs'].set("scores", instances_hands.scores.detach().cpu())
        results[0]['additional_outputs'].set("contact_states", output_classification_contact.detach().cpu())
        results[0]['additional_outputs'].set("dxdymagn_hand", output_dxdymagn.reshape(-1, 3).detach().cpu())
        if self._use_depth_module: results[0]['depth_map'] = self._depth_maps_predicted

        _total = round(sum(self._last_inference_times.values()) * 1000, 2)
        self._last_inference_times = {k: round(v * 1000, 2) for k, v in self._last_inference_times.items()}
        self._last_inference_times["total"] = _total
        self._last_instances_hands = instances_hands
        return results

    ###PREPARE GT LABELS
    def _prepare_gt_labels(self, proposals_match):
        self._c_gt_hands_lr, self._c_gt_hands_contact_state, self._c_gt_hands_dxdymagnitude = [], [], []
        for batch_proposal in proposals_match:
            batch_proposal_hands = batch_proposal[batch_proposal.gt_classes == self._id_hand]
            for idx_proposal in range(len(batch_proposal_hands)): 
                self._c_gt_hands_lr.append(batch_proposal_hands[idx_proposal].gt_sides.item())
                self._c_gt_hands_contact_state.append(batch_proposal_hands[idx_proposal].gt_contact_states.item())
                self._c_gt_hands_dxdymagnitude.append(batch_proposal_hands[idx_proposal].gt_dxdymagn_hands.detach().cpu().numpy()[0])

    ###PREPARE HANDS FEATURES
    def _prepare_hands_features(self, batched_inputs, proposals_match):
        image_width, image_height = batched_inputs[0]['width'], batched_inputs[0]['height']
        boxes, boxes_padded, boxes_padded_depth = [], [], []
        ###FEATURES FROM RESNET101
        for idx_batch, batch_proposal in enumerate(proposals_match):
            batch_proposal_hands = batch_proposal[batch_proposal.gt_classes == self._id_hand]
            boxes.append(batch_proposal_hands.proposal_boxes)
            boxes_padded.append(Boxes(expand_box(batch_proposal_hands.proposal_boxes.tensor.detach().clone(), image_width, image_height, ratio = self._expand_hand_box_ratio)))
            depth_width, depth_height = batched_inputs[idx_batch]["image_for_depth_module"].shape[2], batched_inputs[idx_batch]["image_for_depth_module"].shape[1]
            tmp_boxes = Boxes(boxes_padded[-1].tensor.detach().clone())
            tmp_boxes.scale(scale_x=(depth_width / image_width), scale_y=(depth_height / image_height))
            boxes_padded_depth.append(tmp_boxes)
        rois = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], boxes)      
        self._c_hands_features = self.roi_heads.box_head(rois) 
        rois_padded = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], boxes_padded)      
        self._c_hands_features_padded = self.roi_heads.box_head(rois_padded)
        
        self._last_boxes_padded_rescaled = [box.tensor for box in boxes_padded_depth]
        ###FEATURES FOR CNN CONTACT STATE MODULE
        if self._contact_state_modality != "rgb":
            rgb_images = torch.tensor(np.array([b["image_for_depth_module"] for b in batched_inputs])).to(self.device)
            rgb_images = kornia.color.bgr_to_rgb(rgb_images)
            if not "depth" in self._contact_state_modality:
                c_roi = self._roi_align_cnn_contact_state(rgb_images, [box.tensor for box in boxes_padded_depth])
            else:
                depths = torch.divide(self._depth_maps_predicted.detach().unsqueeze(1), 255)
                rgbd_images = torch.cat((rgb_images, depths), dim=1)
                c_roi = self._roi_align_cnn_contact_state(rgbd_images, [box.tensor for box in boxes_padded_depth])    
            if "mask" in self._contact_state_modality:
                masks = extract_masks_and_resize(proposals_match, batched_inputs[0]["image_for_depth_module"].shape[1:], self._id_hand)
                if len(masks):
                    masks = torch.cat(masks).unsqueeze(1)
                    boxes_cat = torch.cat((torch.arange(0, masks.shape[0]).unsqueeze(1).to(self.device), torch.cat([box.tensor for box in boxes_padded_depth])), dim = 1)
                    self._last_masks_roi = self._roi_align_cnn_contact_state(masks, boxes_cat)
                    c_roi = torch.cat((c_roi, self._last_masks_roi), dim=1)
                else:
                    self._last_masks_roi = torch.zeros(c_roi.shape)[:,:1].to(self.device)
                    c_roi = torch.cat((c_roi, self._last_masks_roi), dim=1)
            self._c_hands_features_cnn = c_roi

    def _prepare_hands_features_inference(self, batched_inputs, instances_hands):
        image_width, image_height = batched_inputs[0]['width'], batched_inputs[0]['height']
        rois = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], [instances_hands.pred_boxes])
        self._c_hands_features = torch.squeeze(self.roi_heads.box_head(rois))
        boxes_padded = Boxes(expand_box(instances_hands.pred_boxes.tensor.detach().clone(), image_width, image_height, ratio = self._expand_hand_box_ratio))
        rois_padded = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], [boxes_padded])
        self._c_hands_features_padded = torch.squeeze(self.roi_heads.box_head(rois_padded))

        if self._contact_state_modality != "rgb":
            boxes_padded_rescaled = Boxes(boxes_padded.tensor.detach().clone())
            boxes_padded_rescaled.scale(scale_x=batched_inputs[0]["image_for_depth_module"].shape[2] / batched_inputs[0]["image"].shape[2] , scale_y=batched_inputs[0]["image_for_depth_module"].shape[1] / batched_inputs[0]["image"].shape[1])     
            self._last_boxes_padded_rescaled = [boxes_padded_rescaled.tensor]
            rgb_images = torch.tensor(np.array([b["image_for_depth_module"] for b in batched_inputs])).to(self.device)
            rgb_images = kornia.color.bgr_to_rgb(rgb_images)
            if not "depth" in self._contact_state_modality:
                c_roi = self._roi_align_cnn_contact_state(rgb_images, [boxes_padded_rescaled.tensor])
            else:
                depths = torch.divide(self._depth_maps_predicted.unsqueeze(1), 255)
                rgbd_images = torch.cat((rgb_images, depths), dim=1)
                c_roi = self._roi_align_cnn_contact_state(rgbd_images, [boxes_padded_rescaled.tensor])
            if "mask" in self._contact_state_modality:
                masks = extract_masks_and_resize([instances_hands], batched_inputs[0]["image_for_depth_module"].shape[1:], self._id_hand)
                if len(masks):
                    masks = torch.cat(masks).unsqueeze(1)
                    boxes_cat = torch.cat((torch.arange(0, masks.shape[0]).unsqueeze(1).to(self.device), boxes_padded_rescaled.tensor), dim = 1)
                    self._last_masks_roi  = self._roi_align_cnn_contact_state(masks, boxes_cat)
                    c_roi = torch.cat((c_roi, self._last_masks_roi), dim=1)
                else:
                    self._last_masks_roi = torch.zeros(c_roi.shape)[:,:1].to(self.device)
                    c_roi = torch.cat((c_roi, self._last_masks_roi), dim=1)
            self._c_hands_features_cnn = c_roi