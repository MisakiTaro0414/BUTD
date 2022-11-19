from pathlib import Path
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, Instances
from torchvision.ops import nms


def fast_rcnn_inference_single_image(boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image):

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    
    # Select max scores
    max_scores, max_classes = scores.max(1)       # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs).cuda() * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]     # Select max boxes according to the max scores.
    
    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = max_classes[keep]
    
    return result, keep

class FasterRCNNBottomUp:
    def __init__(self, cfg_file):
        self.Boxes = 36               
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_file)
        self.predictor = DefaultPredictor(self.cfg)
        
    def __call__(self, image):
        return self.extract(image=image)   
    
    def extract(self, image):
        # Run Backbone Res1-Res4
        with torch.no_grad():
            img_h, img_w = image.shape[:2]
            image = self.predictor.transform_gen.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": img_h, "width": img_w}]
            images = self.predictor.model.preprocess_image(inputs)
            features = self.predictor.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = self.predictor.model.proposal_generator(images, features, None)

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.predictor.model.roi_heads.in_features]
            box_features = self.predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # (sum_proposals, 2048), pooled to 1x1

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
            rcnn_outputs = FastRCNNOutputs(
                self.predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.predictor.model.roi_heads.smooth_l1_beta,
            )

            # Fixed-number NMS
            instances_list, ids_list = [], []
            probs_list = rcnn_outputs.predict_probs()
            boxes_list = rcnn_outputs.predict_boxes()

            for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
                for nms_thresh in np.arange(0.3, 1.0, 0.1):
                    instances, ids = fast_rcnn_inference_single_image(
                        boxes, probs, image_size, 
                        score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.Boxes
                    )
                    if len(ids) >= self.Boxes:
                        break
                instances_list.append(instances)
                ids_list.append(ids)
            
            roi_features = feature_pooled[ids].detach()
            return roi_features

            # Post processing for features
            
            # features_list = feature_pooled.split(rcnn_outputs.num_preds_per_image) # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
            # roi_features_list = []
            # for ids, features in zip(ids_list, features_list):
            #     roi_features_list.append(features[ids].detach())
            
            # outputs.append(roi_features_list)

   
    