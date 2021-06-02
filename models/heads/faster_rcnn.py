from torch import nn
import torch.nn.functional as F



from ..utils.generalized_rcnn import GeneralizedRCNN
from ..utils.anchor import AnchorGenerator

__all__ = [
    "FasterRCNN", "fasterrcnn_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_320_fpn",
    "fasterrcnn_mobilenet_v3_large_fpn"
]


class FasterRCNN(GeneralizedRCNN):
    """
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
    """
    def __init__(self,backbone,num_classes=None,
                #transform parameters
                min_size=800,max_size=1333,
                image_mean=None,image_std=None,
                #RPN parameters
                rpn_anchor_generator=None, rpn_head=None,
                rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                rpn_nms_thresh=0.7,
                rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                rpn_score_thresh=0.0,
                # Box parameters
                box_roi_pool=None, box_head=None, box_predictor=None,
                box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                box_batch_size_per_image=512, box_positive_fraction=0.25,
                bbox_reg_weights=None):
        
        if not hasattr(backbone,"out_channels"): #hasattr函数用于判断对象是否包含对应的属性
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        
