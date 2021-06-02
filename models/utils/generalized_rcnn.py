"""
实现R-CNN的框架：数据异常检查+backbones+necks+heads+losses
"""

from collections import OrderedDict  #dict具有hash特性，是无序的，OrderDict提供了一个有序的字典对象
import torch
from torch import nn,Tensor
import warnings
from typing import Tuple,List,Dict,Optional,Union  #Optional=Union[X,None],不仅可以选择默认类型X，还可赋值为None

class GeneralizedRCNN(nn.Module):
    """
    生成R-CNN的主类
    Args:
        backbone(nn.Module):
        rpn(nn.Module):
        roi_heads(nn.Module):通过features+proposal(RPN)去计算detections/masks
        transform(nn.Module):执行输入到进入网络的数据转换
    """
    def __init__(self,backbone:nn.Module,rpn:nn.Module,roi_heads:nn.Module,transform:nn.Module):
        super(GeneralizedRCNN,self).__init__() #首先找到GeneralizedRCNN的父类（nn.Module），然后把类GeneralizedRCNN的对象self转换为nn.Module的对象，
                                               #然后“被转换”的nn.Module对象调用自己的__init__函数。总而言之，继承了nn.Module的__inint__
        self.transform=transform
        self.backbone=backbone
        self.rpn=rpn
        self.roi_heads=roi_heads
        self._has_warned=False #仅在torchscript mode中使用
    
    def forward(self,images:List[Tensor],targets:Optional[List[Dict[str,Tensor]]]=None)->List[Dict[str,Tensor]]:
        """
        Args:
            images:
            targets:在images中的真实框（可选择）
        Returns:
            List[BoxList] or List[Dict[Tensor]]:
                训练时，返回包含损失值的Dict[Tensor]
                测试时，返回List[BoxList],可增加socres,labels,mask
        """
        if self.training and targets is None:
            raise ValueError("In traing mode,targets should be passed")  #爆出异常，一旦执行该语句，后面的语句都将不执行
        if self.training:
            assert targets is not None  #assert检测后面的条件是否符合，不符合就终止程序
            for target in targets:
                boxes=target["boxes"]
                if isinstance(boxes,torch.Tensor):
                    if len(boxes.shape) !=2 or boxes.shape[-1] !=4:
                        raise ValueError("Expected target boxes to be a tensor of shape [N,4],got{}.".format(boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type Tensor,got{}.".format(type(boxes)))

        original_image_sizes:List[Tuple[int,int]]=[]
        for img in images:
            val=img.shape[-2:]
            assert len(val)==2
            original_image_sizes.append((val[0],val[1]))

        images,targets=self.transform(images,targets)

        #检查boxes的数据问题：x1<x2,y1<y2
        if targets is not None:
            for target_idx,target in enumerate(targets):
                boxes=targets["boxes"]
                degenerate_boxes=boxes[:,2:]<=boxes[:,:2]   #如果数据错误 x2<=x1 或则y2<=y1 则记录为true
                if degenerate_boxes.any():     #any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True
                    #打印出第一个问题数据
                    bb_idx=torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb:List[float]=boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))
        
        features=self.backbone(images.tensors)
        if isinstance(features,torch.Tensor):
            features=OrderedDict([("0",features)])
        proposals,proposal_losses=self.rpn(images,features,targets)
        detections,detector_losses=self.roi_heads(features,proposals,images.image_sizes,original_image_sizes)
        detections=self.transform.postprocess(detections,images.image_sizes,original_image_sizes)

        losses={}#如果被更新的字典中己包含对应的键值对，那么原 value 会被覆盖；如果被更新的字典中不包含对应的键值对，则该键值对被添加进去
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if not self._has_warned:
            warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
            self._has_warned = True
        return losses,detections




        

