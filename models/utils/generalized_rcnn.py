"""
实现R-CNN的框架：backbones+necks+heads+losses
"""

from collections import OrderedDict  #dict具有hash特性，是无序的，OrderDict提供了一个有序的字典对象
import torch
from torch import nn,Tensor
import warnings
from typing import Tuple,List,Dict,Optional,Union   #Optional=Union[X,None],不仅可以选择默认类型X，还可赋值为None

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
                        raise ValueError("Expected target boxes to be a tensor of shape [N,4],got{:}.".format(boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type Tensor,got{:}.".format(type(boxes)))

        original_image_sizes:List[Tuple[int,int]]=[]
        for img in images:
            val=img.shape[-2:]
            assert len(val)==2
            original_image_sizes.append((val[0],val[1]))

        images,targets=self.transform(images,targets)
        

