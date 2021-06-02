import math
import typing
import torch
from torch import nn,Tensor

from typing import List,Optional

from torch.nn.modules import instancenorm
from .image_list import ImageList

class AnchorGenerator(nn.Module):
    """
    为图片生成anchors的模块
    假设 aspect ration=height/width

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """
    __annotations__={
        "cell_anchors":List[torch.Tensor],
    }

    def __init__(self,sizes=((128,256,512),),aspect_ratios=((0.5,1.0,2.0)),):
        super(AnchorGenerator,self).__init__()
        
        if not isinstance(sizes[0],(list,tuple)):
            sizes=tuple((s,) for s in sizes)
        
        if not isinstance(aspect_ratios[0],(list,tuple)):
            aspect_ratios=(aspect_ratios,)*len(sizes)

        assert len(sizes)==len(aspect_ratios)

        self.sizes=sizes
        self.aspect_ratios=aspect_ratios
        self.cell_anchors=[self.generate_anchors(size,aspect_ratio) for 
                            size,aspect_ratio in zip(sizes,aspect_ratios)]
        


    def generate_anchors(self, scales: List[int], aspect_ratios: List[float], dtype: torch.dtype = torch.float32,
                         device: torch.device = torch.device("cpu")):
        scales=torch.as_tensor(scales,dtype=dtype,device=device)
        aspect_ratios=torch.as_tensor(aspect_ratios,dtype=dtype,device=device)
        h_ratios=torch.sqrt(aspect_ratios)     #w*h=1;r=h/w
        w_ratios=1/h_ratios

        ws=(w_ratios[:,None]*scales[None,:]).view(-1)  #w.shape=[3],w[:,None].shape=[3,1],s.shape=[3],s[None,；].shape=[1,3]
        hs=(h_ratios[:,None]*scales[None,:]).view(-1)  #ws.shape=[9]

        base_anchors=torch.stack([-ws,-hs,ws,hs],dim=1)/2  #a.shape=[3,4] b.shape=[3,4] torch.cat([a,b],dim=1).shape=[3,8]
                                                           #torch.stack([a,b],dim=1).shape=[3,2,4],base_anchors.shape=[9,4]
        return base_anchors.round()  #round()函数用于逐元素查找最接近的整数

    def set_cell_anchors(self,dtype:torch.dtype,device:torch.device):
        self.cell_anchors=[cell_anchor.to(dtype=dtype,device=device) for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self):
        return [len(s)*len(a) for s,a in zip(self.sizes,self.aspect_ratios)]


    def grid_anchors(self,grid_sizes:List[List[int]],strides:List[List[Tensor]]) ->List[Tensor]:
        anchors=[]
        cell_anchors=self.cell_anchors
        assert cell_anchors is not None

        if not (len(grid_sizes)==len(strides)==len(cell_anchors)):
            raise ValueError("Anchors should be Tuple[Tuple[int]] because each feature "
                             "map could potentially have different sizes and aspect ratios. "
                             "There needs to be a match between the number of "
                             "feature maps passed and the number of sizes / aspect ratios specified.")
        
        for size,stride,base_anchors in zip(grid_sizes,strides,cell_anchors):
            grid_height,grid_width=size
            stride_height,stride_width=stride
            device=base_anchors.device

            #对于输出anchor，计算[x_center,y_center,x_center,y_center]
            shifts_x = torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            ) * stride_width
            shifts_y = torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            ) * stride_height
            #torch.meshgrid（）的功能是生成网格，可以用于生成坐标
            #a = torch.tensor([1, 2, 3, 4])
            # print(a)
            # b = torch.tensor([4, 5, 6])
            # print(b)
            # x, y = torch.meshgrid(a, b)
            # print(x)
            # print(y)
            # tensor([1, 2, 3, 4])
            # tensor([4, 5, 6])
            # tensor([[1, 1, 1],
            #         [2, 2, 2],
            #         [3, 3, 3],
            #         [4, 4, 4]])
            # tensor([[4, 5, 6],
            #         [4, 5, 6],
            #         [4, 5, 6],
            #         [4, 5, 6]])
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x) 
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)#shifts.shape=[N,4]
            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )#shape [N,1,4]+[1,9,4]->[N,9,4]->[9*n,4]

        return anchors

    def forward(self,image_list:ImageList,feature_maps:List[Tensor]) ->List[Tensor]:
        grid_sizes=[feature_map.shape[-2:] for feature_map in feature_maps]
        image_size=image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]
        self.set_cell_anchors(dtype,device)
        anchors_over_all_feature_maps=self.grid_anchors(grid_sizes,strides)
        anchors:List[List[torch.Tensor]]=[]
        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors


    


