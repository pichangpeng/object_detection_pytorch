from typing import Union
import torch
import torch.nn as nn
from  torch.utils.model_zoo import load_url
from typing import Union,List,Dict,Any

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 
    'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19','make_layers']

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfgs:Dict[str,List[Union[str,int]]]={
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self,features:nn.Module,num_classes:int=1000,init_weights:bool=True) ->None:
        super(VGG,self).__init__()
        self.features=features
        self.avgpool=nn.AdaptiveAvgPool2d((7,7))#通过该函数就可以不管输入图片的大小尺寸，都可以将其压缩到标准的尺度空间
        self.classifier=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),#如y=x+1,x=y，这样会内存需要存储变量y;直接x=x+1可以节省内存空间
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,num_classes),
        )
        if init_weights:
            self._initialize_weights()

    
    def forward(self,x:torch.Tensor) ->torch.Tensor:
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x


    def _initialize_weights(self) ->None:
        for m in self.modules(): #self.children()只包括网络模块的第一代儿子模块，而self.modules()包括网络模块的自己本身和所有后代模块
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)


def make_layers(cfg:List[Union[str,int]],batch_norm:bool=False) ->nn.Sequential:
    layers: List[nn.Module]=[]
    in_channels =3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        elif type(v)==int:
            conv2d=nn.Conv2d(in_channels,v,kernel_size=3,padding=1)
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))  #batchNormzd中num_features为输入batch中图像的channle数（按每一个channle来做归一化）
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(inplace=True))
            in_channels=v
    return nn.Sequential(*layers)#*表示将任意多个参数以元组的形式传入，**表示将参数以字典的形式传入
                                #从nn.Sequential的定义来看，输入要么是orderdict,要么是一系列的模型，遇到list，必须用*号进行转化


def _vgg(name:str,cfg:str,batch_norm:bool,pretrained:bool,progress:bool,model_dir:str="../../data/datasets",**kwargs:Any) ->VGG:
    if pretrained:
        kwargs["init_weights"]=False
    model=VGG(make_layers(cfgs[cfg],batch_norm=batch_norm),**kwargs)
    if pretrained:
        state_dict=load_url(url=model_urls[name],model_dir=model_dir,progress=progress)#progress表示是否出现进度条
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained:bool=False,progress:bool=True,**kwargs:Any) ->VGG:
    """
    Args:
        pretrained(bool):是否加载预训练参数
        progress(bool):是否显示下载数据的进度条
    Return:
        返回VGG模型
    """
    return _vgg("vgg11","A",False,pretrained,progress,**kwargs)


def vgg11_bn(pretrained:bool=False,progress:bool=True,**kwargs:Any) ->VGG:
    """
    Args:
        pretrained(bool):是否加载预训练参数
        progress(bool):是否显示下载数据的进度条
    Return:
        返回VGG模型
    """
    return _vgg("vgg11_bn","A",True,pretrained,progress,**kwargs)
    

def vgg13(pretrained:bool=False,progress:bool=True,**kwargs:Any) ->VGG:
    """
    Args:
        pretrained(bool):是否加载预训练参数
        progress(bool):是否显示下载数据的进度条
    Return:
        返回VGG模型
    """
    return _vgg("vgg13","B",False,pretrained,progress,**kwargs)


def vgg13_bn(pretrained:bool=False,progress:bool=True,**kwargs:Any) ->VGG:
    """
    Args:
        pretrained(bool):是否加载预训练参数
        progress(bool):是否显示下载数据的进度条
    Return:
        返回VGG模型
    """
    return _vgg("vgg13_bn","D",True,pretrained,progress,**kwargs)
    

def vgg16(pretrained:bool=False,progress:bool=True,**kwargs:Any) ->VGG:
    """
    Args:
        pretrained(bool):是否加载预训练参数
        progress(bool):是否显示下载数据的进度条
    Return:
        返回VGG模型
    """
    return _vgg("vgg16","D",False,pretrained,progress,**kwargs)


def vgg16_bn(pretrained:bool=False,progress:bool=True,**kwargs:Any) ->VGG:
    """
    Args:
        pretrained(bool):是否加载预训练参数
        progress(bool):是否显示下载数据的进度条
    Return:
        返回VGG模型
    """
    return _vgg("vgg16_bn","D",True,pretrained,progress,**kwargs)


def vgg19(pretrained:bool=False,progress:bool=True,**kwargs:Any) ->VGG:
    """
    Args:
        pretrained(bool):是否加载预训练参数
        progress(bool):是否显示下载数据的进度条
    Return:
        返回VGG模型
    """
    return _vgg("vgg19","E",False,pretrained,progress,**kwargs)


def vgg19_bn(pretrained:bool=False,progress:bool=True,**kwargs:Any) ->VGG:
    """
    Args:
        pretrained(bool):是否加载预训练参数
        progress(bool):是否显示下载数据的进度条
    Return:
        返回VGG模型
    """
    return _vgg("vgg19_bn","E",True,pretrained,progress,**kwargs)