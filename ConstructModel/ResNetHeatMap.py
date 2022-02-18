from typing import Type, Any, Callable, Union, List, Optional
from sympy import convolution

import torch
import torch.nn as nn
from torch import Tensor
import timm
import torch
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
import matplotlib.pyplot as plt


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print('one:', x.shape)
        x = self.maxpool(x)

        x = self.layer1(x)
        #print('two:', x.shape)
        x = self.layer2(x)
        #print('three:', x.shape)
        x = self.layer3(x)
        #print('four:', x.shape)
        x = self.layer4(x)
        #print('five:', x.shape)
        x = self.avgpool(x)
        #print('six:', x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
model = ResNet(BasicBlock, [2, 2, 2, 2])
#print(model)
model.load_state_dict(torch.load('./ResNet.pth'))
pretrained_model = model
class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model              #用于储存模型
        self.target_layer = target_layer#目标层的名称
        self.gradients = None           #最终的梯度图

    def save_gradient(self, grad):
        self.gradients = grad           #用于保存目标特征图的梯度（因为pytorch只保存输出，相对于输入层的梯度
                                        #，中间隐藏层的梯度将会被丢弃，用来节省内存。如果想要保存中间梯度，必须
                                        #使用register_hook配合对应的保存函数使用，这里这个函数就是对应的保存
                                        #函数其含义是将梯度图保存到变量self.gradients中，关于register_hook
                                        #的使用方法我会在开一个专门的专题，这里不再详述
    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for name,layer in pretrained_model._modules.items():
            #print('name and type(name)',name, type(name))      
            if name == "fc":
                break
            x = layer(x)
            print('name:', name, 'name == self.target_layer', name == self.target_layer, 'self.target_layer:', self.target_layer)
            if name == self.target_layer:  
                conv_output = x                      #将目标特征图保存到conv_output中            
                x.register_hook(self.save_gradient)  #设置将目标特征图的梯度保存到self.gradients中               
        print('conv_output:', type(conv_output), 'x:',type(x))
        return conv_output, x                        #x为最后一层特征图的结果

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer) #用于提取特征图与梯度图
    def generate_cam(self, input_image, target_class=None):
        #1.1 前向传播，计算出目标类的最终输出值model_output，以及目标层的特征图的输出conv_output
        conv_output, model_output = self.extractor.forward_pass(input_image)
        print('type of conv_output:',type(conv_output),'type of model_output', type(model_output))
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        #one hot编码，令目标类置1
        print('model_output.shape', model_output.shape)
        print('conv_output.shape', conv_output.shape)
        print('model_output.size()[-1]',model_output.size()[-1])
        
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        print(one_hot_output.shape)
        one_hot_output[0][target_class] = 1
        # 步骤1.2 反向传播， 获取目标类相对于目标层各特征图的梯度
        target = conv_output.data.numpy()[0]
        # 步骤1.2.1 清零梯度：model.zero_grad()
        self.model.zero_grad()
        # 步骤1.2.2 计算反向传播
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # 步骤1.2.3 获取目标层各特征图的梯度
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # 步骤2.1 对每张梯度图求均值，作为与其对应的特征图的权重
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # 初始化热力图
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # 步骤2.2 计算各特征图的加权值
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        #步骤2.3 对热力图进行后处理，即将结果变换到0~255
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam
    
if __name__ == '__main__':
    
    print(model)
    c = GradCam(model, 'bn1')
    print(c)
    config = resolve_data_config({}, model=model)
    print('transform config:', config)
    transform = create_transform(**config)
    img = Image.open('C:\\Users\\Dawson\\Desktop\\1.jpg').convert('RGB')
    tensor = transform(img).unsqueeze(0)
    cam = c.generate_cam(tensor)
    print(cam)
    fig, ax = plt.subplots()
    hsic_matrix = cam
    im = ax.imshow(hsic_matrix, origin='lower')
    ax.set_xlabel("y", fontsize=15)
    ax.set_ylabel('x', fontsize=15)
    ax.set_title("CKA between ViT and ResMLP", fontsize=18)
    ax.grid(False) 
    
    plt.tight_layout()
    #plt.savefig('/content/drive/My Drive/data/ViT_ResMLP.jpg')
    plt.show()