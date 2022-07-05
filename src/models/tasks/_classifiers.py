import logging
import torchvision
from torchvision.models import inception, resnet, mobilenet
import timm

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class _ViewAs3D(nn.Module):
    def __init__(self):
        super(_ViewAs3D, self).__init__()
    
    def forward(self, x):
        b, c, h, w = x.size()
        return x.view(b, 1, c, h, w)


class _ViewAs2D(nn.Module):
    def __init__(self):
        super(_ViewAs2D, self).__init__()
    
    def forward(self, x):
        b, _, c, h, w = x.size()
        return x.view(b, c, h, w)


class _UnsqueezeAs2D(nn.Module):
    def __init__(self):
        super(_UnsqueezeAs2D, self).__init__()
    
    def forward(self, x):
        b, c = x.size()
        return x.view(b, c, 1, 1)


class ViTAge(nn.Module):
    def __init__(self, channels_org, num_classes, pretrained=False, consensus=True, **kwargs):
        super(ViTAge, self).__init__()
        self._base_model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)

        fc_weights = self._base_model.head.weight
        fc_bias = self._base_model.head.bias
        
        self._base_model.head = nn.Sequential(_UnsqueezeAs2D(),  nn.Conv2d(fc_weights.size(1), num_classes, kernel_size=1, stride=1))

        if num_classes == fc_weights.size(0):
            self._base_model.head[1].weight = nn.Parameter(fc_weights.reshape(num_classes, fc_weights.size(1), 1, 1))
            self._base_model.head[1].bias = nn.Parameter(fc_bias)

        if consensus:
            self._consensus = nn.Sequential(_ViewAs3D(), nn.AdaptiveMaxPool3d(output_size=(num_classes, 1, 1)), _ViewAs2D())
        else:
            self._consensus = nn.Identity()      

        # Fix the weights of the base model when pretrained is True
        if pretrained:
            for param in self._base_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self._base_model(x)
        x = self._consensus(x)
        return x
    

class _InceptionAuxAge(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(_InceptionAuxAge, self).__init__()
        if conv_block is None:
            conv_block = inception.BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc = nn.Conv2d(in_channels = 768, out_channels=num_classes, kernel_size=1, stride=1)
        self.fc.stddev = 0.001  # type: ignore[assignment]

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 5 x 5
        x = self.fc(x)
        # N x num_classes x H' x W'
        return x


class InceptionV3Age(nn.Module):
    def __init__(self, channels_org, num_classes, pretrained=False, aux_logits=True, consensus=True, **kwargs):
        super(InceptionV3Age, self).__init__()
        self._base_model = inception.inception_v3(pretrained=pretrained, progress=False, transform_input=False, aux_logits=aux_logits, init_weights=not pretrained)

        # Fix the weights of the base model when pretrained is True
        if pretrained:
            for param in self._base_model.parameters():
                param.requires_grad = False
        
        if aux_logits:
            aux_weights = self._base_model.AuxLogits.fc.weight
            aux_bias = self._base_model.AuxLogits.fc.bias            
            self._base_model.AuxLogits = _InceptionAuxAge(768, num_classes)

            if num_classes == 1000:
                self._base_model.AuxLogits.fc.weight = torch.nn.Parameter(aux_weights.reshape(num_classes, 768, 1, 1))
                self._base_model.AuxLogits.fc.bias = torch.nn.Parameter(aux_bias)

        if channels_org != 3:
            self._base_model.Conv2d_1a_3x3 = inception.BasicConv2d(channels_org, 32, kernel_size=3, stride=2)
        
        del self._base_model.avgpool
        fc_weights = self._base_model.fc.weight
        fc_bias = self._base_model.fc.bias        
        self._base_model.fc = nn.Conv2d(in_channels= 2048, out_channels=num_classes, kernel_size=1, stride=1)
        if num_classes == 1000:
            self._base_model.fc.weight = torch.nn.Parameter(fc_weights.reshape(num_classes, 2048, 1, 1))
            self._base_model.fc.bias = torch.nn.Parameter(fc_bias)

        if consensus:
            self._consensus = nn.Sequential(_ViewAs3D(), nn.AdaptiveMaxPool3d(output_size=(num_classes, 1, 1)), _ViewAs2D())
        else:
            self._consensus = nn.Identity()        

        if not pretrained:
            self.apply(initialize_weights)

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self._base_model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self._base_model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self._base_model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self._base_model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self._base_model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self._base_model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self._base_model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self._base_model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self._base_model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self._base_model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self._base_model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self._base_model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self._base_model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self._base_model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self._base_model.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux = None
        if self._base_model.AuxLogits is not None:
            if self.training:
                aux = self._base_model.AuxLogits(x)
                aux = self._consensus(aux)
        # N x 768 x 17 x 17
        x = self._base_model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self._base_model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self._base_model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # N x 2048 x H'//8 x W'//8
        x = self._base_model.dropout(x)
        # N x 2048 x H'//8 x W'//8
        x = self._base_model.fc(x)
        # N x num_classes x H'//8 x W'//8
        x = self._consensus(x)

        return x, aux


class ResNetAge(nn.Module):
    def __init__(self, channels_org, num_classes, pretrained=False, consensus=True, **kwargs):
        super(ResNetAge, self).__init__()

        self._base_model = resnet.resnet152(pretrained=pretrained, progress=False)
        
        # Fix the weights of the base model when pretrained is True
        if pretrained:
            for param in self._base_model.parameters():
                param.requires_grad = False

        # Change the classifier from a fully connected layer to a conv2d with kernel size = 1x1
        fc_weights = self._base_model.fc.weight
        fc_bias = self._base_model.fc.bias

        del self._base_model.avgpool
        self._base_model.fc = nn.Conv2d(in_channels=fc_weights.size(1), out_channels=num_classes, kernel_size=1, stride=1)
        
        if num_classes == 1000:
            self._base_model.fc.weight = torch.nn.Parameter(fc_weights.reshape(num_classes, fc_weights.size(1), 1, 1))
            self._base_model.fc.bias = torch.nn.Parameter(fc_bias)
        
        if channels_org != 3:
            self._base_model.conv1 = nn.Conv2d(channels_org, self._base_model.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        
        if consensus:
            self._consensus = nn.Sequential(_ViewAs3D(), nn.AdaptiveMaxPool3d(output_size=(num_classes, 1, 1)), _ViewAs2D())
        else:
            self._consensus = nn.Identity()        

        if not pretrained:
            self.apply(initialize_weights)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self._base_model.conv1(x)
        x = self._base_model.bn1(x)
        x = self._base_model.relu(x)
        x = self._base_model.maxpool(x)

        x = self._base_model.layer1(x)
        x = self._base_model.layer2(x)
        x = self._base_model.layer3(x)
        x = self._base_model.layer4(x)

        x = self._base_model.fc(x)

        x = self._consensus(x)

        return x


class MobileNetAge(nn.Module):
    def __init__(self, channels_org, num_classes, pretrained=False, consensus=True, **kwargs):
        super(MobileNetAge, self).__init__()

        self._base_model = mobilenet.mobilenet_v2(pretrained=pretrained, progress=False)
        
        # Fix the weights of the base model when pretrained is True
        if pretrained:
            for param in self._base_model.parameters():
                param.requires_grad = False

        fc_weights = self._base_model.classifier[1].weight
        fc_bias = self._base_model.classifier[1].bias

        self._base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(fc_weights.size(1), num_classes, kernel_size=1, stride=1),
        )
        
        if num_classes == 1000:
            self._base_model.classifier[1].weight = torch.nn.Parameter(fc_weights.reshape(num_classes, fc_weights.size(1), 1, 1))
            self._base_model.classifier[1].bias = torch.nn.Parameter(fc_bias)

        if channels_org != 3:
            self._base_model.features[0][0] = nn.Conv2d(channels_org, 32, kernel_size=3, stride=2, padding=1, bias=False)

        if consensus:
            self._consensus = nn.Sequential(_ViewAs3D(), nn.AdaptiveMaxPool3d(output_size=(num_classes, 1, 1)), _ViewAs2D())
        else:
            self._consensus = nn.Identity()        

        if not pretrained:
            self.apply(initialize_weights)

    def forward(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self._base_model.features(x)

        x = self._base_model.classifier(x)

        x = self._consensus(x)
        return x


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    

    transform = transforms.Compose([
                    transforms.CenterCrop(768),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    x = []
    # data_dir = r'C:\Users\cervaf\Documents\Datasets\ImageNet\ILSVRC\Data\CLS-LOC\train\n01484850'
    data_dir = r'C:\Users\cervaf\Documents\Datasets\Kodak'
    for im_fn in [fn for fn in os.listdir(data_dir)[-10:] if fn.lower().endswith('.jpeg')]:
        fn = os.path.join(data_dir, im_fn)
        im = Image.open(fn)
        im = transform(im).unsqueeze(0)
        x.append(im)
    x = torch.cat(x, dim=0)

    net_inc_v3 = inception.inception_v3(pretrained=True, progress=False)
    net_inc_v3.eval()

    with torch.no_grad():
        y_inc = net_inc_v3(x)
        if isinstance(y_inc, tuple):
            y_inc, aux_y_inc = y_inc
        top5_prob, top5_catid = torch.topk(torch.softmax(y_inc.squeeze(), dim=1), 3, dim=1)
        print('Inception V3\n', top5_catid)

    # -------------------------------------------------------------------------------------------------
    net = InceptionV3Age(in_channels=3, num_classes=1000, pretrained=True, aux_logits=True, consensus=True)

    net.eval()
    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y, aux_y = y
        
        top5_prob, top5_catid = torch.topk(torch.softmax(y.squeeze(), dim=1), 3, dim=1)
        print('Inception V3 (Consensus: max confidence)\n', top5_catid)
       
    # -------------------------------------------------------------------------------------------------
    net = InceptionV3Age(in_channels=3, num_classes=1000, pretrained=True, aux_logits=True, consensus=False)

    net.eval()
    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y, aux_y = y
        
        top5_prob, top5_catid = torch.topk(torch.softmax(y.squeeze(), dim=1), 3, dim=1)
        print('Inception V3 (Consensus: none)\n', top5_catid)
       
    print('End experiment')
