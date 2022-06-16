import logging
from turtle import forward 

from torchvision.models import inception, resnet, mobilenet

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_dim=1024, dropout=0.0):
        super(PositionalEncoding, self).__init__()

        self._dropout = nn.Dropout2d(p=dropout, inplace=False)

        pos_y, pos_x = torch.meshgrid([torch.arange(max_dim)]*2)
        div_term = torch.exp(torch.arange(0, d_model//2, 2)*(-math.log(max_dim*2)/(d_model//2)))
        
        pe = torch.zeros(max_dim, max_dim, d_model)
        pe[..., 0:d_model//2:2] = torch.sin(pos_x.unsqueeze(dim=2) * div_term)
        pe[..., 1:d_model//2:2] = torch.cos(pos_x.unsqueeze(dim=2) * div_term)
        
        pe[..., d_model//2::2] = torch.sin(pos_y.unsqueeze(dim=2) * div_term)
        pe[..., (d_model//2+1)::2] = torch.cos(pos_y.unsqueeze(dim=2) * div_term)

        pe = pe.permute(2, 0, 1).unsqueeze(dim=0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        _, _, h, w = x.size()
        x = x + self.pe[..., :h, :w]
        return self._dropout(x)


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
        # Adaptive average pooling
        h, w = x.shape[-2:]
        x = F.adaptive_avg_pool2d(x, output_size=(max(1, h//5), max(1, w//5)))
        # N x 768 x H' x W'
        x = self.fc(x)
        # N x num_classes x H' x W'
        return x


class InceptionV3Age(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=False, aux_logits=True, consensus=True):
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

        if in_channels != 3:
            self._base_model.Conv2d_1a_3x3 = inception.BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        
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
        # Adaptive average pooling
        h, w = x.shape[-2:]
        x = F.adaptive_avg_pool2d(x, output_size=(max(1, h//8), max(1, w//8)))
        # N x 2048 x H'//8 x W'//8
        x = self._base_model.dropout(x)
        # N x 2048 x H'//8 x W'//8
        x = self._base_model.fc(x)
        # N x num_classes x H'//8 x W'//8

        x = self._consensus(x)

        return x, aux


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    
    print('Testing random crop masking')

    transform = transforms.Compose([
                    transforms.Resize(768),
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
    net = InceptionV3Age(in_channels=3, num_classes=1000, pretrained=True, consensus=True)

    net.eval()
    with torch.no_grad():
        y, aux_y = net(x)
        top5_prob, top5_catid = torch.topk(torch.softmax(y.squeeze(), dim=1), 3, dim=1)
        print(top5_catid)

    transform = transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    x = []
    data_dir = r'C:\Users\cervaf\Documents\Datasets\Kodak'
    for im_fn in [fn for fn in os.listdir(data_dir)[-10:] if fn.lower().endswith('.jpeg')]:
        fn = os.path.join(data_dir, im_fn)
        im = Image.open(fn)
        im = transform(im).unsqueeze(0)
        x.append(im)
    x = torch.cat(x, dim=0)

    net_inc_v3 = inception.inception_v3(pretrained=True, transform_input=False, progress=True, aux_logits=False, init_weights=False)
    net_inc_v3.eval()
    with torch.no_grad():
        y_inc = net_inc_v3(x)
        top5_prob, top5_catid = torch.topk(torch.softmax(y_inc, dim=1), 3, dim=1)
        print(top5_catid)
    