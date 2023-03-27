import sys
from torchvision.models import inception, resnet, mobilenet, vision_transformer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implement a version of the Mobilenet


class ViTClassifierHead(vision_transformer.VisionTransformer):
    """Implementation of the classifier head from the ViT-B-16 architecture.
    """
    def __init__(self, channels_org=3, channels_bn=768, cut_position=6,
                 patch_size=128,
                 compression_level=4,
                 num_classes=1000,
                 dropout=0.0,
                 **kwargs):
        if cut_position is None:
            cut_position = 6

        if cut_position > 0:
            image_size = patch_size // 2**compression_level
            vit_patch_size = 1

        else:
            image_size = patch_size
            vit_patch_size = 16

        super(ViTClassifierHead, self).__init__(
            image_size=image_size,
            patch_size=vit_patch_size,
            num_layers=12 - cut_position,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=num_classes,
            dropout=dropout)

        # The input `x` is the output of several convolution layers, and the
        # number of channels is already the required as input for the encoder
        # layers of the ViT. For that reason, the projection layer is not
        # longer needed. And the number of encoder layers are reduced too.
        if cut_position > 0:
            self.conv_proj = nn.Conv2d(channels_bn, 768, kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)
        elif channels_org != 3:
            self.conv_proj = nn.Conv2d(channels_org,
                                       self.conv_proj.out_channels,
                                       kernel_size=self.conv_proj.kernel_size,
                                       stride=self.conv_proj.stride,
                                       padding=self.conv_proj.padding,
                                       bias=self.conv_proj.bias is not None)

    def forward(self, x):
        pred = super().forward(x)
        return pred, None


class ResNetClassifierHead(resnet.ResNet):
    """Implementation of the classifier head from the ResNet-152 architecture.
    """
    def __init__(self, channels_org=3, channels_bn=768, cut_position=3,
                 patch_size=128,
                 compression_level=4,
                 num_classes=1000,
                 **kwargs):

        if cut_position is None:
            cut_position = compression_level

        super(ResNetClassifierHead, self).__init__(
            block=resnet.Bottleneck,
            norm_layer=lambda ch: nn.GroupNorm(num_groups=ch, num_channels=ch),
            layers=[3, 8, 36, 3],
            num_classes=num_classes)

        out_channels = [64, 64 * 4, 128 * 4, 256 * 4, 512 * 4]

        if cut_position > 0:
            self.conv1 = nn.Conv2d(channels_bn,
                                   out_channels[cut_position - 1],
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   bias=False)
            self.bn1 = nn.GroupNorm(out_channels[cut_position - 1],
                                    out_channels[cut_position - 1])
            self.maxpool = nn.Identity()

        elif channels_org != 3:
            self.conv1 = nn.Conv2d(channels_org,
                                   self.conv1.out_channels,
                                   kernel_size=self.conv1.kernel_size,
                                   stride=self.conv1.stride,
                                   padding=self.conv1.padding,
                                   bias=self.conv1.bias is not None)

        if cut_position > 1:
            self.layer1 = nn.Identity()

        if cut_position > 2:
            self.layer2 = nn.Identity()

        if cut_position > 3:
            self.layer3 = nn.Identity()

        if cut_position > 4:
            self.layer4 = nn.Identity()

    def forward(self, x):
        y = self._forward_impl(x)
        return y, None


class InceptionV3ClassifierHead(inception.Inception3):
    def __init__(self, channels_org=3, channels_bn=768, cut_position=6,
                 patch_size=128,
                 compression_level=4,
                 num_classes=1000,
                 dropout=0.0,
                 **kwargs):

        super(InceptionV3ClassifierHead, self).__init__(
                    num_classes=num_classes,
                    aux_logits=True,
                    transform_input=False,
                    inception_blocks=None,
                    init_weights=True,
                    dropout=dropout)

        out_channels = [32, 64, 192, 768, 1280, 2048]
        in_shapes = [299, 149, 73, 35, 17, 8, 1]

        bn_shape = patch_size // 2 ** compression_level
        if cut_position is None:
            cut_position = min(map(lambda si: (abs(si[1] - bn_shape), si[0]),
                                   enumerate(in_shapes)))[1]
        
        pad_left_top = (in_shapes[cut_position] - bn_shape) // 2
        pad_right_bottom = in_shapes[cut_position] - bn_shape - pad_left_top
        
        if pad_left_top > 0 or pad_right_bottom > 0:
            self._pre_padding = nn.ReplicationPad2d((pad_left_top,
                                                     pad_right_bottom,
                                                     pad_left_top,
                                                     pad_right_bottom))
        else:
            self._pre_padding = nn.Identity()

        if cut_position > 0:
            self.Conv2d_1a_3x3 = nn.Conv2d(channels_bn,
                                           out_channels[cut_position - 1],
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False)

        elif channels_org != 3:
            self.Conv2d_1a_3x3 = inception.BasicConv2d(channels_org, 32,
                                                       kernel_size=3,
                                                       stride=2)

        if cut_position > 1:
            self.Conv2d_2a_3x3 = nn.Identity()
            self.Conv2d_2b_3x3 = nn.Identity()
            self.maxpool1 = nn.Identity()

        if cut_position > 2:
            self.Conv2d_3b_1x1 = nn.Identity()
            self.Conv2d_4a_3x3 = nn.Identity()
            self.maxpool2 = nn.Identity()

        if cut_position > 3:
            self.Mixed_5b = nn.Identity()
            self.Mixed_5c = nn.Identity()
            self.Mixed_5d = nn.Identity()
            self.Mixed_6a = nn.Identity()
    
        if cut_position > 4:
            self.AuxLogits.conv0 = nn.Identity()
            self.AuxLogits.conv1 = nn.Conv2d(out_channels[cut_position - 1],
                                             768,
                                             kernel_size=1,
                                             stride=1)

            self.Mixed_6b = nn.Identity()
            self.Mixed_6c = nn.Identity()
            self.Mixed_6d = nn.Identity()
            self.Mixed_6e = nn.Identity()
            self.Mixed_7a = nn.Identity()

        if cut_position > 5:
            self.Mixed_7b = nn.Identity()
            self.Mixed_7c = nn.Identity()

    def forward(self, x):
        x = self._pre_padding(x)
        y, y_aux = self._forward(x)
        return y, y_aux


CLASS_MODELS = {
    "ViT": ViTClassifierHead,
    "ResNet": ResNetClassifierHead,
    "InceptionV3": InceptionV3ClassifierHead,
    }


def setup_modules(class_model_type, **kwargs):
    class_model = CLASS_MODELS[class_model_type](**kwargs)
    return class_model


def load_state_dict(model, checkpoint_state):
    if 'class_model' in checkpoint_state.keys():
        model.load_state_dict(checkpoint_state['class_model'])


def classifier_from_state_dict(checkpoint, gpu=False, train=False):
    if isinstance(checkpoint, str):
        checkpoint_state = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint_state = checkpoint

    assert checkpoint_state.get('class_model_type', None) in CLASS_MODELS

    model = setup_modules(**checkpoint_state)

    load_state_dict(model, checkpoint_state)

    # If there are more than one GPU, DataParallel handles automatically the
    # distribution of the work.
    model = nn.DataParallel(model)

    if gpu and torch.cuda.is_available():
        model.cuda()

    if train:
        model.train()
    else:
        model.eval()

    return model


if __name__ == '__main__':
    import os
    from PIL import Image
    from torchvision import transforms
    
    transform = transforms.Compose([
                    transforms.CenterCrop((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    x = []

    data_dir = sys.argv[1]
    for im_fn in [fn for fn in os.listdir(data_dir)[-10:] if fn.lower().endswith('.png')]:
        fn = os.path.join(data_dir, im_fn)
        im = Image.open(fn)
        im = transform(im).unsqueeze(0)
        x.append(im)
    x = torch.cat(x, dim=0)

    net = ResNetClassifierHead(cut_position=0, patch_size=256)

    net.eval()
    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y, aux_y = y

        top5_prob, top5_catid = torch.topk(torch.softmax(y, dim=1), 1, dim=1)
        print('ViT (Consensus: Most confident), y_size:{}, {}\n'.format(y.size(), top5_catid.size()), top5_catid.squeeze(), top5_prob.squeeze())

    print('End experiment')
