import sys
from torchvision.models import inception, resnet, mobilenet, vision_transformer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implement a version of the Inception V3, Mobilenet, and ResNet


class EmptyClassifierHead(nn.Module):
    """This empty classifier is intended for debug the main training process.
    """
    def __init__(self, **kwargs):
        super(EmptyClassifierHead, self).__init__()

    def forward(self, x):
        return None, None


class ViTClassifierHead(vision_transformer.VisionTransformer):
    def __init__(self, channels_bn=768, cut_poisition=6, patch_size=128,
                 compression_level=4,
                 **kwargs):
        if cut_poisition is None:
            cut_poisition = 6

        if cut_poisition > 0:
            image_size = patch_size // 2**compression_level
            vit_patch_size = 1

        else:
            image_size = patch_size
            vit_patch_size = 2**compression_level

        super(ViTClassifierHead, self).__init__(
            image_size=image_size,
            patch_size=vit_patch_size,
            num_layers=12 - cut_poisition,
            num_heads=12,
            hidden_dim=channels_bn,
            mlp_dim=3072)

        # The input `x` is the output of several convolution layers, and the
        # number of channels is already the required as input for the encoder
        # layers of the ViT. For that reason, the projection layer is not
        # longer needed. And the number of encoder layers are reduced too.
        if cut_poisition > 0:
            self.conv_proj = nn.Identity()

    def forward(self, x):
        pred = super().forward(x)
        return pred, None


CLASS_MODELS = {
    "Empty": EmptyClassifierHead,
    "ViT": ViTClassifierHead
    }


def setup_classifier_modules(class_model_type, **kwargs):
    class_model = CLASS_MODELS[class_model_type](**kwargs)
    return dict(class_model=class_model)


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

    net = ViTClassifierHead(cut_poisition=0, patch_size=256)

    net.eval()
    with torch.no_grad():
        y = net(x)
        if isinstance(y, tuple):
            y, aux_y = y

        top5_prob, top5_catid = torch.topk(torch.softmax(y, dim=1), 1, dim=1)
        print('ViT (Consensus: Most confident), y_size:{}, {}\n'.format(y.size(), top5_catid.size()), top5_catid.squeeze(), top5_prob.squeeze())

    print('End experiment')
