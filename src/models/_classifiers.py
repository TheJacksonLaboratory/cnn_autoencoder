import argparse
from collections import OrderedDict
from functools import partial
from torchvision.models import inception, resnet, mobilenet, vision_transformer

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=kernel_size//2)
        self.bnorm = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        fx = self.conv(x)
        fx = self.bnorm(fx)
        fx = self.act(fx)
        return fx


class BaseModel(nn.Module):
    def __init__(self, im_size, channels_bn, num_classes, hidden_block_op=None,
                 hidden_channels=None,
                 out_op=None,
                 **kwargs):
        super(BaseModel, self).__init__()
        if not isinstance(im_size, tuple):
            im_size = (im_size, im_size)

        if out_op is None:
            out_op = nn.Identity

        if hidden_block_op is None and hidden_channels is not None:
            hidden_block_op = ConvBlock

        if hidden_channels is None:
            hidden_channels = []

        self._im_size = im_size

        prev_out_channels = channels_bn
        layers = []
        for h in hidden_channels:
            layers.append(hidden_block_op(in_channels=prev_out_channels, out_channels=h, kernel_size=1, stride=1))
            prev_out_channels = h

        self._layers = nn.Sequential(*layers)

        self._unfold = nn.Unfold(kernel_size=(im_size[0], im_size[1]), stride=(im_size[0], im_size[1]))

        self._classifier = nn.Linear(in_features=prev_out_channels * im_size[0] * im_size[1], out_features=num_classes)

        self._out_op = out_op()

        self._pool_op = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def features(self, x):
        fx = self._layers(x)

        b, c, h, w = fx.shape
        n_h = h // self._im_size[0]
        n_w = w // self._im_size[0]

        fx = self._unfold(fx)

        fx = fx.permute(0, 2, 1)

        fx = self._classifier(fx)

        fx = fx.permute(0, 2, 1).reshape(b, -1, n_h, n_w)

        return fx

    def forward(self, x, **kwargs):
        fx = self.features(x)

        y = self._out_op(fx)

        # This works as a consensus of the label given to the image according
        # to the prediction from all sub-patches.
        y = self._pool_op(y)

        return y, None


class WSIEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation with
    whole slide images.

    The main difference between this class and the original ViT Encoder class
    is that positional encoding is generated from fixed sinusoidal waves
    instead of learned during training. Fixed waves are used because WSI can
    have arbitrary shape in a large scale, which would require large amounts of
    memory to keep track of a learnable positional encoding.
    """
    def __init__(
        self,
        patch_size,
        compression_level,
        max_width=1000000,
        max_height=1000000,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        attention_dropout=0.0,
        dropout=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._patch_size = patch_size
        self._compression_level = compression_level
        self._max_height = (max_height // patch_size) // 2 ** compression_level
        self._max_width = (max_width // patch_size) // 2 ** compression_level

        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = vision_transformer.EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )

        dim_seq = torch.arange(hidden_dim // 4)

        sin_div_x = 1e-5 ** (2 * dim_seq / hidden_dim)
        cos_div_x = 1e-5 ** ((2 * dim_seq + 1) / hidden_dim)
        sin_div_y = 1e-5 ** ((2 * dim_seq + hidden_dim // 2) / hidden_dim)
        cos_div_y = 1e-5 ** ((2 * dim_seq + 1 + hidden_dim // 2) / hidden_dim)

        self.register_buffer("_sin_div_x", sin_div_x)
        self.register_buffer("_cos_div_x", cos_div_x)
        self.register_buffer("_sin_div_y", sin_div_y)
        self.register_buffer("_cos_div_y", cos_div_y)

        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def get_pos_embedding(self, position):
        x = ((position[:, 0] // self._patch_size)
             / 2 ** self._compression_level)
        y = ((position[:, 1] // self._patch_size)
             / 2 ** self._compression_level)

        pos_i_sin = torch.sin(y.view(-1, 1) * self._sin_div_y.view(1, -1))
        pos_i_cos = torch.sin(y.view(-1, 1) * self._cos_div_y.view(1, -1))
        pos_j_sin = torch.sin(x.view(-1, 1) * self._sin_div_x.view(1, -1))
        pos_j_cos = torch.sin(x.view(-1, 1) * self._cos_div_x.view(1, -1))

        pos_embedding = torch.stack((pos_i_sin, pos_i_cos,
                                     pos_j_sin, pos_j_cos), dim=-1)

        pos_embedding = torch.flatten(pos_embedding, -2, -1)
        pos_embedding = pos_embedding.view(-1, 1, self._hidden_dim)

        return pos_embedding

    def forward(self, input, position=None, **kwargs):
        # No extra drop out is perfomed since patches are already randomly
        # sampled from the image.
        b, c, h, w = input.shape
        input = input.permute(1, 0, 2, 3).reshape(1, c, h * b, w)

        assert position is not None, "Positions must be passed to this class"
        input = input + self.get_pos_embedding(position)
        return self.ln(self.layers(input))


class ViTWSIClassifier(nn.Module):
    """Implementation of a ViT classifier to be used with whole slide images as
    inputs.

    This model is intended for use along with an Autoencoder model that
    downsamples the image into a more manageable size.
    """
    def __init__(self, channels_bn=768,
                 num_layers=6,
                 num_heads=12,
                 hidden_dim=768,
                 mlp_dim=3072,
                 num_classes=1000,
                 patch_size=128,
                 compression_level=4,
                 max_width=1000000,
                 max_height=1000000,
                 attention_dropout=0.0,
                 dropout=0.0,
                 **kwargs):
        super().__init__()

        self.patch_size = patch_size // 2 ** compression_level
        self.hidden_dim = hidden_dim
        self.mlp_dim = 3072
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder = WSIEncoder(patch_size,
                                  compression_level,
                                  max_width=max_width,
                                  max_height=max_height,
                                  num_layers=num_layers,
                                  num_heads=num_heads,
                                  hidden_dim=hidden_dim,
                                  mlp_dim=mlp_dim,
                                  attention_dropout=attention_dropout,
                                  dropout=dropout)

        self.conv_proj = nn.Conv2d(in_channels=channels_bn,
                                   out_channels=hidden_dim,
                                   kernel_size=1,
                                   stride=1)

        heads_layers = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        self.heads = nn.Sequential(heads_layers)

        # Init the patchify stem
        fan_in = (self.conv_proj.in_channels
                  * self.conv_proj.kernel_size[0]
                  * self.conv_proj.kernel_size[1])

        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))

        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x):
        n, _, n_h, n_w = x.shape

        # (n, c, n_h, n_w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x, position=None, **kwargs):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x, position)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x, None


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


BASELINE_MODELS = {
    "SVM": {
        "im_size": 16, 
        "hidden_block_op": None,
        "hidden_channels": None,
        "out_op": nn.Tanh
    },
    "Logistic": {
        "im_size": 16, 
        "hidden_block_op": None,
        "hidden_channels": None,
        "out_op": None
    },
    "MLP": {
        "im_size": 16, 
        "hidden_block_op": ConvBlock,
        "hidden_channels": [128, 64, 32],
        "out_op": None
    }
}


CLASS_MODELS = {
    "Base-WSI": BaseModel,
    "ViT-WSI": ViTWSIClassifier,
    "ViT": ViTClassifierHead,
    "ResNet": ResNetClassifierHead,
    "InceptionV3": InceptionV3ClassifierHead,
    }


def setup_modules(class_model_type, **kwargs):
    if class_model_type in BASELINE_MODELS:
        kwargs.update(BASELINE_MODELS[class_model_type])
        class_model_type = "Base-WSI"

    class_model = CLASS_MODELS[class_model_type](**kwargs)
    return class_model


def load_state_dict(model, checkpoint_state):
    if 'class_model' in checkpoint_state.keys():
        model.load_state_dict(checkpoint_state['class_model'])


def classifier_from_state_dict(checkpoint, gpu=False, train=False):
    if isinstance(checkpoint, str):
        checkpoint_state = torch.load(checkpoint, map_location='cpu')
    elif isinstance(checkpoint, argparse.Namespace):
        checkpoint_state = checkpoint.__dict__
    else:
        checkpoint_state = checkpoint

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
    x = torch.rand(10, 320, 1, 128)
    pos = torch.randint(200000, (10, 4))

    model = ViTWSIClassifier(channels_bn=320, num_classes=1,
                             num_layers=12,
                             num_heads=12,
                             hidden_dim=768,
                             mlp_dim=3072,
                             patch_size=16,
                             compression_level=4)

    output = model(x, pos)

    print(output.shape)
