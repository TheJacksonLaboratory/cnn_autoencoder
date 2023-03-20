import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3,
                 downsample_op=nn.MaxPool2d):
        super(DownsamplingUnit, self).__init__()
        if downsample_op is None:
            downsample_op = nn.Identity

        self._dwn_sample = downsample_op(kernel_size=2, stride=2, padding=0)
        self._c1 = nn.Conv2d(channels_in, channels_out,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)
        self._bn1 = nn.BatchNorm2d(channels_out)
        self._c2 = nn.Conv2d(channels_out, channels_out,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)
        self._bn2 = nn.BatchNorm2d(channels_out)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self._dwn_sample(x)
        fx = self._c1(fx)
        fx = self._bn1(fx)
        fx = self._relu(fx)
        fx = self._c2(fx)
        fx = self._bn2(fx)
        fx = self._relu(fx)
    
        return fx


class UpsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_unit, channels_out, kernel_size=3,
                 upsample_op=nn.ConvTranspose2d):
        super(UpsamplingUnit, self).__init__()
        if upsample_op is None:
            upsample_op = nn.Identity

        self._c1 = nn.Conv2d(channels_in, channels_unit,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)
        self._bn1 = nn.BatchNorm2d(channels_unit)
        self._c2 = nn.Conv2d(channels_unit, channels_unit,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)
        self._bn2 = nn.BatchNorm2d(channels_unit)
        self._relu = nn.ReLU(inplace=True)
        self._up_sample = upsample_op(channels_unit, channels_out,
                                      kernel_size=2,
                                      stride=2,
                                      padding=0,
                                      output_padding=0,
                                      bias=True)

    def forward(self, x):
        fx = self._c1(x)
        fx = self._bn1(fx)
        fx = self._relu(fx)
        fx = self._c2(fx)
        fx = self._bn2(fx)
        fx = self._relu(fx)
        fx = self._up_sample(fx)
        return fx


class BottleneckUnit(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3):
        super(BottleneckUnit, self).__init__()
        self._dwn_sample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self._c1 = nn.Conv2d(channels_in, channels_out,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)
        self._bn1 = nn.BatchNorm2d(channels_out)
        self._c2 = nn.Conv2d(channels_out, channels_out,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=kernel_size//2,
                             bias=False)
        self._bn2 = nn.BatchNorm2d(channels_out)
        self._relu = nn.ReLU(inplace=True)
        self._up_sample = nn.ConvTranspose2d(channels_out, channels_in,
                                             kernel_size=2, stride=2,
                                             padding=0,
                                             output_padding=0,
                                             bias=True)

    def forward(self, x):
        fx = self._dwn_sample(x)
        fx = self._c1(fx)
        fx = self._bn1(fx)
        fx = self._relu(fx)
        fx = self._c2(fx)
        fx = self._bn2(fx)
        fx = self._relu(fx)
        fx = self._up_sample(fx)
        return fx


class UNet(nn.Module):
    def __init__(self, channels_org=3, channels_net=64, channels_bn=1024,
                 channels_expansion=2,
                 compression_level=4,
                 num_classes=1,
                 use_analysis_track=True,
                 concat_bridges=True,
                 project_bridges_from_channels=None,
                 **kwargs):
        super(UNet, self).__init__()

        # Concat bridges means feature maps will be concatenated during the
        # synthesis track.
        self._concat_bridges = concat_bridges

        if use_analysis_track:
            project_bridges_from_channels = None

            channels_in_list = [channels_org] 
            channels_in_list += [channels_net * channels_expansion ** c
                                 for c in range(compression_level - 1)]
            channels_out_list = [channels_net * channels_expansion ** c
                                 for c in range(compression_level)]
            downsample_op_list = [None]
            downsample_op_list += [nn.MaxPool2d] * (compression_level - 1)

            analysis_track = []
            for ch_in, ch_out, dws_op in zip(channels_in_list, 
                                             channels_out_list,
                                             downsample_op_list):
                analysis_track.append(
                    DownsamplingUnit(channels_in=ch_in, channels_out=ch_out,
                                     kernel_size=3,
                                     downsample_op=dws_op))

            self.analysis_track = nn.ModuleList(analysis_track)

        else:
            self.analysis_track = []

        channels_in_list = [channels_net * channels_expansion ** c
                            for c in reversed(range(compression_level))]
        channels_out_list = [channels_net * channels_expansion ** (c - 1)
                                for c in reversed(range(compression_level))]

        upsample_op_list = [nn.ConvTranspose2d] * (compression_level - 1)
        upsample_op_list += [None]

        synthesis_track = []
        bridges_projection = []
        for ch_in, ch_out, ups_op in zip(channels_in_list, 
                                         channels_out_list,
                                         upsample_op_list):
            if project_bridges_from_channels is not None and concat_bridges:
                bridges_projection.append(
                    nn.Conv2d(project_bridges_from_channels,
                              ch_in,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)
                )
            else:
                bridges_projection.append(nn.Identity())

            synthesis_track.append(
                UpsamplingUnit(channels_in=ch_in * 2 ** concat_bridges,
                               channels_unit=ch_in,
                               channels_out=ch_out,
                               kernel_size=3,
                               upsample_op=ups_op))

        self.bridges_projection = nn.ModuleList(bridges_projection)
        self.synthesis_track = nn.ModuleList(synthesis_track)

        self.bottleneck = BottleneckUnit(
            channels_net * channels_expansion ** (compression_level - 1),
            channels_bn)

        self.fc = nn.Conv2d(channels_net, num_classes, kernel_size=1, stride=1,
                            padding=0,
                            bias=True)

    def forward(self, x, fx_brg=None):
        fx = x

        if len(self.analysis_track):
            fx_brg = []

        for layer in self.analysis_track:
            fx = layer(fx)
            fx_brg.insert(0, fx)

        # Bottleneck
        fx = self.bottleneck(fx)

        if self._concat_bridges:
            for fx_b, proj, layer in zip(fx_brg,
                                         self.bridges_projection,
                                         self.synthesis_track):
                fx = torch.cat((proj(fx_b), fx), dim=1)
                fx = layer(fx)
        else:
            for layer in self.synthesis_track:
                fx = layer(fx)

        # Pixel-wise class prediction
        y = self.fc(fx)
        return y, None


class JNet(UNet):
    def __init__(self, channels_net=64, channels_bn=1024,
                 channels_expansion=2,
                 compression_level=4,
                 concat_bridges=False,
                 **kwargs):
        super(JNet, self).__init__(channels_net=channels_net,
                                   channels_bn=channels_bn,
                                   channels_expansion=channels_expansion,
                                   compression_level=compression_level,
                                   use_analysis_track=False,
                                   save_bridges=False,
                                   concat_bridges=concat_bridges,
                                   **kwargs)

        self.bottleneck = nn.ConvTranspose2d(
            channels_bn,
            channels_net * channels_expansion ** (compression_level - 1),
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False)


SEG_MODELS = {
    "UNet": UNet,
    "JNet": JNet,
    }


def setup_modules(segment_model_type, **kwargs):
    seg_model = SEG_MODELS[segment_model_type](**kwargs)
    return seg_model


def load_state_dict(model, checkpoint_state):
    if 'seg_model' in checkpoint_state.keys():
        model.load_state_dict(checkpoint_state['seg_model'])


def segmenter_from_state_dict(checkpoint, gpu=False, train=False):
    if isinstance(checkpoint, str):
        checkpoint_state = torch.load(checkpoint, map_location='cpu')
    else:
        checkpoint_state = checkpoint

    if checkpoint_state.get('segment_model_type', None) not in SEG_MODELS:
        return None

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


if __name__ == "__main__":
    seg = JNet(channels_prg=3, channels_net=64, channels_expansion=2,
               channels_bn=320,
               num_classes=1,
               concat_bridges=True,
               project_bridges_from_channels=192)
    x = torch.rand([2, 320, 2, 2])
    x_brg = [torch.rand([2, c, 32 // 2 ** s, 32 // 2 ** s])
             for c, s in [(192, 3), (192, 2), (192, 1), (192, 0)]]
    pred = seg(x, x_brg)
