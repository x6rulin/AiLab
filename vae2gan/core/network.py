"""Generator (Decoder) and Critic (Encoder) networks."""
from math import sqrt, log2
import torch

from vae2gan.ganlib.normalize import pixel_norm, instance_norm, minibatch_stddev_layer
from vae2gan.ganlib.eqlr import Linear, Conv2d
from vae2gan.ganlib.scale import UpscaleConv2d, ConvDownscale2d


def _normalization(key, **kwargs):
    if key == 'none': return []
    if key == 'BN': return [torch.nn.BatchNorm2d(**kwargs)]
    if key == 'LN': return [torch.nn.LayerNorm(**kwargs)]

    raise RuntimeError(f"not supported normaliztion: '{key}'")


_kwargs = lambda fmap, res: {'none': {}, \
                        'BN': dict(num_features=fmap), \
                        'LN': dict(normalized_shape=(fmap, res, res))}

_activate = {
    'none': [lambda: [], 1.],
    'relu': [lambda: [torch.nn.ReLU()], sqrt(2)],
    'lrelu': [lambda: [torch.nn.LeakyReLU(0.2)], sqrt(2)],
    'prelu': [lambda: [torch.nn.PReLU()], sqrt(2)],
    'tanh': [lambda: [torch.nn.Tanh()], 5.0 / 3],
    'sigmoid': [lambda: [torch.nn.Sigmoid()], 1.],
}


class EpilogueLayer(torch.nn.Module):
    """Things to do at the end of each layer. """
    def __init__(self, fmap, size, activate, use_pixel_norm=False, normalization='none'):
        super(EpilogueLayer, self).__init__()

        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(1, fmap, 1, 1)))
        self.act = torch.nn.Sequential(*activate())
        self.pixel_norm = lambda x: pixel_norm(x) if use_pixel_norm else x
        if normalization == 'IN':
            self.normalize = instance_norm
        else:
            self.normalize = torch.nn.Sequential(
                *_normalization(normalization, **_kwargs(fmap, size)[normalization]),
            )

    def forward(self, x):
        x = x + self.bias
        x = self.act(x)
        x = self.pixel_norm(x)
        x = self.normalize(x)

        return x


class _InputG(torch.nn.Module):

    def __init__(self, in_features, fmap_out, size, gain=1., use_wscale=False, **kwargs):
        super(_InputG, self).__init__()

        self.embed = Linear(in_features, fmap_out * size * size, gain=gain/4, use_wscale=use_wscale)
        self.sub_module = torch.nn.Sequential(
            EpilogueLayer(fmap_out, size, **kwargs),
            Conv2d(fmap_out, fmap_out, 3, 1, 1, gain=gain, use_wscale=use_wscale),
            EpilogueLayer(fmap_out, size, **kwargs),
        )

        self.size = size

    def forward(self, x):
        x = self.embed(x)
        x = x.reshape(x.size(0), -1, self.size, self.size)
        x = self.sub_module(x)

        return x


class Lat2Img(torch.nn.Module):
    """Generates images from latents as given channels and size. """
    def __init__(self, latent_size, num_channels=3, use_wscale=True, fused_scale='auto',
                 resolution=128, fmap_base=1024, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_pixel_norm=False, normalization='IN',
                 blur_filter=[1, 2, 1]):
        super(Lat2Img, self).__init__()

        resolution_log2 = int(log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        assert fused_scale in ['auto', True, False]
        _nf = lambda stage: min(int(fmap_base / (2 ** (stage * fmap_decay))), fmap_max)
        activate, gain = _activate[nonlinearity]

        growing = []
        for i in range(2, resolution_log2):
            fmap_in, fmap_out, res = _nf(i - 1), _nf(i), 2 ** (i + 1)
            if fused_scale == 'auto':
                fused_scale = res >= 128

            growing.extend([
                UpscaleConv2d(fmap_in, fmap_out, 3, fused_scale, blur_filter,
                              gain=gain, use_wscale=use_wscale),
                EpilogueLayer(fmap_out, res, activate, use_pixel_norm, normalization),
                Conv2d(fmap_out, fmap_out, 3, 1, 1, gain=gain, use_wscale=use_wscale),
                EpilogueLayer(fmap_out, res, activate, use_pixel_norm, normalization),
            ])

        self.newborn = _InputG(latent_size, _nf(1), 4, gain=gain, use_wscale=use_wscale,
                               activate=activate, use_pixel_norm=use_pixel_norm,
                               normalization=normalization)
        self.growing = torch.nn.Sequential(*growing)
        self.torgb = Conv2d(_nf(resolution_log2 - 1), num_channels, 1, 1, 0, bias=True,
                            gain=1., use_wscale=use_wscale)

    def forward(self, x, retain_bud: bool = False):
        x = self.newborn(x)
        bud = self.growing(x)
        flower = self.torgb(bud)

        return [flower, (bud, flower)][retain_bud]


class Img2Dis(torch.nn.Module):

    def __init__(self, num_dis, num_channels=3, use_wscale=True, fused_scale='auto',
                 resolution=128, fmap_base=1024, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', normalization='none', blur_filter=[1, 2, 1],
                 mbstd_group_size=4, mbstd_num_features=1):
        super(Img2Dis, self).__init__()

        resolution_log2 = int(log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        assert fused_scale in ['auto', True, False]
        _nf = lambda stage: min(int(fmap_base / (2 ** (stage * fmap_decay))), fmap_max)
        activate, gain = _activate[nonlinearity]

        sieving = []
        for i in range(resolution_log2 - 1, 1, -1):
            fmap_in, fmap_out, res = _nf(i), _nf(i - 1), 2 ** (i + 1)
            if fused_scale == 'auto':
                fused_scale = res >= 128

            sieving.extend([
                Conv2d(fmap_in, fmap_in, 3, 1, 1, gain=gain, use_wscale=use_wscale),
                EpilogueLayer(fmap_in, res, activate, False, normalization),
                ConvDownscale2d(fmap_in, fmap_out, 3, fused_scale, blur_filter,
                                gain=gain, use_wscale=use_wscale),
                EpilogueLayer(fmap_out, res//2, activate, False, normalization),
            ])

        self.fromrgb = torch.nn.Sequential(
            Conv2d(num_channels, _nf(resolution_log2 - 1), 1, 1, 0, gain=gain,
                   use_wscale=use_wscale),
            EpilogueLayer(_nf(resolution_log2 - 1), resolution, activate, False, normalization),
        )
        self.sieving = torch.nn.Sequential(*sieving)
        self.final = torch.nn.Sequential(
            Conv2d(_nf(1) + mbstd_num_features, _nf(1), 3, 1, 1, gain=gain, use_wscale=use_wscale),
            EpilogueLayer(_nf(1), 4, activate, False, normalization),
            Conv2d(_nf(1), _nf(1), 4, 1, 0, gain=gain, use_wscale=use_wscale),
            EpilogueLayer(_nf(1), 1, activate, False, normalization),
            Conv2d(_nf(1), num_dis, 1, 1, 0, bias=True, gain=1., use_wscale=use_wscale),
        )

        self.mbstd_group_size = mbstd_group_size
        if mbstd_group_size > 1:
            self.mbstd = lambda x: minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)

    def forward(self, x):
        x = self.fromrgb(x)
        x = self.sieving(x)
        if self.mbstd_group_size > 1:
            x = self.mbstd(x)
        x = self.final(x)

        return x.reshape(x.size(0), -1)


__all__ = ["Lat2Img", "Img2Dis"]
