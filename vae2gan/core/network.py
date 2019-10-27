"""Generator (Decoder) and Critic (Encoder) networks."""
from math import sqrt, log2
import torch


def _normalization(key, **kwargs):
    if key == 'none': return []
    if key == 'BN': return [torch.nn.BatchNorm2d(**kwargs)]
    if key == 'LN': return [torch.nn.LayerNorm(**kwargs)]
    if key == 'IN': return [torch.nn.InstanceNorm2d(**kwargs)]

    raise RuntimeError(f"not supported normaliztion: '{key}'")


_kwargs = lambda fmap, res: {'none': {}, \
                        'BN': dict(num_features=fmap), \
                        'LN': dict(normalized_shape=(fmap, res, res)), \
                        'IN': dict(num_features=fmap)}

_activate = {
    'none': [lambda: [], 1.],
    'relu': [lambda: [torch.nn.ReLU(True)], sqrt(2)],
    'lrelu': [lambda: [torch.nn.LeakyReLU(0.2, True)], sqrt(2)],
    'prelu': [lambda: [torch.nn.PReLU()], sqrt(2)],
    'tanh': [lambda: [torch.nn.Tanh()], 5.0 / 3],
    'sigmoid': [lambda: [torch.nn.Sigmoid()], 1.],
}


def get_weight(shape, gain=sqrt(2), use_wscale=False, lrmul=1.0):
    """Surpports equalized learning rate for weights updating. """
    fan_in = torch.prod(torch.FloatTensor(list(shape[1:])))
    he_std = gain / torch.sqrt(fan_in)

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = torch.FloatTensor([lrmul])

    return init_std, runtime_coef


class _Conv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1, bias=True, gain=sqrt(2), use_wscale=False, lrmul=1.):
        super(_Conv2d, self).__init__()

        self.sub_module = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        init_std, runtime_coef = get_weight(self.sub_module.weight.shape, gain, use_wscale, lrmul)
        torch.nn.init.normal_(self.sub_module.weight, 0., init_std)
        if bias:
            torch.nn.init.constant_(self.sub_module.bias, 0.)

        self.register_buffer("runtime_coef", runtime_coef)

    def forward(self, x):
        return self.sub_module(x * self.runtime_coef)


class _Conv2dLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation=1, groups=1, bias=True, use_wscale=False,
                 lrmul=1., nonlinearity='none', normalization='none', **kwargs):
        super(_Conv2dLayer, self).__init__()

        activate, gain = _activate[nonlinearity]
        self.sub_module = torch.nn.Sequential(
            _Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, gain, use_wscale, lrmul),
            *_normalization(normalization, **kwargs),
            *activate(),
        )

    def forward(self, x):
        return self.sub_module(x)


class _InputLayer(torch.nn.Module):

    def __init__(self, in_features, out_channels, size, use_wscale=False,
                 lrmul=1., nonlinearity='relu', normalization='BN'):
        super(_InputLayer, self).__init__()

        self.size = size

        activate, gain = _activate[nonlinearity]

        self.embed = torch.nn.Linear(in_features, out_channels * size * size)
        self.sub_module = torch.nn.Sequential(
            *_normalization(normalization, **_kwargs(out_channels, size)[normalization]),
            *activate(),
            _Conv2dLayer(out_channels, out_channels, 3, 1, 1, use_wscale=use_wscale,
                         nonlinearity=nonlinearity, normalization=normalization,
                         **_kwargs(out_channels, size)[normalization]),
        )

        init_std, runtime_coef = get_weight(self.embed.weight.shape, gain/4, use_wscale, lrmul)
        torch.nn.init.normal_(self.embed.weight, 0., init_std)
        if self.embed.bias is not None:
            torch.nn.init.constant_(self.embed.bias, 0.)

        self.register_buffer('runtime_coef', runtime_coef)

    def forward(self, x):
        x = self.embed(x * self.runtime_coef)
        x = x.reshape(x.size(0), -1, self.size, self.size)
        x = self.sub_module(x)

        return x


class Lat2Img(torch.nn.Module):

    def __init__(self, latent_size, resolution=128, fmap_base=1024, fmap_decay=1.0, fmap_max=512,
                 num_channels=3, use_wscale=True, nonlinearity='lrelu', normalization='BN'):
        super(Lat2Img, self).__init__()

        resolution_log2 = int(log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self._nf = lambda stage: min(int(fmap_base / (2 ** (stage * fmap_decay))), fmap_max)

        growing = []
        for i in range(2, resolution_log2):
            fmap_in, fmap_out, res = self._nf(i - 1), self._nf(i), 2 ** (i + 1)

            growing.extend([
                torch.nn.UpsamplingBilinear2d(scale_factor=2),
                _Conv2dLayer(fmap_in, fmap_out, 3, 1, 1, use_wscale=use_wscale,
                             nonlinearity=nonlinearity, normalization=normalization,
                             **_kwargs(fmap_out, res)[normalization]),
                _Conv2dLayer(fmap_out, fmap_out, 3, 1, 1, use_wscale=use_wscale,
                             nonlinearity=nonlinearity, normalization=normalization,
                             **_kwargs(fmap_out, res)[normalization])
            ])

        self.sub_module = torch.nn.Sequential(
            _InputLayer(latent_size, self._nf(1), 4, use_wscale, 1., nonlinearity, normalization),
            *growing,
        )
        self.torgb = _Conv2dLayer(self._nf(resolution_log2 - 1), num_channels, 1, 1, 0,
                                  use_wscale=use_wscale, nonlinearity='none', normalization='none')

    def forward(self, x, retain_bud: bool = False):
        bud = self.sub_module(x)
        flower = self.torgb(bud)

        return [flower, (bud, flower)][retain_bud]


class Img2Dis(torch.nn.Module):

    def __init__(self, num_dis, resolution=128, fmap_base=1024, fmap_decay=1.0, fmap_max=512,
                 num_channels=3, use_wscale=True, nonlinearity='lrelu', normalization='LN'):
        super(Img2Dis, self).__init__()

        resolution_log2 = int(log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self._nf = lambda stage: min(int(fmap_base / (2 ** (stage * fmap_decay))), fmap_max)

        sieving = []
        for i in range(resolution_log2 - 1, 0, -1):
            fmap_in, fmap_out, res = self._nf(i), self._nf(i - 1), 2 ** (i + 1)

            sieving.append(_Conv2dLayer(fmap_in, fmap_in, 3, 1, 1, use_wscale=use_wscale,
                                        nonlinearity=nonlinearity, normalization=normalization,
                                        **_kwargs(fmap_in, res)[normalization]),)
            if i > 1:
                sieving.append(_Conv2dLayer(fmap_in, fmap_out, 3, 2, 1, use_wscale=use_wscale,
                                            nonlinearity=nonlinearity, normalization=normalization,
                                            **_kwargs(fmap_out, res//2)[normalization]),)
            else:
                sieving.extend([
                    _Conv2dLayer(fmap_in, fmap_in, 4, 1, 0, use_wscale=use_wscale,
                                 nonlinearity=nonlinearity, normalization=normalization,
                                 **_kwargs(fmap_in, res//4)[normalization]),
                    _Conv2dLayer(fmap_in, num_dis, 1, 1, 0, use_wscale=use_wscale,
                                 nonlinearity='none', normalization='none'),
                ])

        self.fromrgb = _Conv2d(num_channels, self._nf(resolution_log2 - 1),
                               1, 1, 0, gain=1., use_wscale=use_wscale)
        self.sub_moudle = torch.nn.Sequential(*sieving)

    def forward(self, x):
        x = self.fromrgb(x)
        x = self.sub_moudle(x)

        return x.reshape(x.size(0), -1)


__all__ = ["Lat2Img", "Img2Dis"]
