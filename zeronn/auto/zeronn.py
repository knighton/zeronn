import torch


MOM = 7 / 8
EPS = 1 / 1024
INIT_LO = -1 / 16
INIT_HI = 1 / 16


def rand(a, z, s):
    r = torch.rand(*s)
    return r * (z - a) + a


class Module(object):
    def __init__(self):
        self.parameters = []

    def parameter(self, x):
        x.requires_grad_(True)
        self.parameters.append(x)
        return x

    def each_inner_parameter(self):
        yield from ()

    def each_parameter(self):
        yield from self.parameters
        yield from self.each_inner_parameter()

    def zero_grad(self):
        for param in self.each_parameter():
            if param.grad is None:
                continue
            param.grad.zero_()

    def forward(self, x, is_t):
        return x

    def __call__(self, x, is_t):
        return self.forward(x, is_t)

    def update_step(self, lr):
        for param in self.each_parameter():
            if param.grad is None:
                continue
            param.data -= lr * param.grad


class BatchNormNd(Module):
    def __init__(self, dim, mom=MOM, eps=EPS, ndim=None):
        super().__init__()
        self.dim = dim
        self.mom = mom
        self.eps = eps
        self.ndim = ndim
        self.gamma = self.parameter(torch.ones(dim, 1))
        self.beta = self.parameter(torch.zeros(dim, 1))
        self.mov_mean = torch.zeros(dim, 1)
        self.mov_std = torch.ones(dim, 1)

    def forward(self, x, is_t):
        if self.ndim is not None:
            assert x.ndim == self.ndim + 2
        x = x.transpose(0, 1)
        s = x.shape
        x = x.reshape(x.shape[0], -1)
        if is_t:
            x_mean = x.mean(1, keepdim=True)
            x_ctr = x - x_mean
            x_var = (x_ctr ** 2).mean(1, keepdim=True)
            x_std = (x_var + self.eps).sqrt()
            x_norm = x_ctr / x_std
            self.mov_mean = self.mom * self.mov_mean + \
                (1 - self.mom) * x_mean.detach()
            self.mov_std = self.mom * self.mov_std + \
                (1 - self.mom) * x_std.detach()
        else:
            x_norm = (x - self.mov_mean) / self.mov_std
        y = x_norm * self.gamma + self.beta
        y = y.view(*s)
        y = y.transpose(0, 1)
        return y.contiguous()


class BatchNorm0d(BatchNormNd):
    def __init__(self, dim, mom=MOM, eps=EPS):
        super().__init__(dim, mom, eps, 0)


class BatchNorm1d(BatchNormNd):
    def __init__(self, dim, mom=MOM, eps=EPS):
        super().__init__(dim, mom, eps, 1)


class BatchNorm2d(BatchNormNd):
    def __init__(self, dim, mom=MOM, eps=EPS):
        super().__init__(dim, mom, eps, 2)


class BatchNorm3d(BatchNormNd):
    def __init__(self, dim, mom=MOM, eps=EPS):
        super().__init__(dim, mom, eps, 3)


def get_coords(x, ndim):
    if isinstance(x, int):
        return (x,) * ndim

    if isinstance(x, (tuple, list)):
        assert len(x) == ndim
        return tuple(x)

    assert False


class ConvNd(Module):
    def __init__(self, in_channels, out_channels, face, stride, pad, ndim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.face = face
        self.stride = stride
        self.pad = pad
        self.ndim = ndim

        face = get_coords(face, ndim)
        shape = (in_channels, out_channels) + face
        x = rand(INIT_LO, INIT_HI, shape)
        x = x.transpose(0, 1)
        self.weight = self.parameter(x)

        self.bias = self.parameter(rand(INIT_LO, INIT_HI, (out_channels,)))

        self.conv = getattr(F, 'conv%dd' % ndim)

    def forward(self, x, is_t):
        return self.conv(x, self.weight, self.bias, self.stride, self.pad)


class Conv1d(ConvNd):
    def __init__(self, in_channels, out_channels, face, stride, pad):
        super().__init__(in_channels, out_channels, face, stride, pad, 1)


class Conv2d(ConvNd):
    def __init__(self, in_channels, out_channels, face, stride, pad):
        super().__init__(in_channels, out_channels, face, stride, pad, 2)


class Conv3d(ConvNd):
    def __init__(self, in_channels, out_channels, face, stride, pad):
        super().__init__(in_channels, out_channels, face, stride, pad, 3)


class Debug(Module):
    def __init__(self, text=None):
        super().__init__()
        self.text = text or ''

    def forward(self, x, is_t):
        m = x.mean()
        s = x.std()
        print('%s %s %.6f %.6f' % (self.text, tuple(x.shape), m, s))
        return x


class Dropout(Module):
    def __init__(self, rate=0.5):
        super().__init__()
        assert 0 <= rate <= 1
        self.rate = rate

    def forward(self, x, is_t):
        if is_t:
            noise = rand(0, 1, x.shape)
            mask = (self.rate <= noise).type(x.dtype)
            x = x * mask / self.rate
        return x


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = self.parameter(rand(INIT_LO, INIT_HI, (in_dim, out_dim)))
        self.bias = self.parameter(rand(INIT_LO, INIT_HI, (out_dim,)))

    def forward(self, x, is_t):
        return torch.einsum('ni,io->no', [x, self.weight]) + self.bias


class ReLU(Module):
    def forward(self, x, is_t):
        return x.clamp(min=0)


class Reshape(Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x, is_t):
        s = (x.shape[0],) + self.shape
        return x.reshape(*s)


class Flatten(Reshape):
    def __init__(self):
        super().__init__(-1)


class Sequence(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def each_inner_parameter(self):
        for layer in self.layers:
            yield from layer.each_parameter()

    def forward(self, x, is_t):
        for layer in self.layers:
            x = layer.forward(x, is_t)
        return x


class Softmax(Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x, is_t):
        x = x - x.max(self.axis, keepdim=True).values
        return x.softmax(self.axis)
