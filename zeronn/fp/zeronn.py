import torch


MOM = 7 / 8
EPS = 1 / 1024
INIT_LO = -1 / 16
INIT_HI = 1 / 16


def rand(a, z, s):
    r = torch.rand(*s)
    return r * (z - a) + a


class Parameter(object):
    def __init__(self, data):
        self.shape = data.shape
        self.data = data
        self.grad = torch.zeros_like(data)


class Module(object):
    def __init__(self):
        self.parameters = []

    def parameter(self, x):
        x = Parameter(x)
        self.parameters.append(x)
        return x

    def each_inner_parameter(self):
        yield from ()

    def each_parameter(self):
        yield from self.parameters
        yield from self.each_inner_parameter()

    def zero_grad(self):
        for param in self.each_parameter():
            param.grad = torch.zeros_like(param.data)

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
            self.x = x
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
        y = x_norm * self.gamma.data + self.beta.data
        y = y.reshape(*s)
        y = y.transpose(0, 1)
        return y.contiguous()

    def backward(self, dy):
        dy = dy.transpose(0, 1)
        s = dy.shape
        dy = dy.reshape(dy.shape[0], -1)
        x_mean = self.x.mean(1, keepdim=True)
        x_ctr = self.x - x_mean
        x_var = (x_ctr ** 2).mean(1, keepdims=True)
        x_std = (x_var + self.eps).sqrt()
        x_norm = x_ctr / x_std
        self.beta.grad = dy.sum(1, keepdim=True)
        self.gamma.grad = (dy * x_norm).sum(1, keepdim=True)
        dx = dy - dy.mean(1, keepdim=True) - \
            x_ctr / x_var * (dy * x_ctr).mean(1, keepdim=True)
        dx = dx * self.gamma.data / x_std
        dx = dx.reshape(*s)
        return dx.transpose(0, 1)


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
            self.mask = (self.rate <= noise).type(x.dtype)
            x = x * self.mask / self.rate
        return x

    def backward(self, dy):
        return dy * self.mask * self.rate


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = self.parameter(rand(INIT_LO, INIT_HI, (in_dim, out_dim)))
        self.bias = self.parameter(rand(INIT_LO, INIT_HI, (out_dim,)))

    def forward(self, x, is_t):
        if is_t:
            self.x = x
        return torch.einsum('ni,io->no', [x, self.weight.data]) + \
            self.bias.data

    def backward(self, dy):
        self.weight.grad = torch.einsum('ni,no->io', [self.x, dy])
        self.bias.grad = dy.sum(0)
        return torch.einsum('no,io->ni', [dy, self.weight.data])


class ReLU(Module):
    def forward(self, x, is_t):
        if is_t:
            self.x = x
        return x.clamp(min=0)

    def backward(self, dy):
        dx = dy.clone()
        dx[self.x < 0] = 0
        return dx


class Reshape(Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x, is_t):
        if is_t:
            self.x_shape = x.shape
        s = (x.shape[0],) + self.shape
        return x.reshape(*s)

    def backward(self, dy):
        return dy.view(*self.x_shape)


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

    def backward(self, dy):
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy


class Softmax(Module):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x, is_t):
        x = x - x.max(self.axis, keepdim=True).values
        return x.softmax(self.axis)

    def backward(self, dy):
        return dy
