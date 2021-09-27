import torch


MOM = 7 / 8
EPS = 1 / 1024
INIT_LO = -1 / 16
INIT_HI = 1 / 16


fx_dtype = torch.int32


def fx_div(a, b):
    div = torch.div(a, b, rounding_mode='floor')
    mod = a % b
    r = torch.randint(0, 1 << 31, div.shape) % b
    return div + (r < mod)


def i_sqrt(x):
    if isinstance(x, torch.Tensor):
        assert x.dtype == fx_dtype
    elif isinstance(x, int):
        x = torch.tensor([x], dtype=fx_dtype)
    else:
        assert False
    r = torch.zeros_like(x)
    a = 2 ** 30
    while a:
        b = (r + a <= x).type(fx_dtype)
        x = b * (x - r - a) + (1 - b) * x
        r_half = fx_div(r, 2)
        r = b * (r_half + a) + (1 - b) * r_half
        a //= 4
    return r


fx_one = 1 << 10
fx_one_sqrt = i_sqrt(fx_one).item()
fx_min = -1 << (30 - 10)
fx_max = 1 << (30 - 10)


def fx_from_fp(x):
    if isinstance(x, torch.Tensor):
        return (x * fx_one).type(fx_dtype)
    elif isinstance(x, (int, float)):
        return int(x * fx_one)
    else:
        assert False


def fp_from_fx(x):
    if isinstance(x, torch.Tensor):
        return x.type(torch.float32) / fx_one
    elif isinstance(x, (int, float)):
        return float(x) / fx_one
    else:
        assert False


def fx_sq(x):
    return fx_div(x * x, fx_one)


def fx_sqrt(x):
    return i_sqrt(x) * fx_one_sqrt


def fx_exp(x):
    x = fx_one + fx_div(x, 64)
    for i in range(6):
        x = fx_sq(x)
    return x


def fx_sum(x, axis=None, keepdim=False):
    return x.sum(axis, keepdim, dtype=x.dtype)


def fx_mean(x, axis=None, keepdim=False):
    if axis is None:
        d = x.numel()
    else:
        d = x.shape[axis]
    x = fx_sum(x, axis, keepdim)
    return fx_div(x, d)


def fx_matmul(a, b):
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    x = fx_sum(a.unsqueeze(2) * b.unsqueeze(0), 1)
    return fx_div(x, fx_one)


def fx_zeros(*shape):
    return torch.zeros(*shape, dtype=fx_dtype)


def fx_ones(*shape):
    return torch.full(shape, fx_one, dtype=fx_dtype)


def fx_rand(a, z, s):
    x = torch.rand(*s)
    x = x * (z - a) + a
    return fx_from_fp(x)


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
        lr = fx_from_fp(lr)
        for param in self.each_parameter():
            if param.grad is None:
                continue
            param.data -= fx_div(lr * param.grad, fx_one)


class BatchNormNd(Module):
    def __init__(self, dim, mom=MOM, eps=EPS, ndim=None):
        super().__init__()
        self.dim = dim
        self.mom = fx_from_fp(mom)
        self.eps = fx_from_fp(eps)
        self.ndim = ndim
        self.gamma = self.parameter(fx_ones(dim, 1))
        self.beta = self.parameter(fx_zeros(dim, 1))
        self.mov_mean = fx_zeros(dim, 1)
        self.mov_std = fx_ones(dim, 1)

    def forward(self, x, is_t):
        if self.ndim is not None:
            assert x.ndim == self.ndim + 2
        x = x.transpose(0, 1)
        s = x.shape
        x = x.reshape(x.shape[0], -1)
        if is_t:
            self.x = x
            x_mean = fx_mean(x, 1, keepdim=True)
            x_ctr = x - x_mean
            x_var = fx_mean(fx_sq(x_ctr), 1, keepdim=True)
            x_std = fx_sqrt(x_var + self.eps)
            x_norm_num = x_ctr
            x_norm_den = x_std
            self.mov_mean = fx_div(self.mom * self.mov_mean +
                                   (fx_one - self.mom) * x_mean, fx_one)
            self.mov_std = fx_div(self.mom * self.mov_std +
                                  (fx_one - self.mom) * x_std, fx_one)
        else:
            x_norm_num = x - self.mov_mean
            x_norm_den = self.mov_std
        y = fx_div(x_norm_num * self.gamma.data, x_norm_den) + self.beta.data
        y = y.reshape(*s)
        y = y.transpose(0, 1)
        return y.contiguous()

    def backward(self, dy):
        dy = dy.transpose(0, 1)
        s = dy.shape
        dy = dy.reshape(dy.shape[0], -1)
        x_mean = fx_mean(self.x, 1, keepdim=True)
        x_ctr = self.x - x_mean
        x_var = fx_mean(fx_sq(x_ctr), 1, keepdim=True)
        x_std = fx_sqrt(x_var + self.eps)
        x_norm_num = x_ctr
        x_norm_den = x_std
        self.beta.grad = fx_sum(dy, 1, keepdim=True)
        self.gamma.grad = fx_sum(fx_div(dy * x_norm_num, x_norm_den),
                                1, keepdim=True)
        dx = dy - fx_mean(dy, 1, keepdim=True) - fx_div(
            x_ctr * fx_mean(dy * x_ctr, 1, keepdim=True), x_var * fx_one)
        dx = fx_div(dx * self.gamma.data, x_std)
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
        self.rate = fx_from_fp(rate)

    def forward(self, x, is_t):
        if is_t:
            noise = fx_rand(0, 1, x.shape)
            self.mask = (self.rate <= noise).type(x.dtype)
            x = fx_div(x * self.mask * fx_one, self.rate)
        return x

    def backward(self, dy):
        return fx_div(dy * self.mask * self.rate, fx_one)


class Linear(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = self.parameter(fx_rand(INIT_LO, INIT_HI,
                                             (in_dim, out_dim)))
        self.bias = self.parameter(fx_rand(INIT_LO, INIT_HI, (out_dim,)))

    def forward(self, x, is_t):
        if is_t:
            self.x = x
        return fx_matmul(x, self.weight.data) + self.bias.data

    def backward(self, dy):
        self.weight.grad = fx_matmul(self.x.transpose(0, 1), dy)
        self.bias.grad = fx_sum(dy, 0)
        return fx_matmul(dy, self.weight.data.transpose(0, 1))


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
        x = fx_exp(x)
        a = x * fx_one
        b = x.sum(self.axis, keepdim=True)
        return fx_div(a, b)

    def backward(self, dy):
        return dy
