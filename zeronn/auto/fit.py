from argparse import ArgumentParser
import torch
from torch.nn import functional as F
from torch.optim import SGD

from .zeronn import *


def parse_args():
    x = ArgumentParser()
    x.add_argument('--num_epochs', type=int, default=10)
    x.add_argument('--rounds_per_epoch', type=int, default=20)
    x.add_argument('--train_per_round', type=int, default=5)
    x.add_argument('--eval_per_round', type=int, default=1)
    x.add_argument('--batch_size', type=int, default=32)
    x.add_argument('--in_dim', type=int, default=128)
    x.add_argument('--out_dim', type=int, default=8)
    x.add_argument('--signal', type=float, default=0.25)
    x.add_argument('--lr', type=float, default=1 / 128)
    return x.parse_args()


def get_block(in_dim, out_dim):
    return Sequence(
        ReLU(),
        Dropout(),
        Linear(in_dim, out_dim),
        BatchNorm0d(out_dim),
    )


def get_model(in_dim, out_dim):
    a = 64
    b = 32
    c = 16
    return Sequence(
        Linear(in_dim, a),
        BatchNorm0d(a),
        get_block(a, b),
        get_block(b, c),
        ReLU(),
        Linear(c, out_dim),
    )


def make_batch(batch_size, in_dim, out_dim, signal):
    assert out_dim <= in_dim
    assert in_dim % out_dim == 0
    num_repeats = in_dim // out_dim
    y = torch.randint(0, out_dim, (batch_size,))
    y_one_hot = torch.eye(out_dim)[y]
    r = torch.rand(batch_size, num_repeats, out_dim)
    x = signal * y_one_hot.unsqueeze(1) + (1 - signal) * r
    return x.view(batch_size, -1), y



def main(args):
    f = get_model(args.in_dim, args.out_dim)
    opt = SGD(f.each_parameter(), args.lr)
    get_batch = lambda: make_batch(args.batch_size, args.in_dim, args.out_dim,
                                   args.signal)
    for epoch_id in range(args.num_epochs):
        taa = []
        vaa = []
        for round_id in range(args.rounds_per_epoch):
            for batch_id in range(args.train_per_round):
                opt.zero_grad()
                x, y_gold = get_batch()
                y_pred = f(x, True)
                e = F.cross_entropy(y_pred, y_gold)
                e.backward()
                opt.step()
                a = (y_pred.argmax(1) == y_gold).float().mean()
                taa.append(a.item())
            with torch.no_grad():
                for batch_id in range(args.eval_per_round):
                    x, y_gold = get_batch()
                    y_pred = f(x, False)
                    a = (y_pred.argmax(1) == y_gold).float().mean()
                    vaa.append(a.item())
        ta = sum(taa) / len(taa)
        va = sum(vaa) / len(vaa)
        s = '%7d %7.3f %7.3f' % (epoch_id, 100 * ta, 100 * va)
        print(s)


if __name__ == '__main__':
    main(parse_args())
