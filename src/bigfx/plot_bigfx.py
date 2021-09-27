from collections import defaultdict
from matplotlib import pyplot as plt
import os
import sys


def main(in_f, out_d):
    k2vvv = defaultdict(list)
    for line in open(in_f):
        ss = line.split()
        k = ss[0]
        vv = list(map(float, ss[1:]))
        k2vvv[k].append(vv)
    for k in sorted(k2vvv):
        vvv = k2vvv[k]
        x, y_gold, y_pred = zip(*vvv)
        plt.clf()
        plt.title(k)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.plot(x, y_gold, color='blue', linewidth=1)
        plt.plot(x, y_pred, color='red', linewidth=0.5)
        f = os.path.join(out_d, k + '.png')
        plt.minorticks_on()
        plt.grid(True, which='major', linestyle='-', linewidth=0.25)
        plt.grid(True, which='minor', linestyle='--', linewidth=0.25)
        plt.savefig(f, dpi=800)
        

if __name__ == '__main__':
    main(*sys.argv[1:])
