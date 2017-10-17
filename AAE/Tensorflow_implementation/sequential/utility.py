import os, shutil
import operator, functools


def check_dir(_dir, is_restart=False):
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    elif is_restart:
        shutil.rmtree(_dir)
        os.makedirs(_dir)


def nCr(n, r):
    r = min(n - r, r)
    if r == 0:
        return 1
    numer = functools.reduce(operator.mul, range(n, n - r, -1))
    denom = functools.reduce(operator.mul, range(1, r + 1, 1))
    return int(numer / denom)
