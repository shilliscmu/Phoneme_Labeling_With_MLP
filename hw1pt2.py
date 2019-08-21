import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import numpy as np

from Flexbile_MLP import Flexible_MLP
from wsj_loader import WSJ


def main():
    os.environ["WSJ_PATH"] = "/home/ubuntu/hw1 Speech Processing/pt2_data"
    loader = WSJ()
    train = loader.train
    print('Train')
    print('X Shape:', train[0].shape)
    print('Y Shape:', train[1].shape)

    val = loader.dev
    print('Val')
    print('X Shape:', val[0].shape)
    print('Y Shape:', val[1].shape)

    test = loader.test
    print('Test')
    print('X Shape:', test[0].shape)

    load_and_test = sys.argv[1]
    model_name = sys.argv[2]

    mlp = Flexible_MLP(12, (40, 512, 1024, 1024, 512), (2, 2, 2, 2, 2), 138, load_and_test, model_name)
    print('Initialized model')
    if load_and_test == "False":
        mlp.train(train, val, 14, True, 0.01)
    out = mlp.test(test, True)
    print(out)


if __name__ == '__main__':
    main()