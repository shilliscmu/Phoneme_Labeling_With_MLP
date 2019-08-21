import torch
from torch import Tensor

class MyCollateFn:
    def __init__(self, dim=0):
        self.dim = dim

    def pad_collate(self, batch):
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # batch = map(lambda (x, y):
        #             (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)

        batch = map(lambda xy: (pad_tensor(xy[0], pad=max_len, dim=self.dim), pad_tensor(xy[1], pad=max_len, dim=self.dim)), batch)
        xs = []
        ys = []
        for instance in batch:
            xs.append(instance[0])
            ys.append(instance[1])
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        # maps = map(lambda xy: (xy[0], xy[1]), batch)
        # xs = torch.stack(tuple(x_map), dim=0)
        # ys = torch.stack(tuple(y_map), dim=0)
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

def pad_tensor(vector, pad, dim):
    pad_size = list(vector.shape)
    pad_size[dim] = pad - vector.shape[dim]
    return torch.cat([torch.Tensor(vector).float(), torch.zeros(*pad_size)], dim=dim)