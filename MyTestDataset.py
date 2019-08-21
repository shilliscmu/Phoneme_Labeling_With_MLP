from torch.utils import data
import numpy as np
import itertools
from collections import deque


class MyTestDataset(data.Dataset):
    def __init__(self, utterances, one_sided_context_size):
        self.left_border_frame_indices = [len(utt) for utt in utterances[0]]
        self.left_border_frame_indices.insert(0,0)
        self.left_border_frame_indices.pop()
        self.left_border_frame_indices = np.cumsum(self.left_border_frame_indices)
        self.right_border_frame_indices = np.array([index - 1 for index in self.left_border_frame_indices])
        self.right_border_frame_indices = np.delete(self.right_border_frame_indices, 0)

        self.frames = list(itertools.chain.from_iterable(map(lambda utt: utt, utterances[0])))
        self.num_frames = len(self.frames)
        self.one_sided_context_size = one_sided_context_size

        self.right_border_frame_indices = np.append(self.right_border_frame_indices, self.num_frames-1)
        self.border_frames = np.concatenate((self.left_border_frame_indices, self.right_border_frame_indices))
        self.border_frames = set(self.border_frames)

        self.left_border_frame_indices = set(self.left_border_frame_indices)
        self.right_border_frame_indices = set(self.right_border_frame_indices)

        self.pad = np.zeros_like(self.frames[0])
        # for k in range(self.one_sided_context_size):
        #     self.frames.insert(0, pad)
        #     self.frames.append(pad)

        print('We have ' + repr(len(self.frames)) + ' frames.')

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        # x = []
        # x.append(self.frames[index])
        # if self.one_sided_context_size > 0:
        #     x = deque(x)
        #     for j in range(1, self.one_sided_context_size +1):
        #         x.appendleft(self.frames[index-j])
        #         x.append(self.frames[index+j])
        #     x = np.concatenate(list(x))

        # index = index+self.one_sided_context_size

        frame_borders = np.arange(index - self.one_sided_context_size, index + self.one_sided_context_size + 1)
        needs_padding = [border in self.border_frames for border in frame_borders]
        if True in needs_padding and needs_padding.index(True) != 0 and needs_padding.index(True) != len(
                needs_padding) - 1:
            if needs_padding.count(True) > 1:
                a_num_paddings = needs_padding.index(True)
                if a_num_paddings < self.one_sided_context_size:
                    num_paddings = a_num_paddings + 1
                else:
                    num_paddings = a_num_paddings
            else:
                num_paddings = needs_padding.index(True)
            if num_paddings > self.one_sided_context_size:
                needs_padding.reverse()
                num_paddings = needs_padding.index(True) + 1
                pad = np.vstack([self.pad] * num_paddings)
                x = self.frames[
                    index - self.one_sided_context_size:index + self.one_sided_context_size + 1 - num_paddings]
                try:
                    x = np.concatenate((x, pad))
                except ValueError:
                    print(index)
            elif num_paddings == self.one_sided_context_size:
                pad = np.vstack([self.pad] * num_paddings)
                if index in self.left_border_frame_indices:
                    x = self.frames[
                        index - self.one_sided_context_size + num_paddings:index + self.one_sided_context_size + 1]
                    x = np.concatenate((pad, x))
                else:
                    x = self.frames[
                        index - self.one_sided_context_size:index + self.one_sided_context_size + 1 - num_paddings]
                    x = np.concatenate((x, pad))
            else:
                pad = np.vstack([self.pad] * num_paddings)
                x = self.frames[
                    index - self.one_sided_context_size + num_paddings:index + self.one_sided_context_size + 1]
                # x.insert(pad, 0)
                x = np.concatenate((pad, x))
        else:
            x = self.frames[index - self.one_sided_context_size:index + self.one_sided_context_size + 1]
        x = np.ravel(x)
        return x