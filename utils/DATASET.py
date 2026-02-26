from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
import torch
from torch_geometric.data import Batch
import numpy as np
import math

class _Dataset(Dataset):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        return data, label

class BalancedBatchSampler(Sampler):
    def __init__(self, pos_indices, neg_indices, batch_size, data_source):
        
        super().__init__(data_source)
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.batch_size = batch_size
        self.pos_ratio = len(pos_indices) / (len(pos_indices) + len(neg_indices))

        self.pos_per_batch = math.ceil(batch_size * self.pos_ratio)
        self.neg_per_batch = batch_size - self.pos_per_batch

    def __iter__(self):
        my_list = list()
        pos_indices = np.random.permutation(self.pos_indices)
        neg_indices = np.random.permutation(self.neg_indices)

        pos_idx, neg_idx = 0, 0
        while pos_idx <= len(pos_indices) and neg_idx <= len(neg_indices):
            batch_pos = pos_indices[pos_idx:pos_idx + self.pos_per_batch]
            batch_neg = neg_indices[neg_idx:neg_idx + self.neg_per_batch]
            pos_idx += self.pos_per_batch
            neg_idx += self.neg_per_batch
            pos_dis = len(pos_indices) - pos_idx
            neg_dis = len(neg_indices) - neg_idx

            batch = np.concatenate([batch_pos, batch_neg])
            np.random.shuffle(batch)
            my_list.extend(list(batch))

            if pos_dis < self.pos_per_batch or neg_dis < self.neg_per_batch:
                batch_pos = pos_indices[pos_idx:]
                batch_neg = neg_indices[neg_idx:]
                batch = np.concatenate([batch_pos, batch_neg])
                np.random.shuffle(batch)
                my_list.extend(list(batch))
                break

        return iter(my_list)

    def __len__(self):
        return len(self.pos_indices) + len(self.neg_indices)

def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])

    return batchA, batchB, batchC, batchD
