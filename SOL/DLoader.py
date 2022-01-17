import torch as th
from torch.utils.data import Dataset

from SOL import extractor


class MyDataset(Dataset):
    def __init__(self, data, seq = 5):
        self.data = data.data
        self.base = data.base
        self.target = data.price
        self.seq = seq
        self.daily_size = [day.shape[0] - seq for day in self.data]
        self.size = sum(self.daily_size)

    def __getitem__(self, index):
        index +=self.seq-1
        data = self.data[index: self.seq]
        base =


        return {'data': x, 'target': y}

    def __len__(self):
        return self.size


dataset = MyDataset()
loader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=2
)

for batch in loader:
    data = batch['data']
    target = batch['target']