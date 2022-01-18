from typing import Dict, Union

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader

from SOL import extractor
from SOL.DLoader import DLoader
from SOL.model import *
from stable_baselines3.common import utils

device = 'cuda' if th.cuda.is_available() else 'cpu'
th.manual_seed(777)
if device == 'cuda':
    th.cuda.manual_seed_all(777)


learning_rate = 0.005
batch_size = 128
num_classes = 10
epochs = 10000


def obs_as_tensor(
    obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: th.device
) -> Union[th.Tensor, TensorDict]:
    if isinstance(obs, th.Tensor):
        return obs.to(device)
    elif isinstance(obs, dict):
        return {key: _obs.to(device) for (key, _obs) in obs.items()}
    else:
        raise Exception(f"Unrecognized type of observation {type(obs)}")

def val(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=128)
    with th.no_grad():
        sum = 0

        for data, target in test_loader:
            data = obs_as_tensor(data, device)
            pred = model(data).cpu().numpy()
            target = target.numpy()
            abs = dataset.abs_diff(pred, target)
            sum +=np.sum(abs)
        print('Eval diff  = {:>.6}'.format(sum/dataset.__len__()))
    model.train()



eval_interval = 5
def train(data, testdata, validdatae):
    train_loader = DataLoader(data, batch_size=batch_size, pin_memory=True, shuffle= True)
    model = OutterModel(data.feature_len, data.seq, data.ta_len, data.ta_seq, data.base_len).to(device)
    model.train()

    rmse = RMSELoss().to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-5)

    for epoch in range(epochs):  # epochs수만큼 반복
        avg_loss = 0
        for data, target in train_loader:
            data = obs_as_tensor(data, device)
            target = obs_as_tensor(target, device)
            optimizer.zero_grad()
            hypothesis = model(data)
            loss = rmse(hypothesis, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss / len(train_loader)  # loss 값을 변수에 누적하고 train_loader의 개수로 나눔 = 평균

        if (epoch +1)%eval_interval ==0:
            print('[Epoch: {:>4}] loss = {:>.9}'.format(epoch + 1, avg_loss))
            print("==== TEST ===")
            val(model, testdata)
            print("==== VALID ===")
            val(model, validdatae)

    # test
     # evaluate mode로 전환 dropout 이나 batch_normalization 해제



if __name__ == '__main__':
    data, valid = extractor.load_ml()
    test = valid[:3]
    data = DLoader(data)
    valid = DLoader(valid, data.normalizer)
    test = DLoader(test, data.normalizer)
    train(data,test, valid )
