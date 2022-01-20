import random
import time
from typing import Dict, Union

from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from torch.utils.data import DataLoader

from SOL import extractor
from SOL.DLoader import DLoader
from SOL.model import *
from stable_baselines3.common import utils

device = 'cuda' if th.cuda.is_available() else 'cpu'
th.manual_seed(777)
if device == 'cuda':
    th.cuda.manual_seed_all(777)


learning_rate = 0.001
batch_size = 64
num_classes = 10
epochs = 20000


# def obs_as_tensor(
#     obs: Union[np.ndarray, Dict[Union[str, int], np.ndarray]], device: th.device
# ) -> Union[th.Tensor, TensorDict]:
#     if isinstance(obs, th.Tensor):
#         return obs.to(device)
#     elif isinstance(obs, dict):
#         return {key: _obs.to(device) for (key, _obs) in obs.items()}
#     else:
#         raise Exception(f"Unrecognized type of observation {type(obs)}")
#


def unit(arr, thre = 0.001):
    plus = np.where(arr > thre, 1, 0)
    minus = np.where(arr < -thre, -1, 0)
    return plus + minus

def trigger(pre_d, pre_p, target):

    # print("---")
    # print(np.round(target[100], 3))
    # print(np.round(pre_p[100], 3))
    # print(np.round(pre_d[100], 3))
    direct = unit(pre_d, 0.5) #upper
    anti = unit(pre_d, 0.05) #lower
    price = unit(pre_p, 0.02)
    all_sum = np.concatenate((price, direct, anti ), axis=1)
    all_sum = np.sum(all_sum, axis=-1)
    tri = np.abs(all_sum)
    tri = np.where(tri >=4, 1,0)
    return tri


def consense(pre_d, pre_p):
    direct = unit(pre_d, 0.5)
    price = unit(pre_p, 0.001)
    all_sum = np.concatenate((price, direct), axis=1)
    all_sum = np.sum(all_sum, axis=-1)
    tri = np.abs(all_sum)
    tri = np.where(tri >=3, 1,0)
    return np.sum(tri)


def trim_p(price, direct):
    price = np.minimum(price, np.array([10, 10, 0]))
    price = np.maximum(price, np.array([-10, 0, -10]))
    direct = np.minimum(direct, np.array([10, 0]))
    direct = np.maximum(direct, np.array([0, -10]))
    return price, direct

def trade(pred, target, denormali):
    pre_p, pre_d = pred[:,:3], pred[:,3:]
    pre_p = denormali(pre_p)
    pre_p, pre_d = trim_p(pre_p, pre_d)

    true = denormali(target)
    tri = trigger(pre_d, pre_p, true)
    pred_quality = consense(pre_d, pre_p)
    cnt = tri.sum()
    diff=0
    correct =0
    if cnt > 0:
        tru = true[:, 0]
        predict = pre_p[:,0]
        cor_direct = np.where(unit(tru, 0.005) == unit(predict, 0.005), 1, 0)
        correct = np.sum(cor_direct *tri)
        diff = (predict - tru) * tri
        diff = np.sum(np.abs(diff))

        test = np.where(tri ==1)
        idx = random.choice(test[0])
        print("---", idx)
        print(np.round(true[idx], 3))
        print(np.round(pre_p[idx], 3))
        print(np.round(pre_d[idx], 3))


    return diff, cnt, correct, pred_quality

def valall(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=1000)
    with th.no_grad():
        sum = 0

        for data, target, _ in test_loader:
            pred = model(data)[:,0:1].cpu().numpy()
            target = target.cpu().numpy()
            abs = dataset.abs_diff(pred, target)
            sum +=np.sum(abs)
        print('ALL Eval diff  = {:>.6}'.format(sum/dataset.__len__()))
    model.train()


def val(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=000)
    with th.no_grad():
        sum = 0
        pre_cnt = 0
        cor_d_cnt = 0
        same_direct = 0

        for data, target, direct in test_loader:

            pred = model(data).cpu().numpy()
            diff, cnt, correct, pre_same = trade(pred, target.cpu().numpy(), dataset.denorm_target)
            sum += diff
            pre_cnt +=cnt
            cor_d_cnt += correct
            same_direct += pre_same

        if pre_cnt ==0:
            print('Eval No cnt: {}(p{}, n{})  ---  Pre_X: {})'.format(dataset.__len__(), dataset.pos_direct, dataset.neg_direct, same_direct))
        else: print('Eval diff  = {:>.6}, cnt: {}({})/{}(p{}, n{})  ---   Pre_X: {})'.format(
            sum/pre_cnt, pre_cnt, cor_d_cnt, dataset.__len__(), dataset.pos_direct, dataset.neg_direct, same_direct))
    model.train()



eval_interval = 1
def train(data, testdata, validdatae):
    train_loader = DataLoader(data, batch_size=batch_size, shuffle= True)
    model = OutterModel(data.feature_len, data.seq, data.ta_len, data.ta_seq, data.base_len).to(device)
    model.train()
    # for param in model.module.parameters():
    #     param.requires_grad = False
    #
    #

    rmse = nn.MSELoss().to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-5)

    start_tim = time.time()

    for epoch in range(epochs):  # epochs수만큼 반복
        avg_loss = 0
        for data, target, direction in train_loader:
            # data = obs_as_tensor(data, device)
            # target = obs_as_tensor(target, device)
            # direction = obs_as_tensor (direction, device)
            target = th.cat([target, direction], dim=1)
            optimizer.zero_grad()
            hypothesis = model(data)
            loss = rmse(hypothesis, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss / len(train_loader)  # loss 값을 변수에 누적하고 train_loader의 개수로 나눔 = 평균

        if (epoch +1)%eval_interval ==0:
            now = time.time()

            print('==== [Epoch: {:>4}] loss = {:>.9}'.format(epoch + 1, avg_loss), int(now-start_tim))
            print("==== VALID (overlap, all) ===")
            val(model, testdata)
            val(model, validdatae)
            valall(model, testdata)
            valall(model, validdatae)
            start_tim = now


    # test
     # evaluate mode로 전환 dropout 이나 batch_normalization 해제



if __name__ == '__main__':
    data, valid = extractor.load_ml()
    test = valid[:3]
    valid = valid[3:]
    data = DLoader(data)
    valid = DLoader(valid, data.normalizer)
    test = DLoader(test, data.normalizer)
    train(data,test, valid )
