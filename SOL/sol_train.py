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

def trigger(pre_d, pre_p):
    direct = unit(pre_d, 0.33)
    price_direct = unit(pre_p, 0.02)
    return ((direct + price_direct)/2).astype(int)


def trade(pre_p, pre_d, target, denormali, tru_direct):
    pre_d = pre_d[:, 0]
    pre_p = denormali(pre_p)[:, 0]
    p_d = unit(pre_d)
    p_p = unit(pre_p)
    same_direct =  p_d - p_p
    same_direct = np.where(same_direct != 0)[0].shape[0]

    pre_d = trigger(pre_d, pre_p)
    tri = np.abs(pre_d)
    cnt = tri.sum()

    diff=0
    correct =0
    if cnt > 0:
        tru = denormali(target)[:, 0]
        cor_direct = np.where(unit(tru, 0.005) == pre_d, 1, 0)
        correct = np.sum(cor_direct *tri)

        diff = (pre_p - tru) * tri
        diff = np.sum(np.abs(diff))
    return diff, cnt, correct, same_direct

def valall(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=128)
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
    test_loader = DataLoader(dataset, batch_size=1000)
    with th.no_grad():
        sum = 0
        pre_cnt = 0
        cor_d_cnt = 0
        same_direct = 0

        for data, target, direct in test_loader:
            # data = obs_as_tensor(data, device)
            pred = model(data).cpu().numpy()
            pre_p, pre_d = pred[:,0:1], pred[:,1:2]
            diff, cnt, correct, pre_same = trade(pre_p, pre_d, target.cpu().numpy(), dataset.denorm_target, direct)
            sum += diff
            pre_cnt +=cnt
            cor_d_cnt += correct
            same_direct += pre_same

        if pre_cnt ==0:
            print('Eval No cnt: {}(p{}, n{})  ---  Pre_X: {})'.format(dataset.__len__(), dataset.pos_direct, dataset.neg_direct, same_direct))
        else: print('Eval diff  = {:>.6}, cnt: {}({})/{}(p{}, n{})  ---   Pre_X: {})'.format(
            sum/pre_cnt, pre_cnt, cor_d_cnt, dataset.__len__(), dataset.pos_direct, dataset.neg_direct, same_direct))
    model.train()



eval_interval = 5
def train(data, testdata, validdatae):
    train_loader = DataLoader(data, batch_size=batch_size, shuffle= True)
    model = OutterModel(data.feature_len, data.seq, data.ta_len, data.ta_seq, data.base_len).to(device)
    model.train()


    rmse = nn.MSELoss().to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-5)
    print(model)
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
