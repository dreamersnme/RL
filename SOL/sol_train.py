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
batch_size = 128
num_classes = 10
epochs = 2000
eval_interval = 3

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
    thre = pre_d.shape[1] *2
    direct = unit(pre_d, 0.5) #upper
    price = unit(pre_p, 0.02)

    all_sum = np.concatenate((price, direct), axis=1)
    all_sum = np.sum(all_sum, axis=-1)
    tri = np.abs(all_sum)
    tri = np.where(tri >=thre, 1,0)
    return tri


def consense(pre_d, pre_p):
    direct = unit(pre_d, 0.02) #upper
    price = unit(pre_p, 0.01)

    all_sum = np.concatenate((price, direct), axis=1)
    all_sum = np.sum(all_sum, axis=-1)
    tri = np.abs(all_sum)
    tri = np.where(tri >=10, 1,0)
    return np.sum(tri)



def trade(pred,price_len, target, denormali):

    pre_p, pre_d = pred[:,:price_len], pred[:,price_len:]
    pre_p = denormali(pre_p)

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
            pred = model(data)[:,:dataset.spec.price_len].cpu().numpy()
            pred = dataset.denorm_target(pred)
            true = dataset.denorm_target(target.cpu().numpy())
            diff = np.sum(np.abs(pred[:,0]-true[:,0]))
            sum +=np.sum(diff)
        print('ALL Eval diff  = {:>.6}'.format(sum/dataset.__len__()))
    model.train()


def val(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=2000)
    with th.no_grad():
        sum = 0
        pre_cnt = 0
        cor_d_cnt = 0
        same_direct = 0

        for data, target, direct in test_loader:

            pred = model(data).cpu().numpy()
            diff, cnt, correct, pre_same = trade(pred, dataset.spec.price_len, target.cpu().numpy(), dataset.denorm_target)
            sum += diff
            pre_cnt +=cnt
            cor_d_cnt += correct
            same_direct += pre_same

        if pre_cnt ==0:
            print('Eval No cnt: {}(p{}, n{})  ---  Pre_X: {})'.format(dataset.__len__(), dataset.pos_direct, dataset.neg_direct, same_direct))
        else: print('Eval diff  = {:>.6}, cnt: {}({})/{}(p{}, n{})  ---   Pre_X: {})'.format(
            sum/pre_cnt, pre_cnt, cor_d_cnt, dataset.__len__(), dataset.pos_direct, dataset.neg_direct, same_direct))
    model.train()




class CECK():
    best_loss = 1000
    file_name = 'best-model.pt'
    def save_best(self, model, loss):
        loss = loss.cpu().detach().numpy()
        if loss <= self.best_loss:
            th.save(model.state_dict(), self.file_name)
            print("SAVED: ", loss)
            self.best_loss = loss

    def load(self, model):
        try:
            model.load_state_dict(th.load(self.file_name))
            # print(model.state_dict())
        except:
            print("NEW MODEL")
            pass


def train(data, testdata, validdatae):
    ckp = CECK()
    train_loader = DataLoader(data, batch_size=batch_size, shuffle= True)
    model = OutterModel(data.spec).to(device)
    ckp.load(model)

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

        ckp.save_best(model, avg_loss)

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
    data = extractor.load_ml()
    valid = data[-8:]
    data = data[:-5]

    test = valid[:3]
    valid = valid[3:]


    REF = DataSpec (data[0])


    data = DLoader(data, REF)
    valid = DLoader(valid, REF, data.normalizer)
    test = DLoader(test, REF, data.normalizer)
    train(data,test, valid )
