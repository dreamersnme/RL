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


learning_rate = 0.00001
batch_size = 512
num_classes = 10
epochs = 3000
eval_interval = 10



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
        # print("---", idx)
        # print(np.round(true[idx], 3))
        # print(np.round(pre_p[idx], 3))
        # print(np.round(pre_d[idx], 3))


    return diff, cnt, correct, pred_quality

def valall(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=1000)
    with th.no_grad():
        sum = 0
        multi_sum = np.zeros(dataset.spec.price_len)
        for data, target, _ in test_loader:
            pred = model(data)[:,:dataset.spec.price_len].cpu().numpy()
            pred = dataset.denorm_target(pred)
            true = dataset.denorm_target(target.cpu().numpy())
            diff = np.sum(np.abs(pred[:,0]-true[:,0]))
            sum +=np.sum(diff)

            multi_diff = np.sum(np.abs(pred-true), axis=0)

            multi_sum = multi_sum+multi_diff
        print('Eval diff: {:>.3}'.format(sum/dataset.__len__()), "  S:", np.round(multi_sum/dataset.__len__(), 3))
    model.train()

eval_rmse = nn.MSELoss().to(device)
def val_loss(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=1000)
    with th.no_grad():
        avg_loss = 0
        for data, target, direction in test_loader:
            target = th.cat([target, direction], dim=1)
            hypothesis = model(data)
            loss = eval_rmse(hypothesis, target)
            avg_loss += loss / len(test_loader)
        print('Loss Eval: {:>.4}'.format(avg_loss))
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
            print('TEST No cnt: {}(p{}, n{})  ---  Pre_X: {})'.format(dataset.__len__(), dataset.pos_direct, dataset.neg_direct, same_direct))
        else: print('TEST diff: {:>.3}, cnt: {}({})/{}(p{}, n{})  ---   Pre_X: {})'.format(
            sum/pre_cnt, pre_cnt, cor_d_cnt, dataset.__len__(), dataset.pos_direct, dataset.neg_direct, same_direct))
    model.train()



class CECK():
    best_loss = 1000
    file_name = PRETRAINED
    def save_best(self, model, loss):
        loss = loss.cpu().detach().numpy()
        if loss <= self.best_loss:
            th.save(model.state_dict(), self.file_name)
            print("SAVED: ", loss)
            self.best_loss = loss

    def load(self, spec):

        try:
            model = OutterModel(spec).to(device)
            model.load_state_dict(th.load(self.file_name))
            return model
        except:
            print("NEW MODEL")
            return OutterModel(spec).to(device)


def train(data, testdata, validdatae):
    ckp = CECK()
    train_loader = DataLoader(data, batch_size=batch_size, shuffle= True)
    model = ckp.load(data.spec)

    model.train()

    rmse = nn.MSELoss().to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-4)


    start_tim = time.time()

    for epoch in range(epochs):  # epochs수만큼 반복
        avg_loss = 0
        for data, target, direction in train_loader:

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
            valall(model, testdata)
            valall(model, validdatae)
            val_loss(model, testdata)
            val_loss(model, validdatae)
            start_tim = now



if __name__ == '__main__':
    t_data, valid, test, tri = extractor.load_trainset(10)
    REF = DataSpec (t_data[0])

    data = DLoader(t_data, REF)
    test = DLoader(test, REF, data.normalizer)
    valid = DLoader(tri, REF, data.normalizer)
    train(data,test, valid)
