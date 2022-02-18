import random
import time


from torch.utils.data import DataLoader

from SOL import extractor
from SOL.DLoader import DLoader
from SOL.extractor import Extractor
from SOL.model import *
from CONFIG import *


device = 'cuda' if th.cuda.is_available() else 'cpu'
th.manual_seed(777)
if device == 'cuda':
    th.cuda.manual_seed_all(777)


# learning_rate = 0.001
# batch_size = 500
learning_rate = 0.002
batch_size = 512
epochs = 1000
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
    test_loader = DataLoader(dataset, batch_size=dataset.__len__())
    with th.no_grad():
        sum = 0
        multi_sum = np.zeros(dataset.spec.price_len)
        for data, target, _ in test_loader:
            pred = model(data)[:,:dataset.spec.price_len].cpu().numpy()
            pred = dataset.spec.denorm_target(pred)
            true = dataset.spec.denorm_target(target.cpu().numpy())
            diff = np.sum(np.abs(pred[:,0]-true[:,0]))
            sum += np.sum(diff)
            multi_diff = np.sum(np.abs(pred-true), axis=0)
            multi_sum = multi_sum+multi_diff

        print('Eval diff: {:>.3}'.format(sum/dataset.__len__()), "  S:", np.round(multi_sum/dataset.__len__(), 3))
    model.train()

eval_rmse = nn.MSELoss().to(device)
def val_loss(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=dataset.__len__())
    with th.no_grad():
        avg_loss = 0
        for data, target, direction in test_loader:
            target = th.cat([target, direction], dim=1)
            hypothesis = model(data)
            loss = eval_rmse(hypothesis, target)
            avg_loss += loss / len(test_loader)

    model.train()
    return avg_loss.cpu().numpy().item()


def val(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=dataset.__len__())
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
            # th.save(model.extra)
            print("SAVED: ", loss)
            self.best_loss = loss

    def load(self, spec):
        try:
            model = OutterModel(spec).to(device)
            model.load_state_dict(th.load(self.file_name))
            return model
        except:
            print("NEW MODEL")
            model = OutterModel(spec).to(device)
            model.printsummary()
            return model

def predictor_test(model):
    from SOL.predictor import Predictor
    model = Predictor(model)
    model.price_test()


def train(iter, learning_rate, traindata, testdata, validdatae, val_day_list):

    ckp = CECK()
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle= True)
    model = ckp.load(traindata.spec)

    model.train()

    rmse = nn.MSELoss().to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-4)


    start_tim = time.time()

    total_time = 0
    print("DATA SIZE", traindata.__len__() )
    for epoch in range(epochs):  # epochs수만큼 반복
        avg_loss = 0

        start = time.time()
        for data, target, direction in train_loader:

            target = th.cat([target, direction], dim=1)
            optimizer.zero_grad()
            hypothesis = model(data)
            loss = rmse(hypothesis, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss / len(train_loader)  # loss 값을 변수에 누적하고 train_loader의 개수로 나눔 = 평균

        total_time += time.time()-start
        ckp.save_best(model, avg_loss)

        if (epoch +1)%eval_interval ==0:
            now = time.time()
            print('==== [Epoch{}: {:>4}] loss = {:>.4}'.format(iter, epoch + 1, avg_loss), int(total_time), int((eval_interval*traindata.__len__())/total_time), traindata.day_cnt)
            total_time =0
            print("==== VALID (overlap, all) ===")
            valall(model, testdata)
            valall(model, validdatae)
            loss1 = val_loss(model, testdata)
            loss2 = val_loss(model, validdatae)
            print('Loss Eval: {:>.4}'.format(loss1))
            print('Loss Eval: {:>.4}'.format(loss2))
            print('AVG: {:>.4}'.format((loss1+loss2)/2))
            print("DAYS:  ", [ np.round(val_loss(model, dd),2) for dd in  val_day_list])

            # predictor_test(model)

            start_tim = now


if __name__ == '__main__':
    REF, t_data, valid, test, tri = Extractor.load()

    dataD = DLoader(t_data, REF)
    testD = DLoader(t_data, REF)
    triD = DLoader(tri, REF)
    valid_list = [DLoader([dd], REF) for dd in t_data + tri ]

    iter = 8
    for i in range(iter):
        learning_rate= learning_rate *0.5
        print(i, "LLLRRRR :", learning_rate, " for Epoch:", epochs)
        train(i, learning_rate, dataD,testD, triD, valid_list)
