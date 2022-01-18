from torch.utils.data import DataLoader

from SOL import extractor
from SOL.DLoader import DLoader, TARGET

import numpy as np
import torch as th




def ttest (dataset):
    normalizer = dataset.normalizer[TARGET]


    test_loader = DataLoader (dataset, batch_size=128)
    with th.no_grad ():
        sum = 0
        for data, target in test_loader:
            pred = np.random.normal(loc=normalizer.mean, scale=normalizer.var)
            target = target.numpy ()
            abs = dataset.abs_diff (pred, target)
            sum += np.sum (abs)
        print ('Eval diff  = {:>.6}'.format (sum / dataset.__len__ ()))

if __name__ == '__main__':
    data, valid = extractor.load_ml()
    test = valid[:3]
    data = DLoader(data)
    valid = DLoader(valid, data.normalizer)
    ttest(valid)




