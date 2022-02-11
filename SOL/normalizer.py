from typing import Tuple

import numpy as np



class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class Standardizer(RunningMeanStd):
    epsilon: float = 1e-4
    def __init__(self, shape):
        super(Standardizer,self).__init__(shape=shape)
    def norm(self, obs):
        return np.clip(((obs - self.mean) / np.sqrt(self.var + self.epsilon)).astype(np.float32), -5, 5)
    def denorm(self, data):
        return ( np.clip(data, -5, 5) * np.sqrt (self.var + self.epsilon)) + self.mean


class Normalizer:
    norm_val = 0.95
    def __init__(self, shape: Tuple[int, ...] = ()):
        self.min = None
        self.max = None

    def update(self, arr: np.ndarray) -> None:
        batch_min = np.min(arr, axis=0)
        batch_max = np.max(arr, axis=0)
        self.min = batch_min if self.min is None else np.minimum(batch_min, self.min)
        self.max = batch_max if self.max is None else np.maximum(batch_max, self.max)

        self.gap = self.max - self.min


    def norm(self, obs):
        norm1 = ( (obs - self.min) / self.gap)
        norm1 = self.norm_val * norm1
        norm2 = np.clip(norm1, 0, 1)
        return norm2

    def denorm(self, data):
        cliped  = np.clip(data, 0, 1)
        denorm1 =  cliped / self.norm_val
        denorm2 = (denorm1 * self.gap) + self.min
        return denorm2


class BothScaler:
    def __init__(self, shape: Tuple[int, ...] = ()):
        shape = [ int(a/2) for a in list(shape)]
        shape = tuple(shape)
        self.NORMALI = Normalizer(shape=shape)
        self.STAND = Standardizer(shape=shape)

    def update(self, arr: np.ndarray) -> None:
        self.NORMALI.update(arr)
        self.STAND.update(arr)

    def norm(self, obs):
        norm1 = self.NORMALI.norm(obs)
        norm2 =  self.STAND.norm(obs)
        return np.concatenate((norm1, norm2), axis=-1)

    def denorm(self, data):
        size = data.shape[-1]
        size = int(size/2)
        data = data[..., :size]
        return self.NORMALI.denorm(data)

