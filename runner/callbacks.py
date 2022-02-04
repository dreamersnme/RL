import time

from stable_baselines3.common.callbacks import BaseCallback


class LearnEndCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super(LearnEndCallback, self).__init__(verbose)
        self.start_num = None
        self.last_aloss = None
        self.last_closs = None

    def _on_step(self): return True


    def _on_training_start(self) -> None:
        self.start_num = self.model._n_updates
        self.start_tm = time.time()
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_training_end(self) -> None:
        log = self.logger.name_to_value.copy()
        self.last_aloss = log["train/actor_loss"]
        self.last_closs = log["train/critic_loss"]
        number_delta = (self.model._n_updates - self.start_num)*self.model.n_steps
        self.fps = int( number_delta/ (time.time()-self.start_tm))


