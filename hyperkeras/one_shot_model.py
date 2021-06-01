# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from .hyper_keras import HyperKeras
from hypernets.core.trial import Trail
from hypernets.core.callbacks import EarlyStoppingError

import time


class OneShotModel(HyperKeras):
    def __init__(self, searcher, optimizer, loss, metrics,
                 epochs=320,
                 batch_size=64,
                 controller_train_per_epoch=True,
                 controller_train_steps=50,
                 dispatcher=None,
                 callbacks=[],
                 reward_metric=None,
                 max_model_size=0,
                 one_shot_train_sampler=None,
                 visualization=False):
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_controller_per_epoch = controller_train_per_epoch
        self.controller_train_steps = controller_train_steps
        super(OneShotModel, self).__init__(searcher, optimizer, loss, metrics,
                                           dispatcher=dispatcher,
                                           callbacks=callbacks,
                                           reward_metric=reward_metric,
                                           max_model_size=max_model_size,
                                           one_shot_mode=True,
                                           one_shot_train_sampler=one_shot_train_sampler,
                                           visualization=visualization)

    def search(self, X, y, X_eval, y_eval, max_trials=None, dataset_id=None, trial_store=None, **fit_kwargs):
        self.start_search_time = time.time()
        try:
            trial_no = 1
            for epoch in range(self.epochs):
                print(f'One-shot model epoch({epoch}) training...')
                self.fit_one_shot_model_epoch(X, y, batch_size=self.batch_size, epoch=epoch)
                if self.train_controller_per_epoch:
                    trial_no = self.train_controller(X_eval, y_eval, self.controller_train_steps, max_trials, trial_no)

            if not self.train_controller_per_epoch:
                print(f'Architecture searching...')
                self.train_controller(X_eval, y_eval, max_trials, max_trials, trial_no)
        except EarlyStoppingError:
            print(f'Early stopping')
            # TODO: early stopping

    def train_controller(self, X_val, y_val, steps, max_trials, trial_from):
        trial_no = trial_from

        for con_step in range(steps):
            if max_trials is not None and trial_no >= max_trials:
                break
            start_time = time.time()
            space_sample = self.searcher.sample()
            estimator = self._get_estimator(space_sample)
            for callback in self.callbacks:
                callback.on_build_estimator(self, space_sample, estimator, trial_no)
                callback.on_trial_begin(self, space_sample, trial_no)

            metrics = estimator.evaluate(X_val, y_val, batch_size=self.batch_size, metrics=[self.reward_metric])
            reward = self._get_reward(metrics, self.reward_metric)
            elapsed = time.time() - start_time
            trial = Trail(space_sample, trial_no, reward, elapsed)
            improved = self.history.append(trial)

            if improved:
                self.best_model = estimator
            self.searcher.update_result(space_sample, reward)

            for callback in self.callbacks:
                callback.on_trial_end(self, space_sample, trial_no, reward, improved, elapsed)
                trial_no += 1
        return trial_no

    def derive_arch(self, X_test, y_test, num=10):
        for i in range(num):
            space_sample = self.searcher.sample()
            estimator = self._get_estimator(space_sample)
            metrics = estimator.evaluate(X_test, y_test, batch_size=self.batch_size, metrics=[self.reward_metric])
            reward = self._get_reward(metrics, self.reward_metric)
            print('>' * 50)
            estimator.summary()
            print(f'Reward on test set: {reward}')
            print('>' * 50)
