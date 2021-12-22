import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import numpy as np
import os
import json
import math

class SGDRScheduler(Callback):
    def __init__(self,
                 path_to_checkpoints = 'Checkpoints',
                 min_lr = 7e-5,
                 start_cyclical = 100,
                 base_lr = 7e-3,
                 cyclical_base_lr = 7e-4,
                 cyclical_epoch=10,
                 warmup_epoch=10,
                 last_epoch=0):

        self.path_to_checkpoints = path_to_checkpoints
        self.min_lr = min_lr
        self.base_lr = base_lr
        self.start_cyclical = start_cyclical
        self.cyclical_epoch = cyclical_epoch
        self.cyclical_base_lr = cyclical_base_lr
        self.warmup_epoch = warmup_epoch
        self.last_epoch = last_epoch
        self.next_checkpoint_epoch = 20

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        if self.last_epoch < self.warmup_epoch:
            print(self.last_epoch)
            return self.min_lr + self.last_epoch*(self.base_lr - self.min_lr)/self.warmup_epoch
        elif self.last_epoch < self.start_cyclical:
            return self.min_lr + (self.base_lr-self.min_lr)*(1+math.cos(math.pi*(self.last_epoch-self.warmup_epoch)/(self.start_cyclical-self.warmup_epoch))) / 2
        else:
            return self.min_lr + (self.cyclical_base_lr-self.min_lr)*(1+math.cos(math.pi* ((self.last_epoch-self.start_cyclical)% self.cyclical_epoch)/self.cyclical_epoch)) / 2

    def on_train_begin(self, opt, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(opt.learning_rate, self.min_lr)

    def on_epoch_end(self,epoch, model, losses, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if self.last_epoch == self.next_checkpoint_epoch:
            path_h5 = os.path.join(self.path_to_checkpoints, 'epoch_{}.h5'.format(epoch))
            model.save(path_h5)
            path_json = os.path.join(self.path_to_checkpoints, 'epoch_{}.json'.format(epoch))
            logs['lr'] = model.optimizer.lr
            logs['epoch'] = epoch
            logs['path_to_h5'] = path_h5
            logs['loss'] = np.array(losses)
            with open(path_json, 'w') as f:
                json.dump(logs, f)
            K.set_value(model.optimizer.lr, self.clr())
            if self.next_checkpoint_epoch <=80:
                self.next_checkpoint_epoch += 20
            else:
                self.next_checkpoint_epoch += 10
        K.set_value(model.optimizer.learning_rate, self.clr())
        self.last_epoch = self.last_epoch + 1
    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.save('final_model.h5')
