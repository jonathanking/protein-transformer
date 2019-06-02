'''A wrapper class for optimizer '''
import numpy as np


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps, simple=True):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)
        self.cur_lr = self.init_lr
        self.simple = simple

    def step(self):
        "Step with the inner optimizer"
        self._optimizer.step()
        if self.simple:
            self.n_current_steps += 1
            if self.n_current_steps % 20 == 0:
                for param_group in self._optimizer.param_groups:
                    param_group['lr'] /= 5
                    self.cur_lr = param_group['lr']
        else:
            self._update_learning_rate()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        self.cur_lr = lr
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return (self._optimizer.state_dict(), self.n_warmup_steps, self.n_current_steps, self.init_lr, self.cur_lr,
                self.simple)
