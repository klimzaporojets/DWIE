def create_lr_scheduler(optimizer, config, max_epochs, num_training_instances):
    if 'lr-scheduler' not in config:
        return NoneScheduler(optimizer)
    elif config['lr-scheduler']['type'] == 'default':
        return DefaultScheduler(optimizer, config['lr-scheduler'])
    elif config['lr-scheduler']['type'] == 'new':
        return LRScheduler(optimizer, config['lr-scheduler'], max_epochs, num_training_instances)
    elif config['lr-scheduler']['type'] == 'linear-decay':
        return LinearDecayScheduler(optimizer, config['lr-scheduler'], max_epochs, num_training_instances)
    elif config['lr-scheduler']['type'] == 'exponential-decay':
        return ExponentialDecayScheduler(optimizer, config['lr-scheduler'], max_epochs, num_training_instances)
    elif config['lr-scheduler']['type'] == 'reciprocal-decay':
        return ReciprocalDecayScheduler(optimizer, config['lr-scheduler'], max_epochs, num_training_instances)
    elif config['lr-scheduler']['type'] == 'steps':
        return StepScheduler(optimizer, config['lr-scheduler'], num_training_instances)
    else:
        raise BaseException('no such scheduler:', config['lr-scheduler']['type'])


## linear ramp up,  constant, exponential decay
class LRScheduler:

    def __init__(self, optimizer, config, num_epoch, steps_per_epoch=1):
        self.optimizer = optimizer
        self.lrate0 = config['lrate0']
        self.gamma = config['gamma']
        self.t0 = config['t0'] * steps_per_epoch
        self.t1 = config['t1'] * steps_per_epoch
        self.t2 = num_epoch * steps_per_epoch
        self.t = 1
        self.lrate = 0

    def step(self):
        self.t += 1
        if self.t <= self.t0:
            self.lrate = self.t / self.t0 * self.lrate0
        elif self.t <= self.t1:
            self.lrate = self.lrate0
        elif self.t <= self.t2:
            fraction = (self.t - self.t1) / (self.t2 - self.t1)
            self.lrate = self.lrate0 * (self.gamma ** fraction)

        for group in self.optimizer.param_groups:
            group['lr'] = self.lrate

        return self.lrate


class DefaultScheduler:

    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.lrate = config['lrate0']

    def step(self):
        for group in self.optimizer.param_groups:
            group['lr'] = self.lrate

        return self.lrate


class LinearDecayScheduler:

    def __init__(self, optimizer, config, num_epoch, steps_per_epoch=1):
        self.optimizer = optimizer
        self.lrate0 = config['lrate0']
        self.gamma = config['gamma']
        self.t0 = config['t0'] * steps_per_epoch
        self.t1 = config['t1'] * steps_per_epoch
        self.t = 1
        self.lrate = 0

    def step(self):
        self.t += 1
        if self.t <= self.t0:
            self.lrate = self.lrate0
        elif self.t <= self.t1:
            fraction = (self.t - self.t0) / (self.t1 - self.t0)
            self.lrate = self.lrate0 * (self.gamma * fraction + 1.0 * (1 - fraction))

        for group in self.optimizer.param_groups:
            group['lr'] = self.lrate

        return self.lrate


class ExponentialDecayScheduler:

    def __init__(self, optimizer, config, num_epoch, steps_per_epoch=1):
        self.optimizer = optimizer
        self.lrate0 = config['lrate0']
        self.gamma = config['gamma']
        self.t0 = config['t0'] * steps_per_epoch
        self.t1 = config['t1'] * steps_per_epoch
        self.t = 1
        self.lrate = 0

    def step(self):
        self.t += 1
        if self.t <= self.t0:
            self.lrate = self.lrate0
        elif self.t <= self.t1:
            fraction = (self.t - self.t0) / (self.t1 - self.t0)
            self.lrate = self.lrate0 * (self.gamma ** fraction)

        for group in self.optimizer.param_groups:
            group['lr'] = self.lrate

        return self.lrate


class ReciprocalDecayScheduler:

    def __init__(self, optimizer, config, num_epoch, steps_per_epoch=1):
        self.optimizer = optimizer
        self.lrate0 = config['lrate0']
        self.t0 = config['t0'] * steps_per_epoch
        self.t1 = config['t1'] * steps_per_epoch
        self.t = 1
        self.lrate = 0

        gamma = config['gamma']
        self.fnc = lambda x: gamma / ((1.0 - gamma) * x + gamma)

    def step(self):
        self.t += 1
        if self.t <= self.t0:
            self.lrate = self.lrate0
        elif self.t <= self.t1:
            fraction = (self.t - self.t0) / (self.t1 - self.t0)
            self.lrate = self.lrate0 * self.fnc(fraction)

        for group in self.optimizer.param_groups:
            group['lr'] = self.lrate

        return self.lrate


class NoneScheduler:

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        for group in self.optimizer.param_groups:
            return group['lr']


class StepScheduler:
    def __init__(self, optimizer, config, steps_per_epoch=1):
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.lrs = {k: v for k, v in zip(config['ts'], config['lr'])}
        self.t = 0

    def step(self):
        step = int(self.t / self.steps_per_epoch)
        self.t += 1

        if step in self.lrs:
            self.lrate = self.lrs[step]

        for group in self.optimizer.param_groups:
            group['lr'] = self.lrate
        return self.lrate
