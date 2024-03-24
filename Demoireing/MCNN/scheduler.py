import math

import torch


class scheduler:
    '''学习率衰减'''

    def __init__(self, sch='CosineAnnealingWarmRestarts', epochs=100):
        super(scheduler, self).__init__()
        self.sch = sch

        if sch == 'StepLR':
            self.step_size = epochs // 5  # – Period of learning rate decay.
            self.gamma = 0.5  # – Multiplicative factor of learning rate decay. Default: 0.1
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif sch == 'MultiStepLR':
            self.milestones = [60, 120, 150]  # – List of epoch indices. Must be increasing.
            self.gamma = 0.1  # – Multiplicative factor of learning rate decay. Default: 0.1.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif sch == 'ExponentialLR':
            self.gamma = 0.99  # – Multiplicative factor of learning rate decay.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif sch == 'CosineAnnealingLR':
            self.T_max = 50  # – Maximum number of iterations. Cosine function period.
            self.eta_min = 0.00001  # – Minimum learning rate. Default: 0.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif sch == 'ReduceLROnPlateau':
            self.mode = 'min'  # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
            self.factor = 0.1  # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
            self.patience = 10  # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
            self.threshold = 0.0001  # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
            self.threshold_mode = 'rel'  # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
            self.cooldown = 0  # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
            self.min_lr = 0  # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
            self.eps = 1e-08  # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
        elif sch == 'CosineAnnealingWarmRestarts':
            self.T_0 = 50  # – Number of iterations for the first restart.
            self.T_mult = 2  # – A factor increases T_{i} after a restart. Default: 1.
            self.eta_min = 1e-6  # – Minimum learning rate. Default: 0.
            self.last_epoch = -1  # – The index of last epoch. Default: -1.
        elif sch == 'WP_MultiStepLR':
            self.warm_up_epochs = 10
            self.gamma = 0.1
            self.milestones = [125, 225]
        elif sch == 'WP_CosineLR':
            self.warm_up_epochs = 20

    def get_scheduler(self, optimizer):
        assert self.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                            'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
        if self.sch == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.step_size,
                gamma=self.gamma,
                last_epoch=self.last_epoch
            )
        elif self.sch == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.milestones,
                gamma=self.gamma,
                last_epoch=self.last_epoch
            )
        elif self.sch == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=self.gamma,
                last_epoch=self.last_epoch
            )
        elif self.sch == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.T_max,
                eta_min=self.eta_min,
                last_epoch=self.last_epoch
            )
        elif self.sch == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.mode,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
                threshold_mode=self.threshold_mode,
                cooldown=self.cooldown,
                min_lr=self.min_lr,
                eps=self.eps
            )
        elif self.sch == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.T_0,
                T_mult=self.T_mult,
                eta_min=self.eta_min,
                last_epoch=self.last_epoch
            )
        elif self.sch == 'WP_MultiStepLR':
            lr_func = lambda epoch: epoch / self.warm_up_epochs if epoch <= self.warm_up_epochs else self.gamma ** len(
                [m for m in self.milestones if m <= epoch])
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
        elif self.sch == 'WP_CosineLR':
            lr_func = lambda epoch: epoch / self.warm_up_epochs if epoch <= self.warm_up_epochs else 0.5 * (
                    math.cos((epoch - self.warm_up_epochs) / (self.epochs - self.warm_up_epochs) * math.pi) + 1)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

        return scheduler
