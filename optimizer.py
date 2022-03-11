# -*- coding:utf-8 -*-
"""
@Time: 2022/03/08 22:57
@Author: KI
@File: optimizer.py
@Motto: Hungry And Humble
"""
from torch.optim import Optimizer


class PerOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(PerOptimizer, self).__init__(params, defaults)

    def step(self, F, D_1, D_2, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        # for group in self.param_groups:
        #     for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
        #         if p.grad is None:
        #             continue
        #         dp = p.grad.data + c.data - ci.data
        #         p.data = p.data - dp.data * group['lr']

        return loss
