# -*- coding:utf-8 -*-
"""
@Time: 2022/03/08 12:50
@Author: KI
@File: server.py
@Motto: Hungry And Humble
"""

import torch
import numpy as np
import random
from client import train, test, local_adaptation
from model import ANN
import copy


# Implementation for per-fedavg server
class PerFed:
    def __init__(self, args):
        self.args = args
        self.nn = ANN(args=self.args, name='server').to(args.device)
        self.nns = []
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in range(self.args.r):
            print('round', t + 1, ':')
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)  # st
            # dispatch parameters
            self.dispatch(index)
            # local updating
            self.client_update(index)
            # aggregation parameters
            self.aggregation(index)

        return self.nn

    def aggregation(self, index):
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len

        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data / len(index)
                # params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index):  # update nn
        for k in index:
            self.nns[k] = train(self.args, self.nns[k])

    def global_test(self):
        for j in range(self.args.K):
            model = copy.deepcopy(self.nn)
            model.name = self.args.clients[j]
            model = local_adaptation(self.args, model)
            test(self.args, model)
