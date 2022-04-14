# -*- coding:utf-8 -*-
"""
@Time: 2022/03/08 12:25
@Author: KI
@File: client.py
@Motto: Hungry And Humble
"""
from itertools import chain

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
import copy
from torch.autograd.functional import hessian

from data_process import nn_seq_wind


# def train(args, model):
#     model.train()
#     Dtr, Dte = nn_seq_wind(model.name, args.B)
#     model.len = len(Dtr)
#     print('training...')
#     data = [x for x in iter(Dtr)]
#     for epoch in range(args.E):
#         origin_model = copy.deepcopy(model)
#         final_model = copy.deepcopy(model)
#         # 1. step1
#         model = one_step(args, data, model)
#         # 2. step2
#         model = get_grad(args, data, model)
#         # model为F(w)
#         # 3. step3
#         origin_model = get_grad2(args, data, origin_model)
#         # 3.
#         for param, grad1, grad2 in zip(final_model.parameters(), origin_model.parameters(), model.parameters()):
#             I = torch.ones_like(param.data)
#             grad = (I - args.alpha * grad1.data) * grad2.data
#             param.data = param.data - args.beta * grad
#
#         model = copy.deepcopy(final_model)
#
#     return model


def train(args, model):
    model.train()
    original_model = copy.deepcopy(model)
    Dtr, Dte = nn_seq_wind(model.name, args.B)
    model.len = len(Dtr)
    print('training...')
    data = [x for x in iter(Dtr)]
    for epoch in range(args.E):
        # step1
        model = one_step_1(args, data, model, lr=args.alpha)
        # step2
        model = one_step_2(args, data, original_model, model, lr=args.beta)

    return model


def one_step_1(args, data, model, lr):
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq = seq.to(args.device)
    label = label.to(args.device)
    y_pred = model(seq)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.MSELoss().to(args.device)
    loss = loss_function(y_pred, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model


def one_step_2(args, data, original_model, model, lr):
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq = seq.to(args.device)
    label = label.to(args.device)
    # meta function parameter
    y_pred = model(seq)
    # update original model
    optimizer = torch.optim.Adam(original_model.parameters(), lr=lr)
    loss_function = nn.MSELoss().to(args.device)
    loss = loss_function(y_pred, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return original_model


def get_grad(args, data, model):
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq = seq.to(args.device)
    label = label.to(args.device)
    y_pred = model(seq)
    loss_function = nn.MSELoss().to(args.device)
    loss = loss_function(y_pred, label)
    # grad_x = torch.autograd.grad(loss, model.parameters())
    loss.backward()
    return model


def get_grad2(args, data, model):
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq = seq.to(args.device)
    label = label.to(args.device)
    y_pred = model(seq)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)
    loss_function = nn.MSELoss().to(args.device)
    loss = loss_function(y_pred, label)
    loss.backward(create_graph=True)
    optimizer.zero_grad()
    loss.backward()

    return model


def local_adaptation(args, model):
    model.train()
    Dtr, Dte = nn_seq_wind(model.name, 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)
    loss_function = nn.MSELoss().to(args.device)
    loss = 0
    # one step
    for epoch in range(args.local_epochs):
        for seq, label in Dtr:
            seq, label = seq.to(args.device), label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('local_adaptation loss', loss.item())

    return model


def test(args, ann):
    ann.eval()
    Dtr, Dte = nn_seq_wind(ann.name, args.B)
    pred = []
    y = []
    for (seq, target) in Dte:
        with torch.no_grad():
            seq = seq.to(args.device)
            y_pred = ann(seq)
            pred.extend(list(chain.from_iterable(y_pred.data.tolist())))
            y.extend(list(chain.from_iterable(target.data.tolist())))

    pred = np.array(pred)
    y = np.array(y)
    print('mae:', mean_absolute_error(y, pred), 'rmse:',
          np.sqrt(mean_squared_error(y, pred)))
