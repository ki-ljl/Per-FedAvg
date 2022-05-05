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

from data_process import nn_seq_wind


def train(args, model):
    """
    Client training
    :param args:hyperparameters
    :param model:server model
    :return:client model after training
    """
    model.train()
    Dtr, Dte = nn_seq_wind(model.name, args.B)
    model.len = len(Dtr)
    print('training...')
    data = [x for x in iter(Dtr)]
    for epoch in range(args.E):
        origin_model = copy.deepcopy(model)
        final_model = copy.deepcopy(model)
        # step1
        model = one_step(args, data, model, lr=args.alpha)
        # step2
        model = get_grad(args, data, model)
        # step3
        hessian_params = get_hessian(args, data, origin_model)
        # step 4
        cnt = 0
        for param, param_grad in zip(final_model.parameters(), model.parameters()):
            hess = hessian_params[cnt]
            cnt += 1
            I = torch.ones_like(param.data)
            grad = (I - args.alpha * hess) * param_grad.grad.data
            param.data = param.data - args.beta * grad

        model = copy.deepcopy(final_model)

    return model


def one_step(args, data, model, lr):
    """
    :param args:hyperparameters
    :param data: a batch of data
    :param model:original client model
    :param lr:learning rate
    :return:model after one step gradient descent
    """
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


def get_grad(args, data, model):
    """
    :param args:hyperparameters
    :param data:a batch of data
    :param model: model after one step gradient descent
    :return: gradient
    """
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq = seq.to(args.device)
    label = label.to(args.device)
    y_pred = model(seq)
    loss_function = nn.MSELoss().to(args.device)
    loss = loss_function(y_pred, label)
    loss.backward()

    return model


def get_hessian(args, data, model):
    """
    :param args:hyperparameters
    :param data: a batch of data
    :param model: original model
    :return: hessian matrix
    """
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq = seq.to(args.device)
    label = label.to(args.device)
    y_pred = model(seq)
    loss_function = nn.MSELoss().to(args.device)
    loss = loss_function(y_pred, label)
    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    hessian_params = []
    for k in range(len(grads)):
        hess_params = torch.zeros_like(grads[k])
        for i in range(grads[k].size(0)):
            # w or b?
            if len(grads[k].size()) == 2:
                for j in range(grads[k].size(1)):
                    hess_params[i, j] = torch.autograd.grad(grads[k][i][j], model.parameters(), retain_graph=True)[k][
                        i, j]
            else:
                hess_params[i] = torch.autograd.grad(grads[k][i], model.parameters(), retain_graph=True)[k][i]
        hessian_params.append(hess_params)

    return hessian_params


def local_adaptation(args, model):
    """
    Adaptive training
    :param args:hyperparameters
    :param model: federated global model
    :return:final model after adaptive training
    """
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

        print('local_adaptation loss', loss.item())

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
