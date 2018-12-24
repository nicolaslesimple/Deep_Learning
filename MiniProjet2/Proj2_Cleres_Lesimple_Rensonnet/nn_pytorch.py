# -*- coding: utf-8 -*-

import torch
from torch import Tensor, manual_seed
from torch import nn  # not allowed in project 2; just for comparison
from torch.autograd import Variable

def compute_nb_errors(model, data_input, data_target, one_hot=False, batch_size=100):
    """Compute number of classification errors of a given model.
    """
    nb_errors = 0
    Ndata = data_input.size(0)
    for b_start in range(0, data_input.size(0), batch_size):
        bsize_eff = batch_size - max(0, b_start + batch_size - Ndata)   # accounts for boundary effects
        batch_output = model(data_input.narrow(0, b_start, bsize_eff))  # Nbatch x 2 if one_hot=True, Nbatch otherwise
        if one_hot:
            pred_label = batch_output.max(dim=1)[1]  # size Nbatch
            data_label = data_target.narrow(0, b_start, bsize_eff).max(dim=1)[1]  # could be done outside the batch loop; size is Nbatch
            nb_err_batch = 0
            for k in range(bsize_eff): # not very efficient but safest bet
                if data_label.data[k] != pred_label.data[k]: # data extracts torch.Tensor out of Variable
                    nb_err_batch = nb_err_batch + 1
        else:
            nb_err_batch = (batch_output.max(1)[1] != data_target.narrow(0, b_start, bsize_eff)).long().sum()
        # treated as short 1-byte int otherwise!!
        nb_errors += nb_err_batch
    if isinstance(nb_errors, torch.autograd.Variable):
        nb_errors = nb_errors.data[0]
    return nb_errors

def train_model(model, train_input, train_target, criterion, optimizer, n_epochs=50, batch_size=100, log_loss=False, one_hot=None):
    """Train model
    """
    Nprint_stdout = 5  # number of times loss is printed to standard output
    Ntrain = train_input.size(0)
    for i_ep in range(0, n_epochs):
        epoch_loss = 0
        for b_start in range(0, train_input.size(0), batch_size):
            model.zero_grad()
            # accounts for boundary effect
            bsize_eff = batch_size - max(0, b_start + batch_size - Ntrain)

            # forward pass
            batch_output = model(train_input.narrow(0, b_start, bsize_eff))
            batch_loss = criterion(batch_output, train_target.narrow(0, b_start, bsize_eff))  # instance of Variable
            epoch_loss = epoch_loss + batch_loss.data[0]

            # backward pass
            batch_loss.backward()

            # parameter update
            optimizer.step()

        # print progress
        err_str = ""  # training error rate to be displayed
        if one_hot is not None:
            ep_err = compute_nb_errors(model, train_input, train_target, one_hot)
            err_str = "(error rate {:3.2g} %)".format(100*ep_err/Ntrain)

        if log_loss and i_ep % round(n_epochs/Nprint_stdout) == 0:
            print("epoch {:d}/{:d}: training loss {:4.3g} {:s}"
                  "".format(i_ep+1, n_epochs, epoch_loss, err_str))


def create_miniproject2_model(nonlin_activ=nn.ReLU()):
    return nn.Sequential(nn.Linear(2, 25), nonlin_activ,
                         nn.Linear(25, 25), nonlin_activ,
                         nn.Linear(25, 25), nonlin_activ,
                         nn.Linear(25, 2))
