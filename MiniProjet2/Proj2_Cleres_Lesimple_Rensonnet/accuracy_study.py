# -*- coding: utf-8 -*-

from torch import FloatTensor, LongTensor, manual_seed
import nn_group14 as nn14

import torch
from torch import Tensor
from torch import nn  # not allowed in project 2; just for comparison
from torch.autograd import Variable
import nn_pytorch as nnpt

import toy_model
import time
# %% Run multiple times to study effect of one given meta parameter

# Make runs reproducible
myseed = 141414
manual_seed(myseed)

# Select one_hot for MSE optimization for instance
one_hot = False

# Generate train and test standardized data
Ntrain = 1000
Ntest = Ntrain
train_input, train_target, test_input, test_target = toy_model.generate_data_standardized(Ntrain, one_hot)

# Time model creation, training and testing
Nreps = 15

# Meta parameters
b_size = 100
Nepochs = 100
eta = 0.005
momentum = 0
par_init = -1  # negative value to use default param initialization
log_loss = False

meta_par_vals = [0.2, 0.6, 1.0] # eta [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1] # Nepochs [25, 50, 75, 100, 125, 150, 200, 250, 300]

time_elapsed_gr14 = FloatTensor(len(meta_par_vals), Nreps).zero_()
train_err_gr14 = FloatTensor(len(meta_par_vals), Nreps).zero_()
test_err_gr14 = FloatTensor(len(meta_par_vals), Nreps).zero_()

time_elapsed_pt = FloatTensor(len(meta_par_vals), Nreps).zero_()
train_err_pt = FloatTensor(len(meta_par_vals), Nreps).zero_()
test_err_pt = FloatTensor(len(meta_par_vals), Nreps).zero_()

for i_b in range(len(meta_par_vals)):
    par_init = meta_par_vals[i_b]  # select meta parameter in left-hand side
    for irep in range(Nreps):
        ##########################
        # GROUP 14 IMPLEMENTATION
        ##########################
        t_start = time.time()

        # Create model
        non_lin_activ = nn14.ReLU  # ReLU, Tanh, Sigmoid
        model = nn14.create_miniproject2_model(non_lin_activ)

        # Initialization
        if par_init > 0:
            for p in model.param():
                p.data.uniform_(-par_init, par_init)

        # Loss function
        loss_function_g14 = nn14.CrossEntropyLoss()  # MSELoss(), NLLLoss(), CrossEntropyLoss()
        if isinstance(loss_function_g14, nn14.MSELoss):
            assert one_hot is True, "Use one_hot targets with MSELoss"

        # Optimizer
        optimizer_g14 = nn14.SGD(model.param(), lr=eta, momentum=momentum)

        # Train model
        nn14.train_model(model, train_input, train_target, loss_function_g14, optimizer_g14, Nepochs, b_size, log_loss)

        # Evaluate
        nb_train_err = nn14.compute_nb_errors(model, train_input, train_target, one_hot)
        nb_test_err = nn14.compute_nb_errors(model, test_input, test_target, one_hot)
        print("Gr14 NN (rep {:d}/{:d}, par val {}) Train error rate {}%, test error rate {}%".format(irep+1, Nreps, meta_par_vals[i_b], 100*nb_train_err/Ntrain, 100*nb_test_err/Ntest))

        # Store results
        time_elapsed_gr14[i_b, irep] = time.time() - t_start
        train_err_gr14[i_b, irep] = 100*nb_train_err/Ntrain
        test_err_gr14[i_b, irep] = 100*nb_test_err/Ntest

        ##########################
        # PYTORCH IMPLEMENTATION
        ##########################
        t_start = time.time()

        # Create model
        non_lin_activation = nn.ReLU()  # nn.Tanh(), nn.ReLU(), nn.Sigmoid(),...
        model = nnpt.create_miniproject2_model(non_lin_activation)

        # Initialization
        if par_init > 0:
            for p in model.parameters():
                p.data.uniform_(-par_init, par_init)

        # Loss function
        loss_function = torch.nn.CrossEntropyLoss()  #MSELoss, NLLLoss, CrossEntropyLoss

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum)

        # Train model
        if one_hot:
            nnpt.train_model(model, Variable(train_input), Variable(train_target.float()),
                                loss_function, optimizer, Nepochs, b_size, log_loss, one_hot)
        else:
            nnpt.train_model(model, Variable(train_input), Variable(train_target.long()),
                                loss_function, optimizer, Nepochs, b_size, log_loss, one_hot)

        # Evaluate and store results
        n_err_train = nnpt.compute_nb_errors(model, Variable(train_input), Variable(train_target), one_hot, b_size)
        n_err_test = nnpt.compute_nb_errors(model, Variable(test_input), Variable(test_target), one_hot, b_size)
        print("PyTorch NN (rep {:d}/{:d}, par val {}): train error {:g}%, test error {:g}%".format(irep+1, Nreps, meta_par_vals[i_b], n_err_train*100/Ntrain, n_err_test*100/Ntest))

        time_elapsed_pt[i_b, irep] = time.time() - t_start
        train_err_pt[i_b, irep] = n_err_train*100/Ntrain
        test_err_pt[i_b, irep] = n_err_test*100/Ntest

# %% Plots
import matplotlib.pyplot as plt
x_sh = 0.1*max(meta_par_vals)/len(meta_par_vals)  # x shift for visualization
xplt_1 = [p + x_sh for p in meta_par_vals]
xplt_2 = [p - x_sh for p in meta_par_vals]

fig = plt.figure()
ax = plt.gca()
# ax.set_xscale('log')  # for learning rate for instance
plt.errorbar(meta_par_vals, train_err_gr14.mean(dim=1).numpy(), train_err_gr14.std(dim=1), fmt='-sb', label="Group 14 - train")
plt.errorbar(xplt_1, test_err_gr14.mean(dim=1).numpy(), test_err_gr14.std(dim=1), fmt='-xb', label="Group 14 - test")
plt.errorbar(meta_par_vals, train_err_pt.mean(dim=1).numpy(), train_err_pt.std(dim=1), fmt='-sr', label="PyTorch - train")
plt.errorbar(xplt_2, test_err_pt.mean(dim=1).numpy(), test_err_pt.std(dim=1), fmt='-xr', label="PyTorch - test")
plt.ylim((0, 55))  # suitable for train/test error rates
plt.xlabel("initialization range a", fontweight='bold')
plt.ylabel("Error rate [%]", fontweight='bold')
plt.legend()

#fig.savefig("error_vs_init.eps", format='eps')

