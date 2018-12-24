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
# %% Time various repetitions for different batch sizes

# Make runs reproducible
myseed = 141414
manual_seed(myseed)

# Select one_hot for MSE optimization for instance
one_hot = True

# Generate train and test standardized data
Ntrain = 1000
Ntest = Ntrain
train_input, train_target, test_input, test_target = toy_model.generate_data_standardized(Ntrain, one_hot)

# Time model creation, training and testing
Nreps = 15
batch_sizes =  [200, 300] # [1, 2, 3, 4, 5, 10, 15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1000]
time_elapsed_gr14 = FloatTensor(len(batch_sizes), Nreps).zero_()
train_err_gr14 = FloatTensor(len(batch_sizes), Nreps).zero_()
test_err_gr14 = FloatTensor(len(batch_sizes), Nreps).zero_()

time_elapsed_pt = FloatTensor(len(batch_sizes), Nreps).zero_()
train_err_pt = FloatTensor(len(batch_sizes), Nreps).zero_()
test_err_pt = FloatTensor(len(batch_sizes), Nreps).zero_()

# Meta parameters
Nepochs = 30
eta = 1e-2
momentum = 0
log_loss = False


for i_b in range(len(batch_sizes)):
    b_size = batch_sizes[i_b]
    for irep in range(Nreps):
        ##########################
        # GROUP 14 IMPLEMENTATION
        ##########################
        t_start = time.time()

        # Create model
        non_lin_activ = nn14.ReLU  # ReLU, Tanh, Sigmoid
        model = nn14.create_miniproject2_model(non_lin_activ)

        # Loss function
        loss_function_g14 = nn14.MSELoss()  # MSELoss(), NLLLoss(), CrossEntropyLoss()
        if isinstance(loss_function_g14, nn14.MSELoss):
            assert one_hot is True, "Use one_hot targets with MSELoss"

        # Optimizer
        optimizer_g14 = nn14.SGD(model.param(), lr=eta, momentum=momentum)

        # Train model
        nn14.train_model(model, train_input, train_target, loss_function_g14, optimizer_g14, Nepochs, b_size, log_loss)

        # Evaluate
        nb_train_err = nn14.compute_nb_errors(model, train_input, train_target, one_hot)
        nb_test_err = nn14.compute_nb_errors(model, test_input, test_target, one_hot)
        print("Gr14 NN (rep {:d}/{:d}, bsize {:d}) Train error rate {}%, test error rate {}%".format(irep+1, Nreps, b_size, 100*nb_train_err/Ntrain, 100*nb_test_err/Ntest))

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

        # Loss function
        loss_function = torch.nn.MSELoss()  #MSELoss, NLLLoss, CrossEntropyLoss

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
        print("PyTorch NN (rep {:d}/{:d}, bsize {:d}): train error {:g}%, test error {:g}%".format(irep+1, Nreps, b_size, n_err_train*100/Ntrain, n_err_test*100/Ntest))

        time_elapsed_pt[i_b, irep] = time.time() - t_start
        train_err_pt[i_b, irep] = n_err_train*100/Ntrain
        test_err_pt[i_b, irep] = n_err_test*100/Ntest

# %% Plots
import matplotlib.pyplot as plt
fig_time = plt.figure()
plt.semilogy(batch_sizes, time_elapsed_gr14.mean(dim=1).numpy(), '-ob', label="Group 14")
plt.semilogy(batch_sizes, time_elapsed_pt.mean(dim=1).numpy(), '-xr', label="PyTorch")
plt.xlabel("batch size", fontweight='bold')
plt.ylabel("Time [s]", fontweight='bold')
plt.legend()
#fig_time.savefig("logtime_vs_bsize", format='eps')

speedup_gr14 = time_elapsed_gr14.mean(dim=1)[0]/time_elapsed_gr14.mean(dim=1)
speedup_pt = time_elapsed_pt.mean(dim=1)[0]/time_elapsed_pt.mean(dim=1)

fig_speedup = plt.figure()
plt.plot(batch_sizes, speedup_gr14.numpy(), '-ob', label="Group 14")
plt.plot(batch_sizes, speedup_pt.numpy(), '-xr', label="PyTorch")
plt.xlabel("batch size", fontweight='bold')
plt.ylabel("Speed-up", fontweight='bold')
plt.legend()
#fig_speedup.savefig("speedup_vs_bsize", format='eps')