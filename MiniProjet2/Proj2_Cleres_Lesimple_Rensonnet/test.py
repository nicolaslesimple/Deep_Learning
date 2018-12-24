# -*- coding: utf-8 -*-

from torch import FloatTensor, manual_seed
import nn_group14 as nn14
import toy_model
import time

# Make runs reproducible
myseed = 141414
manual_seed(myseed)

# Select one_hot for MSE optimization for instance (this is checked below)
one_hot = True

# Generate train and test standardized data
Ntrain = 1000
Ntest = Ntrain
train_input, train_target, test_input, test_target = toy_model.generate_data_standardized(Ntrain, one_hot)

# Time model creation, training and testing
Nruns = 5
time_elapsed = FloatTensor(Nruns).zero_()
train_err = FloatTensor(Nruns).zero_()
test_err = FloatTensor(Nruns).zero_()

# Meta parameters (work in most combinations of loss funs and activations)
batch_size = 100
Nepochs = 150
eta = 0.01
momentum = 0.85
log_loss = True  # print to stdout during training

for irep in range(Nruns):
    t_start = time.time()

    # Create model (nn14.create_miniproject2_model could also be used)
    non_lin_activ = nn14.ReLU  # ReLU, Tanh, Sigmoid
    nonlin = non_lin_activ.nonlin_str()

    fc1 = nn14.Linear(2, 25, nonlinearity=nonlin)
    fc2 = nn14.Linear(25, 25, nonlinearity=nonlin)
    fc3 = nn14.Linear(25, 25, nonlinearity=nonlin)
    fc_out = nn14.Linear(25, 2)
    model = nn14.Sequential([fc1, non_lin_activ(),
                             fc2, non_lin_activ(),
                             fc3, non_lin_activ(),
                             fc_out])

    # Loss function
    lossfunction = nn14.MSELoss()  # MSELoss(), NLLLoss(), CrossEntropyLoss()
    if isinstance(lossfunction, nn14.MSELoss):
        assert one_hot is True, "Use one_hot targets with MSELoss"
    else:
        assert one_hot is False, "Do NOT use one_hot targets with NLLLoss and CrossEntropyLoss"

    # Optimizer
    optimizer = nn14.SGD(model.param(), lr=eta, momentum=momentum)

    # Train model
    nn14.train_model(model, train_input, train_target, lossfunction, optimizer, Nepochs, batch_size, log_loss, one_hot)

    # Final train and test errors
    nb_train_err = nn14.compute_nb_errors(model, train_input, train_target, one_hot)
    nb_test_err = nn14.compute_nb_errors(model, test_input, test_target, one_hot)
    print("Run {:d}/{:d}: train error rate {}%, test error rate {}%\n".format(irep+1, Nruns, 100*nb_train_err/Ntrain, 100*nb_test_err/Ntest))

    time_elapsed[irep] = time.time() - t_start
    train_err[irep] = 100*nb_train_err/Ntrain
    test_err[irep] = 100*nb_test_err/Ntest

print("Mean +- standard deviation over {:d} runs:".format(Nruns))
print("Time to initialize, train and test: {:4.3g}+-{:4.3g} [s]."
      "".format(time_elapsed.mean(), time_elapsed.std()))
print("Train error rate: {:4.3g}+-{:3.2g}%.".format(train_err.mean(), train_err.std()))
print("Test error rate: {:4.3g}+-{:3.2g}%.".format(test_err.mean(), test_err.std()))

