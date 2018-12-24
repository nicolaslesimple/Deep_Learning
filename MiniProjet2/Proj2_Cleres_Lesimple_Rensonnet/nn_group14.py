# -*- coding: utf-8 -*-

import math
from torch import FloatTensor, LongTensor, arange


# %% Parameter class
class Parameter(object):
    """Object representing a model parameter along with its gradient.

    Args:
        pytorch_tensor (Tensor): Float or LongTensor that the parameter should
            be initialized to.

    Attributes:
        data (Tensor): tensor containing the parameter value.
        grad (Tensor): tensor of the same shape as data containing the current
            gradient.
    """

    def __init__(self, pytorch_tensor):
        self.data = pytorch_tensor.clone()  # tensors passed by reference
        self.grad = pytorch_tensor.clone().fill_(0)  # maintains Float or Long

    def zero_grad(self):
        """Set gradient to zero.
        """
        self.grad.fill_(0)


# %% Module class and child classes

class Module (object):
    """ Base class for Modules.
    Modules are intended as the elementary building blocks making up complex
    neural networks.
    """
    def forward(self, * input):
        raise NotImplementedError

    def backward(self, * gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass


class Linear(Module):
    """Fully-connected linear layer.

    Args:
        input_units (int): number of input units to the layer.
        output_units (int): number of output units to the layer.
        bias (bool): indicate whether a bias should be added to each output.
            Default: True.
        nonlinearity (str): non-linear activation layer which the output is fed
            to. This will affect weight initialization. Default: sigmoid (for
            a initialization gain of 1).

    Attributes:
        input_units (int): number of input units to the layer.
        output_units (int): number of output units to the layer.
    """

    def __init__(self, input_units, output_units, bias=True, nonlinearity=None):
        super(Linear, self).__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.weights = Parameter(FloatTensor(output_units, input_units))
        self.input = None  # last input used for forward pass

        if bias:
            self.bias = Parameter(FloatTensor(output_units))
        else:
            self.bias = None

        if nonlinearity is None:
            nonlinearity = 'sigmoid'

        self._initialize_parameters(nonlinearity)

    def _initialize_parameters(self, nonlinearity):
        """Initialize weights and biases of fully-connected layer so that the
            variance of the activations is controlled.
        Args:
            nonlinearity (str): 'sigmoid', 'tanh', 'relu' or None. Scales the
                variance of the parameters to account for the effect of the
                non-linear activation. This makes most sense if the same non-
                linear activation is used throughout the network.

        """
        # Variance correction associated with nonlinear activation of network
        nonlinearity = nonlinearity.lower()
        if nonlinearity == 'sigmoid':
            self.init_gain = 1.0
        elif nonlinearity == 'tanh':
            self.init_gain = 5.0 / 3.0
        elif nonlinearity == 'relu':
            self.init_gain = math.sqrt(2.0)
        else:
            raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

        # Control variance of activations (not of gradients)
        uniform_corr = math.sqrt(3)  # account for uniform distribution
        stdv = uniform_corr * self.init_gain / math.sqrt(self.input_units)
        # TMP this is to match PyTorch
        stdv = 1/ math.sqrt(self.input_units)
        self.weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def param(self):
        if self.bias is not None:
            return [self.weights, self.bias]
        else:
            return [self.weights]

    def zero_grad(self):
        for p in self.param():
            p.zero_grad()

    def forward(self, x):
        """
        Applies forward pass of fully connected layer to input x

        Args:
            x (FloatTensor): if 2D must have size Nb x input_units, where Nb
                is the batch size. If x is 1D, it is assumed that Nb=1.

        Returns:
            FloatTensor: Nb x output_units, always 2D.

        """
        # store current input for backward pass
        self.input = x.clone().view(-1, self.input_units)

        if self.bias is not None:
            # automatic broadcasting for bias:
            return self.input.mm(self.weights.data.t()) + self.bias.data.view(-1, self.output_units)
        else:
            return self.input.mm(self.weights.data.t())

    def backward(self, dl_dout):
        """
        Applies backward pass of fully connected layer starting from gradient
        with respect to current output.

        Args:
            dl_dout (FloatTensor): if 2D, must have size Nb x output_units,
                where Nb is the batch size. If 1D, it is assumed that Nb=1.
                Contains the derivative of the batch loss with respect to
                each output unit, for each batch sample, of the current
                backward pass.

        Returns:
            FloatTensor: Nb x input_units tensor containing the derivative of
                the batch loss with respect to each input unit, for each batch
                sample, of the current backward pass.
        """
        ndim = len(list(dl_dout.size()))
        assert ndim > 0, "dl_dout argument cannot be empty"
        Nb = 1  # case where dl_dout is 1D, only one sample
        if ndim > 1:
            Nb = dl_dout.size(0)

        # Gradient increment for weights (broadcasting for batch-processing)
        # (sum contributions of all samples in the batch)
        grad_inc = (dl_dout.view(Nb, self.output_units, 1) *
                    self.input.view(Nb, 1, self.input_units)).sum(0)
        self.weights.grad.add_(grad_inc)

        # Gradient increment for bias
        # (sum of contributions of all samples in the batch)
        if self.bias is not None:
            self.bias.grad.add_(dl_dout.view(Nb, self.output_units).sum(0))

        # Return dl_din
        return dl_dout.view(Nb, self.output_units).mm(self.weights.data)


class ReLU(Module):
    """Rectified linear unit activation layer.
    """
    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None  # last input used for forward pass

    def forward(self, x):
        # store current input for backward pass
        self.input = x.clone()

        out = x.clone()
        out[x < 0] = 0
        return out

    def backward(self, dl_dout):
        return dl_dout * self._grad(self.input)

    def _grad(self, x):
        dout_dx = x.clone().fill_(1)
        dout_dx[x < 0] = 0
        return dout_dx

    @staticmethod
    def nonlin_str():
        return "relu"


class Tanh(Module):
    """Hyperbolic tangent activation layer.
    """

    def __init__(self):
        super(Tanh, self).__init__()
        self.input = None  # last input used for forward pass

    def forward(self, x):
        # store current input for backward pass
        self.input = x.clone()
        return x.tanh()

    def backward(self, dl_dout):
        return dl_dout * self._grad(self.input)

    def _grad(self, x):
        return 1-x.tanh().pow(2)

    @staticmethod
    def nonlin_str():
        return "tanh"


class Sigmoid(Module):
    """Sigmoid activation layer.
    """
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.input = None

    def forward(self, x):
        # store current input for backward pass
        self.input = x.clone()
        return x.sigmoid()

    def backward(self, dl_dout):
        return dl_dout * self._grad(self.input)

    def _grad(self, x):
        return (x.exp() + x.mul(-1).exp() + 2).pow(-1)

    @staticmethod
    def nonlin_str():
        return "sigmoid"


class Sequential(Module):

    def __init__(self, moduleList):
        super(Sequential, self).__init__()
        self.moduleList = moduleList
        self.input = None

    def forward(self, x):
        self.input = x.clone()
        output = x.clone()
        for m in self.moduleList:
            output = m.forward(output)
        return output

    def backward(self, dl_dout):
        dl_din = dl_dout.clone()
        for m in self.moduleList[::-1]:
            dl_din = m.backward(dl_din)
        return dl_din

    def param(self):
        par_list = []
        for m in self.moduleList:
            par_list.extend(m.param())
        return par_list

    def zero_grad(self):
        for m in self.moduleList:
            m.zero_grad()


class LogSoftMax(Module):
    """ Layer that applies logarithm and softmax component-wise.
    """
    def __init__(self):
        super(LogSoftMax, self).__init__()
        self.input = None

    def forward(self, x):
        self.input = x.clone()
        # shift by max for numerical stability
        x_norm = x - x.max(dim=1, keepdim=True)[0]
        e_x = x_norm.exp()
        return x_norm - e_x.sum(dim=1, keepdim=True).log()

    def backward(self, dl_dout):
        # shift by max for numerical stability
        x_norm = self.input - self.input.max(1, keepdim=True)[0]
        e_x = x_norm.exp()  # b_size x dim
        softmax_x = e_x / e_x.sum(dim=1, keepdim=True)  # div uses broadcasting to keep b_size x dim
        return (-softmax_x * dl_dout.sum(dim=1, keepdim=True)) + dl_dout  # mul uses braodcasting


# %% Loss functions
class Loss(object):
    """ Base class for Loss functions.
    """
    def loss(self, output, target):
        raise NotImplementedError

    def backward(self, output, target):
        """
        Returns derivative of loss(output, target) with respect to output.
        """
        raise NotImplementedError


class MSELoss(Loss):
    """ Mean squared error loss
    """
    def loss(self, output, target):
        """Computes the sum of the squared differences between each sample of
        the batch and sums over all batch samples.

        Args:
            output (FloatTensor): model output, of size Nb x d1 x d2 x ...
                where Nb is the batch size.
            target (Tensor): must have the same size as output. Converted to
                FloatTensor automatically.

        Returns:
            float: a scalar floating-point value.
        """
        return (output-target.float()).pow(2).sum()/output.size(0)  # sum over all samples and entries

    def backward(self, output, target):
        return 2 * (output-target.float()) / output.size(0)


class NLLLoss(Loss):
    """ Negative log likelihood loss.
    """

    def loss(self, output, target):
        """
        """
        # Get dimension
        ndim = len(list(output.size()))
        assert ndim == 2, "output argument must have size Nb x d"
        Nb = output.size(0)
        out_dim = output.size(1)
        # sum the "on-target" activations across the batch:
        return - output.view(-1)[arange(0, Nb).long()*out_dim + target.long()].sum()/Nb

    def backward(self, output, target):
        dl_din = FloatTensor(output.size()).fill_(0)
        # Get dimension
        ndim = len(list(output.size()))
        assert ndim == 2, "output argument must have size Nb x d"

        Nb = output.size(0)
        for i in range(Nb):
            dl_din[i, target[i]] = -1/Nb
        return dl_din


class CrossEntropyLoss(Loss):
    """ Cross entropy loss.

        Equivalent to using NLLLoss and adding a final LogSoftMax layer to
        the network.

        Attributes:
            nll (NLLLoss): NLL loss is used internally, coupled with self.lsm
            lsm (LogSoftMax): internal LogSoftMax Module to simulate adding an
                extra LogSoftMax layer to the network being trained.
    """
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.nll = NLLLoss()
        self.lsm = LogSoftMax()

    def loss(self, output, target):
        return self.nll.loss(self.lsm.forward(output), target)

    def backward(self, output, target):
        # input -> LSM -> s -> NLL -> output
        dl_ds = self.nll.backward(output, target)
        return self.lsm.backward(dl_ds)


# %% Optimizer class
class Optimizer(object):
    """ Base class for optimizers.
    """
    def __init__(self, params):
        self.params = params

    def step(self, * input):
        raise NotImplementedError


class SGD(Optimizer):
    """ Stochastic gradient descend with fixed learning rate and momentum.

    Args:
        params (iterable of type Parameter): iterable yielding the parameters
            (Parameter objects) of the model to optimize, typically a list.
        lr (float): strictly positive learning rate or gradient step length
        momentum (float): non-negative weight of the momentum or inertia term
            (Rumelhart et al., Nature 1986). If set to 0, vanilla SGD is
            performed. Default: 0.

    Attributes:
        params (iterable of type Parameter): iterable yielding the parameters
            of the model to optimize, typically a list of Parameter objects.
        lr (float): learning rate or gradient step length.
        momentum (float): momentum or inertia term.
        step_buf (dict): contains the previous increment for each parameter if
            momentum is non-zero.

    """

    def __init__(self, params, lr, momentum=0):
        super(SGD, self).__init__(params)
        assert lr > 0, "learning rate should be strictly positive"
        self.lr = lr
        assert momentum >= 0, "momentum term should be non-negative"
        self.momentum = momentum
        if self.momentum > 0:
            self.step_buf = {}
            for p in self.params:
                self.step_buf[p] = FloatTensor(p.grad.size()).zero_()

    def step(self):
        """Updates the parameters of model using either vanilla SGD (if
        momentum is zero) or SGD with inertia. Parameter steps are used for the
        next iteration if an inertia term is present.
        """
        for p in self.params:
            param_step = self.lr * p.grad
            if self.momentum > 0:
                param_step.add_(self.momentum * self.step_buf[p])
                self.step_buf[p] = param_step.clone()
            p.data.add_(-param_step)


# %% Train and test functions

def compute_nb_errors(model, data_input, data_target, one_hot=False, batch_size=100):
    """Compute number of classification errors of a given model.

    Args:
        model (Module): Module trained for classification
        data_input (FloatTensor): must have 2D size Nb x Nin, where Nb
            is the batch size and Nin the number of input units of model.
        data_target (FloatTensor): must have 2D size Nb x Nout, where Nout
            is the number of output units of the model, which should match the
            number of classes. One-hot encoding must be used, i.e.
            data_target[i,j]=1 if data sample i belongs to class j, and
            data_target[i,j]=-1 otherwise.
        one_hot (bool): specify if one-hot encoding was used for the target.
            Default: False.
        batch_size (int): batch size which should be used for an efficient
            forward pass. Does not necessarily need to be a divider of the
            number of data samples, althgough this is often desirable for
            statistical reasons. Note that this parameter does not influence
            model training at all. Default: 100.
    """
    Ndata = data_input.size(0)
    if one_hot:
        data_label = data_target.max(dim=1)[1]
    nb_errors = 0
    for b_start in range(0, data_input.size(0), batch_size):
        # account for boundary effects:
        bsize_eff = batch_size - max(0, b_start + batch_size - Ndata)
        # batch output has size Nbatch x 2 if one_hot=True, Nbatch otherwise:
        batch_output = model.forward(data_input.narrow(0, b_start, bsize_eff))
        if one_hot:
            pred_label = batch_output.max(dim=1)[1]  # has size Nbatch
            nb_err_batch = 0
            for k in range(bsize_eff):
                if data_label[b_start+k] != pred_label[k]:
                    nb_err_batch = nb_err_batch + 1
        else:
            nb_err_batch = (batch_output.max(1)[1] !=
                            data_target.narrow(0, b_start, bsize_eff)).long().sum()
        # conversion to Long solves serious overflow problem; otherwise the above results are treated as 1-byte short ints
        nb_errors += nb_err_batch
    return nb_errors


def train_model(model, train_input, train_target, criterion, optimizer, n_epochs=50, batch_size=100, log_loss=False, one_hot=None):
    """Train model.

    Args:
        model (Module)
        train_input (FloatTensor)
        train_target (Tensor): converted to float if needed
        criterion (Loss): loss function
        optimizer (Optimizer): optimizer
        n_epochs (int)
        batch_size (int)
        log_loss (bool): set to True to print training progress a few times.
        one_hot (bool): if specified, used to print the train error.
            Default:None

    """
    Nprint_stdout = 5  # number of times loss is printed to standard output
    Ntrain = train_input.size(0)
    for i_ep in range(n_epochs):
        ep_loss = 0.0
        for b_start in range(0, Ntrain, batch_size):
            model.zero_grad()

            # account for boundary effects
            bsize_eff = batch_size - max(0, b_start + batch_size - Ntrain)

            # forward pass
            output = model.forward(train_input.narrow(0, b_start, bsize_eff))
            batch_loss = criterion.loss(output, train_target.narrow(0, b_start, bsize_eff))
            ep_loss += batch_loss

            # backward pass
            dl_dout = criterion.backward(output, train_target.narrow(0, b_start, bsize_eff))
            dl_dx = model.backward(dl_dout)

            # parameter update
            optimizer.step()

        # print progress
        err_str = ""  # training error rate to be displayed
        if one_hot is not None:
            ep_err = compute_nb_errors(model, train_input, train_target, one_hot)
            err_str = "(error rate {:3.2g} %)".format(100*ep_err/Ntrain)

        if log_loss and i_ep % round(n_epochs/Nprint_stdout) == 0:
            print("epoch {:d}/{:d}: training loss {:4.3g} {:s}"
                  "".format(i_ep+1, n_epochs, ep_loss, err_str))

# %% Create toy example MLP
def create_miniproject2_model(nonlin_activ=ReLU):
    """Create the neural network used in the validation of mini-project 2
    """
    nonlin = nonlin_activ.nonlin_str()

    fc1 = Linear(2, 25, nonlinearity=nonlin)
    fc2 = Linear(25, 25, nonlinearity=nonlin)
    fc3 = Linear(25, 25, nonlinearity=nonlin)
    fc_out = Linear(25, 2)
    model = Sequential([fc1, nonlin_activ(),
                        fc2, nonlin_activ(),
                        fc3, nonlin_activ(),
                        fc_out])
    return model
