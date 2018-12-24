import math
from torch import FloatTensor, manual_seed
import time
import toy_model

########################################################################################################
# Creation of the class Neural_Network :
########################################################################################################

class NeuralNetwork:
    ########################################################################################################
    # Initialisation
    ########################################################################################################
    def __init__(self, n_data=None, n_output=None, list_hidden_neurons=None):
        self.n_output = n_output  # number of classes for the output. In our case, a probability that correspond to 0 or 1
        self.n_data = n_data  # number of dimensions of the input. In our case we have 2 : x and y.
        self.list_hidden_neurons = list_hidden_neurons  # list that represent the number of hidden node in each layer.
        self.net = self.net_creation()  # Building of the network thanks to the function you can see below

    ########################################################################################################
    # Creation of all the weights ot the nodes that makes the network
    ########################################################################################################
    def net_creation(self):
        ########## We define a internal function for one layer because we will use it for each layer ##########
        def layer_creation(nb_node_input, nb_node_output):
            layer = []  # we create the list that will contain information for each node in the layer
            for i in range(nb_node_output):
                s = 0.3
                weights = FloatTensor(nb_node_input, 1).uniform_(-s,s)  # We create a vector corresponding to the weights for each node : if intput layer have 10 nodes, the vector will have a size of ten.
                bias = FloatTensor(nb_node_input, 1).uniform_(-s,s)  # We initialize the weights and bias thanks to the random function which give a number between 0 and 1
                layer.append({'weights': weights, 'bias': bias, 'output': None,'delta': None})  # We add the weigths informatons to each node
            return layer

        ########## Now we want to use the layer_creation function to create the network for all the layer : input layer, hidden and output layer ##########
        net = []  # This list will have each layer in memory with information of each node in it (weights,output,delta)
        if len(self.list_hidden_neurons) == 0:  # If the lenght is 0, it means that there are no hidden layers
            net.append(layer_creation(self.n_data, self.n_output))
        else:  # else there are one or more hidden layers
            net.append(layer_creation(self.n_data, self.list_hidden_neurons[0]))  # input connection with first hidden layer
            for i in range(1, len(self.list_hidden_neurons)):
                net.append(layer_creation(self.list_hidden_neurons[i - 1],self.list_hidden_neurons[i]))  # connection between hidden layers
            net.append(layer_creation(self.list_hidden_neurons[len(self.list_hidden_neurons) - 1],self.n_output))  # connection between the last hidden layer and the output layer
        return net  # we return the object network which is a list of layer. A layer is a list of tuple with one tuple for each node. A tuple is made by 3 part with weghts vector, output and delta vector.

    ########################################################################################################
    # Forward method : it allows us to create and update the node's output from the input and thus we save at each node the values we want
    ########################################################################################################
    def forward_method(self, data):
        ########## We define the activation function because we will use it a lot in for loop ##########
        def x_activation(weights, bias, inputs):
            value = 0.0
            for i in range(len(weights)):
                value = value + weights[i] * inputs[i] + bias[i]  # The function only compute the sum of the different component of the vector weights times the component of the input vector
            return value

        tmp_data = data
        for layer in self.net:
            output = []
            for node in layer:  # We iterate on each node on each layer of the network
                activation = x_activation(node['weights'], node['bias'],tmp_data)  # We compute activation and apply transfer to it
                node['output'] = self.ReLU(activation)  # We apply transfer function to it
                output.append(node['output'])  # We save our update on the ouput node and on the output list
            tmp_data = output
        return tmp_data

    ########################################################################################################
    # Backward Method to take in account the error
    ########################################################################################################
    def backward_method(self, target):
        n_layers = len(self.net)
        loss = 0  # useful to print the loss for each epoch
        for i in reversed(range(n_layers)):
            layer = self.net[i]
            errors = list()
            if i == n_layers - 1:
                # Last layer: errors = target output difference
                for j, node in enumerate(layer):
                    error = (target[j] - node['output'])
                    errors.append(error)
                    loss = error ** 2 + loss
                loss = loss / len(layer)
            else:
                # Previous layers: error = weights sum of frontward node deltas
                for j, node in enumerate(layer):
                    error = 0.0
                    for node in self.net[i + 1]:
                        error += node['weights'][j] * node['delta']
                    errors.append(error)
            # Update delta using our errors
            for j, node in enumerate(layer):
                node['delta'] = errors[j] * self.ReLU_derivative(node['output'])
        return loss

    ########################################################################################################
    # Reach the optimal weights thanks to the error thanks to an update of the weights
    ########################################################################################################
    def update_method_weights_and_bias(self, x, learning_rate=0.05):
        for i_layer, layer in enumerate(self.net):# Update weights forward layer by layer
            if i_layer == 0: # Choose previous layer output to update current layer weights
                inputs = x
            else:
                inputs = FloatTensor(len(self.net[i_layer - 1])).zero_()  # -1 take into account the input layer
                for i_node, node in enumerate(self.net[i_layer - 1]):
                    inputs[i_node] = node['output']
            for node in layer: # Update weights using delta rule for single layer neural network
                for j, input in enumerate(inputs):
                    dW = learning_rate * node['delta'] * input
                    node['weights'][j] += dW
                    dB = learning_rate * node['delta']
                    node['bias'][j] += dB

    ########################################################################################################
    # Sigmoid Transfer Function
    ########################################################################################################
    def sigmoid(self, input_values):
        return 1.0 / (1.0 + math.exp(-input_values))

    ########################################################################################################
    # Sigmoid Derivative Transfer Function
    ########################################################################################################
    def sigmoid_derivative(self, input_values):
        return (1.0 / (1.0 + math.exp(-input_values))) * (1 - (1.0 / (1.0 + math.exp(-input_values))))

    ########################################################################################################
    # Linear Transfer Function
    ########################################################################################################
    def linear(self, input_values):
        return float(input_values)

    ########################################################################################################
    # Linear Derivative Transfer Function
    ########################################################################################################
    def linear_derivative(self, input_values):
        input_values = 1.0
        return float(input_values)

    ########################################################################################################
    # Hyperbolique Tangente Transfer Function
    ########################################################################################################
    def tanh(self, input_values):
        return math.tanh(input_values)

    ########################################################################################################
    # Hyperbolique Tangente Derivative Transfer Function
    ########################################################################################################
    def tanh_derivative(self, input_values):
        return 1 - (math.tanh(input_values) ** 2)

    ########################################################################################################
    # ReLU Transfer Function
    ########################################################################################################
    def ReLU(self, input_values):
        if float(input_values) <= 0:
            return float(0.0)
        else:
            return float(input_values)

    ########################################################################################################
    # ReLU Derivative Transfer Function
    ########################################################################################################
    def ReLU_derivative(self, input_values):
        if (float(input_values) > 0):
            return float(1.0)
        else:
            return float(0.0)

    ########################################################################################################
    # Train the model
    ########################################################################################################
    def _train(self, train_values, label, number_epochs=500, learning_rate=0.05, ):
        for epoch in range(number_epochs):
            loss = 0
            for (x, l) in zip(train_values, label):
                self.forward_method(x)  # Forward method to update the node
                target = FloatTensor(self.n_output).zero_()  # Create the label thanks to the real label we have as input which is y_train
                target[int(l)] = 1  # If label is 0, y_target is [1,0] and if label is 1, y_target is [0,1]
                loss = loss + self.backward_method(target)  # Backward method that will take into account the error
                self.update_method_weights_and_bias(x,learning_rate=learning_rate)  # Update the weights thanks to the update method
            if epoch % int(number_epochs/4) == 0:
                print('epoch :', epoch, '==> loss: ', loss / len(train_values))

    ########################################################################################################
    # Predcit the output with the forward method after the training
    ########################################################################################################
    def _predict_the_output(self, data):
        prediction = FloatTensor(len(data)).zero_()
        for j, x in enumerate(data):  # We enumerate on all inputs
            output = self.forward_method(x)  # Create the probability that makes the output
            if output[0] < output[1]:
                prediction[j] = 1  # We check which class is the highest one and thus we define our output
            else:
                prediction[j] = 0
        return prediction  # We return the output


# #######################################################################################
# Declaration of the parameters
# #######################################################################################

# Make runs reproducible
myseed = 141414
manual_seed(myseed)

data = 1000

list_hidden_neurons = [25, 25, 25]  # number of neurons in each hidden layers
l_r = 0.01  # learning rate
epochs = 5  # epochs

print('Model of our Neural Network:')
print('\n n_hidden_nodes = {}'.format(list_hidden_neurons))
print(' learning rate = {}'.format(l_r))
print(" number of epochs = {}".format(epochs))

# #######################################################################################
# Crate standardized data (X,y)
# #######################################################################################
print("\nCreating Data ...")
X, y, data_test, label_test = toy_model.generate_data_standardized(data)
dim, d = X.shape
n_classes = 2

print(" X.shape = {}".format(X.shape))
print(" y.shape = {}".format(y.shape))
print(" n_classes = {}".format(n_classes))

Nreps = 15
comp_times = FloatTensor(Nreps)
train_err = FloatTensor(Nreps)
test_err = FloatTensor(Nreps)

for i in range(Nreps):
    t = time.time()
    # #######################################################################################
    # Create model, train and predict :
    # #######################################################################################
    acc_train, acc_test = [],[]

    X_train, y_train = X, y
    X_test, y_test = data_test, label_test

    model = NeuralNetwork(n_data=d, n_output=n_classes, list_hidden_neurons=list_hidden_neurons) # Declaration of the model
    model._train(X_train, y_train, number_epochs=epochs, learning_rate=l_r, )

    y_train_predict = model._predict_the_output(X_train) # Predict

    # #######################################################################################
    # Compute accuracy for each training and testing set
    # #######################################################################################
    acc_train_counter = 0
    for j in range(len(y_train)):
        if (int(y_train[j]) == int(y_train_predict[j])):
            acc_train_counter = acc_train_counter + 1
    # Compute training accuracy score from predicted values
    acc_train.append(100 * acc_train_counter / len(y_train))
    # Print results
    print("Train accuracy = {:.2f}%, (n_train = {})".format(acc_train[-1],len(X_train)))


    # #######################################################################################
    # Test our network
    # #######################################################################################

    #  Make predictions for training and test data
    y_test_predict_real = model._predict_the_output(data_test)

    # Accuracy
    acc_test_real_counter = 0
    for j in range(len(label_test)):
        if (int(label_test[j]) == int(y_test_predict_real[j])):
            acc_test_real_counter = acc_test_real_counter + 1
    # Compute training/test accuracy score from predicted values
    acc_test_real = (100 * acc_test_real_counter / len(label_test))

    # timing
    t_el = time.time()-t
    print('The Test Accuracy is {:4.3g}%'.format(acc_test_real))
    comp_times[i] = t_el
    print('Elapsed time run {:d}/{:d}: {:4.3g}[s]'.format(i+1, Nreps, t_el))

    # Store
    train_err[i] = 100 - 100 * acc_train_counter / len(y_train)
    test_err[i] = 100 - 100 * acc_test_real_counter / len(label_test)

print("\nNdata={:d}, Nepochs={:d}, lr={}".format(data, epochs, l_r))
print("Train error rate {:}+-{}".format(train_err.mean(), train_err.std()))
print("Test error rate {:}+-{:}".format(test_err.mean(), test_err.std()))
print("Comp time {:4.3g}+-{:4.3g}".format(comp_times.mean(), comp_times.std()))
