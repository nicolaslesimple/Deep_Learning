import random
import math
from torch import FloatTensor
import time
from torch import LongTensor

########################################################################################################
# Find indexes for Cross validation
########################################################################################################
def cross_validation(dim, folds_number):
    index_permutation = []
    index_folds = []
    nb_fold = int(dim / folds_number)
    for i in range(dim):
        index_permutation.append(i)
    for i in range(folds_number):
        begin = i * nb_fold
        last = min([(i + 1) * nb_fold, dim])
        index_folds.append(index_permutation[begin:last])
    return index_folds


########################################################################################################
# Generation of data with standardization :
########################################################################################################
def generate_disc_set(nb):
    input = FloatTensor(nb, 2).uniform_(0, 1)# We create with uniform distribution nb points in the space [0,1]^2
    target = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long() # we create the label
    return input, target


def generate_data_standardized(nb):
    ########## Generation of the data ##########
    train_input, train_target = generate_disc_set(nb)
    test_input, test_target = generate_disc_set(nb)

    ########## Calculation of mean and std ##########
    mean, std = train_input.mean(), train_input.std()

    ########## Standardization ##########
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    return train_input, train_target, test_input, test_target

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
        def layer_creation(nb_node_input,nb_node_output):
            layer = []  # we create the list that will contain information for each node in the layer
            for i in range(nb_node_output):
                weights = FloatTensor(nb_node_input, 1).uniform_(0,1)  # We create a vector corresponding to the weights for each node : if intput layer have 10 nodes, the vector will have a size of ten.
                bias = FloatTensor(nb_node_input, 1).uniform_(0,1)  # We initialize the weights and bias thanks to the random function which give a number between 0 and 1
                layer.append({'weights': weights, 'bias': bias, 'output': None,'delta': None})  # We add the weigths informatons to each node
            return layer

        ########## Now we want to use the layer_creation function to create the network for all the layer : input layer, hidden and output layer ##########
        net = [] # This list will have each layer in memory with information of each node in it (weights,output,delta)
        if len(self.list_hidden_neurons) == 0:  # If the lenght is 0, it means that there are no hidden layers
            net.append(layer_creation(self.n_data, self.n_output))
        else:  # else there are one or more hidden layers
            net.append(layer_creation(self.n_data, self.list_hidden_neurons[0]))  # input connection with first hidden layer
            for i in range(1, len(self.list_hidden_neurons)):
                net.append(layer_creation(self.list_hidden_neurons[i - 1],self.list_hidden_neurons[i]))  # connection between hidden layers
            net.append(layer_creation(self.list_hidden_neurons[len(self.list_hidden_neurons) - 1], self.n_output))  # connection between the last hidden layer and the output layer

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
        # i = 0
        for layer in self.net:
            output = []
            for node in layer:  # We iterate on each node on each layer of the network
                activation = x_activation(node['weights'], node['bias'],tmp_data)  # We compute activation and apply transfer to it
                node['output'] = self.tanh(activation)  # We apply transfer function to it
                output.append(node['output'])  # We save our update on the ouput node and on the output list
                # print('weight :', node['weights'], 'bias :', node['bias'])
                # print('layer ', i, 'output ', node['output'], 'node number', j)
            tmp_data = output
            # i = i + 1

        return tmp_data

    # AATTNTTTENTION We need to change the error to MSE et tous les commentaire
    ########################################################################################################
    # Backward Method to take in account the error
    ########################################################################################################
    def backward_method(self, target):
        # Perform backward-pass through network to update node deltas
        n_layers = len(self.net)
        loss = 0  # usefull to print the loss for each epoch
        # print('AUTRE LAYER')
        for i in reversed(range(n_layers)):
            layer = self.net[i]

            # Compute errors either:
            # - explicit target output difference on last layer
            # - weights sum of deltas from frontward layers
            errors = list()
            if i == n_layers - 1:
                # Last layer: errors = target output difference
                for j, node in enumerate(layer):
                    error = math.pow((target[j] - node['output']),1)
                    errors.append(error)
                    loss = error + loss
                loss = loss / len(layer)
            else:
                # Previous layers: error = weights sum of frontward node deltas
                for j, node in enumerate(layer):
                    error = 0.0
                    for node in self.net[i + 1]:
                        error += node['weights'][j] * node['delta']

                    errors.append(error)

            # Update delta using our errors
            # The weight update will be:
            # dW = learning_rate * errors * transfer' * input
            #    = learning_rate * delta * input
            for j, node in enumerate(layer):
                node['delta'] = errors[j] * self.tanh_derivative(node['output'])
        return loss

    ########################################################################################################
    # Updating of the weights to reach the optimal weights thanks to the errror
    ########################################################################################################
    def update_method_weights_and_bias(self, x, learning_rate=0.05):

        # Update weights forward layer by layer
        for i_layer, layer in enumerate(self.net):

            # Choose previous layer output to update current layer weights
            if i_layer == 0:
                inputs = x
            else:
                inputs = FloatTensor(len(self.net[i_layer - 1])).zero_()  # -1 prend en compte l'input layer
                for i_node, node in enumerate(self.net[i_layer - 1]):
                    inputs[i_node] = node['output']

            # Update weights using delta rule for single layer neural network
            # The weight update will be:
            # dW = learning_rate * errors * transfer' * input
            #    = learning_rate * delta * input
            for node in layer:
                for j, input in enumerate(inputs):
                    dW = learning_rate * node['delta'] * input
                    node['weights'][j] += dW
                    dB = learning_rate * node['delta']
                    node['bias'][j] += dB
                    # print (node['weights'])

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
        input_values=1.0
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
        if (input_values) <=0:
            return float(0.0)
        else:
            return float(input_values)

    ########################################################################################################
    # ReLU Derivative Transfer Function
    ########################################################################################################
    def ReLU_derivative(self, input_values):
        if (float(input_values)>0):
            return float(1.0)
        else:
            return float(0.0)
       #input_values[input_values <= 0] = 0
       #input_values[input_values> 0] = 1
       #return input_values


    ########################################################################################################
    # Train the model
    ########################################################################################################
    def _train(self, train_values, label, number_epochs=500, learning_rate=0.05, ):
        for epoch in range(number_epochs):
            output = []
            loss = 0
            for (x, l) in zip(train_values, label):
                self.forward_method(x)  # Forward method to update the node
                target = FloatTensor(self.n_output).zero_()  # Create the label thanks to the real label we have as input which is y_train
                target[int(l)] = 1  # If label is 0, y_target is [1,0] and if label is 1, y_target is [0,1]
                loss = loss + self.backward_method(target)  # Backward method that will take into account the error
                #  print(self.backward_method(target))
                # print(loss)
                self.update_method_weights_and_bias(x,learning_rate=learning_rate)  # Update the weights thanks to the update method
            print('epoch :', epoch, '==> loss: ', loss / len(train_values))
            #    print('loss: ', loss / len(train_values))
            #  loss = []
            #   error=0
            #    layer = self.network[-1]
            #     print (output)
            #      for j in range (len(label)):
            #           error += (label[j]-output[j])**2
            #        loss.append(error/len(layer))
            #         print ('len',len(layer))
            #          print ("Epoch : {} --> Loss : {}".format(epoch,loss[-1] ))

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


def main(data,hidden,lr,epoch_):
    # #######################################################################################
    # Declaration of the parameters
    # #######################################################################################
    list_hidden_neurons = hidden  # number of neurons in each hidden layers
    #n_folds = 2  # number of folds we will use for our cross validation
    l_r = lr  # learning rate
    epochs = epoch_ # epochs

    print('Model of our Neural Network:')
    print('\n n_hidden_nodes = {}'.format(list_hidden_neurons))
   # print(" number of folds = {}".format(n_folds))
    print(' learning rate = {}'.format(l_r))
    print(" number of epochs = {}".format(epochs))

    # #######################################################################################
    # Read data (X,y) and normalize X
    # #######################################################################################
    print("\nCreating Data ...")
    X, y, data_test, label_test = generate_data_standardized(data)
    dim, d = X.shape  # extract shape of X
    n_classes = 2

    print(" X.shape = {}".format(X.shape))
    print(" y.shape = {}".format(y.shape))
    print(" n_classes = {}".format(n_classes))

    # #######################################################################################
    # Create cross-validation folds
    # These are a list of a list of indices for each fold
    # #######################################################################################
    # idx_all = np.arange(0, N)
    idx_all = []
    for i in range(dim):
        idx_all.append(i)
    #idx_folds = cross_validation(dim, n_folds)

    # #######################################################################################
    # Train and evaluate the model on each fold
    # #######################################################################################
    acc_train, acc_test = list(), list()  # training/test accuracy score

    start = time.time()
    print("        Start of fold {}...".format(1))

    X_train, y_train = X, y
    X_test, y_test = X, y
    # Build neural network classifier model and train
    model = NeuralNetwork(n_data=d, n_output=n_classes, list_hidden_neurons=list_hidden_neurons)
    model._train(X_train, y_train, number_epochs=epochs, learning_rate=l_r, )

    # Make predictions for training and test data
    y_train_predict = model._predict_the_output(X_train)
    y_test_predict = model._predict_the_output(X_test)

    # #######################################################################################
    # Compute accuracy for each training and testing set
    # #######################################################################################
    acc_train_counter = 0
    for j in range(len(y_train)):
        if (int(y_train[j]) == int(y_train_predict[j])):
            acc_train_counter = acc_train_counter + 1
    acc_test_counter = 0
    for j in range(len(y_test)):
        if (int(y_test[j]) == int(y_test_predict[j])):
            acc_test_counter = acc_test_counter + 1

    # Compute training/test accuracy score from predicted values
    acc_train.append(100 * acc_train_counter / len(y_train))
    acc_test.append(100 * acc_test_counter / len(y_test))
    # Print
    print("Fold {}/{}: train acc = {:.2f}%, test acc = {:.2f}% (n_train = {}, n_test = {})".format(1, 1,
                                                                                                   acc_train[-1],
                                                                                                   acc_test[-1],
                                                                                                   len(X_train),
                                                                                                   len(X_test)))

    end = time.time()
    print("        Execution time of fold {}: {} seconds. \n".format(i + 1, math.ceil(end - start)))

    # #######################################################################################
    # Print results
    # #######################################################################################
    print("\nAvg train acc = {:.2f}%".format(sum(acc_train) / float(len(acc_train))))
    print("Avg test acc = {:.2f}%".format(sum(acc_test) / float(len(acc_test))))

    # #######################################################################################
    # Test our network
    # #######################################################################################
    print('Test with new data : ')
    #  Make predictions for training and test data
    y_test_predict_real = model._predict_the_output(data_test)

    # Accuracy
    acc_test_real_counter = 0
    for j in range(len(label_test)):
        if (int(label_test[j]) == int(y_test_predict_real[j])):
            acc_test_real_counter = acc_test_real_counter + 1
    # Compute training/test accuracy score from predicted values
    acc_test_real = (100 * acc_test_real_counter / len(label_test))
    print('The Test Accuracy is : ', acc_test_real)

    print(label_test)
    print(y_test_predict_real)

# Driver
if __name__ == "__main__":
    main(100, [25,25,25], 0.01, 100)
    """
"A FAIRE :\n",
    "- mettre long tensor pour tous les int\n",
    "- remplacer toutes les liste par des tensor\n",
    "- Coder la loss : MSE \n",
    - seqential et linear module
    - faire la relu ...
    "- batch    \n",
    "\n",
    "- Normalisation L2\n",
    "- utiliser softmax derniÃ¨re layer\n",
    "- implementer sequential\n",
    "- checker si j ai bien fais la cross validation = checker si j ai bien pris la moyenne des tous les paramÃ¨tres\n",
    "- coder dropout et minibatch\n",
    "- optimizer en enlevant les boucle for et en faisant des matrice multiplication\n",
    "\n"
    """
