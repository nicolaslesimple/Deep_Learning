# DeepLearning2018

This is a main folder of the deep learning mini-projects for the Deep Learning Class at EPFL of Spring 2018

### Mini-Project 1 - Electroencephalography (EEG) recordings
The objective of this project is to train a predictor of finger movements from Electroencephalography (EEG) recordings. 

### Mini-Project 2 - Operational Prototype of a Deep Learning in Python without using deep learning libraries as pyTorch, Tensor Flow, Keras, ...
Deep Learning class project, we demonstrate our ability to design an operational prototype of a deep learning in Python. We provide user-friendly routines for the easy instantiation, training and testing of deep architectures resembling multi-layer perceptrons (MLP). Drawing inspiration from the open-source PyTorch project, fully-connected layers and non-linear activation layers are easily assembled using the Module.Sequential container. The mean squared error (MSE), negative log-likelihood (NLL) and cross-entropy (CE) loss functions are implemented and stochastic gradient descent (SGD) is provided in its vanilla form as well as with a momen- tum term. Backward and forward methods handle batch data, leveraging the power of vectorized computations with PyTorch Tensors. Similarly to PyTorch, the bulk of the training is done by the Module.backward(), Loss.backward() and Optimizer.step() methods. We validate our framework on a toy-example of binary classification for points inside or outside a disk in a 2D plane. Although the results are very sensitive to meta- parameters, our results are globally shown to be comparable to those of PyTorch, both in accuracy and in efficiency.
