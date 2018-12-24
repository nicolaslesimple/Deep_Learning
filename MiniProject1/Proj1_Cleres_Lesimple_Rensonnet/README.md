# DeepLearning2018

### Requierements 
#### Hardware
For Mac OSX - MacBook Pro (Retina, 15-inch, Mid 2014)
CPU: 2.5 GHz Intel Core i7 GPU: NVIDIA GeForce GT 750M 2 GB

#### Software
anaconda
conda install pytorch torchvision -c pytorch
pip install -U scikit-learn
scipy
matplotlib
numpy

# Python .py files
Please find in the folder the test.py file which only needs to be launched with the command python test.py. 

### test.py
This piece of code runs the model number 3 that we implemented. The code runs the trainning and testing for the best performing CNN that was implemented by our group (David Cleres, Nicolas Lesimple & Gaëtan Rensonnet). 
There are no paramters to give to the programm. During runtime the user is informed about the loss at each epoch and the accuracy of the model on the validation and the training datasets. 

### utility.py
The utility file cointained all the functions that were useful all along the project in order to perform the preprocessing (adding white noise, cutting the last part of the signal because it was very noisy, computing the FFT, ...). 
The model file cointained a summary of all the different models that we tried all along of the project. 

### dlc_bci.py
The dlc_bci file was given and left unchanged. 

# Python .ipynb files

### Basic Classifiers - Legacy.ipynb: 
contains the code for a basic implementation of the machine learning (not deep learning) classifiers that were implemented in the frame of this project. Furthermore, it contains important preprocessing steps and shows the first steps of the project (before implementig the deep learning part). Feature engineering was also in this document. 

### Visualize data.ipynb
enables to visualized the EEG signal per electrode for all the patients. It was useful to see all the signal in order to see if with the human eye it was possible to recognize some papers. 

### MLP.ipynb
This code implemented the fully connected neural network of the report. 

# Excel File 
The excel file attached to the submission contained SOME of the results that we got by running the neural networks and the claissfiers. A lot has been done on paper at the beginning or only to see if a certain architecture was working. Once the architectures were fixed we tried to work together on this sheet. 

