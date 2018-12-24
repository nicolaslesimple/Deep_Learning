import torch
from torch import Tensor
from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F

class conv2DNet_1(nn.Module):
    def __init__(self, output_units):
        """Initializes neural network with 3 convolutional layers and 1 fully-connected layer.
        
        Args:
            - Nchannels (int): number of EEG channels
            - Nsamples (int): number of time points in each EEG signal
            - output_units (int): number of output units, e.g. 1 for training with loss torch.nn.BCELoss or 2 with 
            loss torch.nn.CrossEntropyLoss            
            
            """
        super(conv2DNet_1, self).__init__() 
        self.fc_inputs = 64*32*12
        # Layer 1
        channelsConv1a = 8
        channelsConv1b = 16
        
        self.conv1a = nn.Conv2d(1, channelsConv1a, (5, 5), padding=5, dilation = 2)  # does not change size if combined with above padding
        self.batchnorm1a = nn.BatchNorm2d(channelsConv1a)
        self.conv1b = nn.Conv2d(channelsConv1a, channelsConv1b, (5, 5), padding=5, dilation = 2)  # does not change size if combined with above padding
        self.batchnorm1b = nn.BatchNorm2d(channelsConv1b)
        self.pooling1 = nn.MaxPool2d((2, 2), stride = 2) 

        # Layer 2
        channelsConv2a = 8
        channelsConv2b = 16
        
        self.conv2a = nn.Conv2d(1, channelsConv2a, (1, 3), padding=1)  # does not change size if combined with above padding
        self.batchnorm2a = nn.BatchNorm2d(channelsConv2a)
        self.conv2b = nn.Conv2d(channelsConv2a, channelsConv2b, (1, 3), padding=1)  # does not change size if combined with above padding
        self.batchnorm2b = nn.BatchNorm2d(channelsConv2b)
        self.pooling2 = nn.MaxPool2d((2, 2), stride = 2)  

        # Layer 3
        channelsConv3a = 32
        channelsConv3b = 64
        
        self.conv3a = nn.Conv2d(channelsConv2b, channelsConv3a, (3, 1), padding=1)  # does not change size if combined with above padding
        self.batchnorm3a = nn.BatchNorm2d(channelsConv3a)
        self.conv3b = nn.Conv2d(channelsConv3a, channelsConv3b, (3, 1), padding=1)  # does not change size if combined with above padding
        self.batchnorm3b = nn.BatchNorm2d(channelsConv3b)
        self.pooling3 = nn.MaxPool2d((2, 2), stride = 2) 
        
        # FC Layer
        self.fc1 = nn.Linear(self.fc_inputs, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_units)

        
    def forward(self, x):
        """Applies forward pass consisting of 3 convolutional layers followed by a fully-connected linear layer.
        
        Args:
            - x (torch.autograd.Variable): the input batch. It has dimension batch_size x Nchannel x Nsamples x 1,
            where Nchannel is the number of EEG channels and Nsamples the number of time points.
        
        Returns:
            - (torch.autograd.Variable) of size either batch_size x output_units   
        
        """
    
        # Layer 1
        x = F.relu(self.conv1a(x))              # bsize x l1_channels x 1 x Nsamples
        x = self.batchnorm1a(x)
        x = F.relu(self.conv1b(x))              # bsize x l1_channels x 1 x Nsamples
        x = self.batchnorm1b(x)
        x = self.pooling1(x)
        x = F.dropout2d(x, 0.5)

        # Layer 2
        x = self.conv2a(x)              # bsize x l1_channels x 1 x Nsamples
        x = self.batchnorm2a(x)
        x = self.conv2b(x)             # bsize x l1_channels x 1 x Nsamples
        x = self.batchnorm2b(x)
        #x = self.pooling2(x)
        x = F.dropout2d(x, 0.6)

        # Layer 3
        x = self.conv3a(x)              # bsize x l1_channels x 1 x Nsamples
        x = self.batchnorm3a(x)
        x = self.conv3b(x)             # bsize x l1_channels x 1 x Nsamples
        x = self.batchnorm3b(x)
        #x = self.pooling3(x)
        x = F.dropout2d(x, 0.6)
        
        #print(x.shape)
        
        # Fully-connected Layer
        x = x.view(-1, self.fc_inputs)  # bsize x (l3_channels*floor(l1_channels/4)*floor(Nsamples/16))
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        #x = F.dropout(x, 0.3)
        x = F.sigmoid(self.fc3(x))            # bisze x self.fc1.out_features  
        
        if self.fc3.out_features == 1:
            x = x.view(-1)                     # bsize (1D if 1 output unit)
        
        #print(x.shape)
        return x
    
class conv2DNet_2(nn.Module):
    def __init__(self, Nchannels, Nsamples, output_units):
        """Initializes neural network with 3 convolutional layers and 1 fully-connected layer.
        
        Args:
            - Nchannels (int): number of EEG channels
            - Nsamples (int): number of time points in each EEG signal
            - output_units (int): number of output units, e.g. 1 for training with loss torch.nn.BCELoss or 2 with 
            loss torch.nn.CrossEntropyLoss            
            
            """
        super(conv2DNet_2, self).__init__()
        # Layer 1
        l1_channels = 16  
        self.conv1 = nn.Conv2d(1, l1_channels, (Nchannels, 1), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(l1_channels, False) # final size bsize x 1 x l1_channels x Nsamples

        # Layer 2
        l2_channels = 4
        l2_temp_window = 32
        l2_l1channel_overlap = 2
        self.padding1 = nn.ZeroPad2d((l2_temp_window // 2, l2_temp_window // 2 - 1, l2_l1channel_overlap//2-1, l2_l1channel_overlap//2)) # left, right, top, bottom
        self.conv2 = nn.Conv2d(1, l2_channels, (l2_l1channel_overlap, l2_temp_window))  # does not change size if combined with above padding
        self.batchnorm2 = nn.BatchNorm2d(l2_channels, False)
        self.pooling2 = nn.MaxPool2d((2, 4)) # final size bsize x l2_channels x floor(l1_channels/2) x floor(Nsamples/4)

        # Layer 3
        l3_channels = 4
        l3_temp_window = 4
        l3_l2channel_overlap = 8
        self.padding2 = nn.ZeroPad2d((l3_temp_window//2, l3_temp_window//2-1, l3_l2channel_overlap//2, l3_l2channel_overlap//2-1))
        self.conv3 = nn.Conv2d(l2_channels, l3_channels, (l3_l2channel_overlap, l3_temp_window))
        self.batchnorm3 = nn.BatchNorm2d(l3_channels, False)
        self.pooling3 = nn.MaxPool2d((2, 4)) # final size bsize x l3_channels x floor(l1_channels/4) x floor(Nsamples/16)

        # FC Layer
        fc_inputs = l3_channels * (l1_channels//4) * (Nsamples//16)
        self.fc1 = nn.Linear(fc_inputs, output_units)
        
        
    def forward(self, x):
        """Applies forward pass consisting of 3 convolutional layers followed by a fully-connected linear layer.
        
        Args:
            - x (torch.autograd.Variable): the input batch. It has dimension batch_size x Nchannel x Nsamples x 1,
            where Nchannel is the number of EEG channels and Nsamples the number of time points.
        
        Returns:
            - (torch.autograd.Variable) of size either batch_size x output_units   
        
        """        
        # Layer 1
        x = F.elu(self.conv1(x))              # bsize x l1_channels x 1 x Nsamples
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 2, 1, 3)             # bsize x 1 x l1_channels x Nsamples

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))              # bsize x l2_channels x l1_channels x Nsamples
        x = self.batchnorm2(x)       
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)                  # bsize x l2_channels x floor(l1_channels/2) x floor(Nsamples/4)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))              # bsize x l3_channels x floor(l1_channels/2) x floor(Nsamples/4)
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)                  # bsize x l3_channels x floor(l1_channels/4) x floor(Nsamples/16)

        # Fully-connected Layer
        x = x.view(-1, self.fc1.in_features)  # bsize x (l3_channels*floor(l1_channels/4)*floor(Nsamples/16))
        x = F.sigmoid(self.fc1(x))            # bisze x self.fc1.out_features  
        
        if self.fc1.out_features == 1:
            x = x.view(-1)                     # bsize (1D if 1 output unit)
        
        return x
     
#Best Performing model         
class conv2DNet_3(nn.Module):
    def __init__(self, output_units):
        super(conv2DNet_3, self).__init__()
        
        #4*1*54 for 125 HZ 
        #4*1*42 for 100 Hz signal 
        self.fc_inputs = 4*1*42
        
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(1, 4, kernel_size=(1,5), dilation=2))
        self.conv.add_module("BN_1", torch.nn.BatchNorm2d(4, False))                     
        self.conv.add_module("conv_2", torch.nn.Conv2d(4, 4, kernel_size=(28,1)))
        self.conv.add_module("BN_2", torch.nn.BatchNorm2d(4, False))

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(self.fc_inputs, 4))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout(0.2))
        self.fc.add_module("fc2", torch.nn.Linear(4, output_units)) #add batch_norms
        self.fc.add_module("sig_1", torch.nn.Sigmoid()) 
        
    def forward(self,x):     
        x = self.conv.forward(x)
        x = x.view(-1, self.fc_inputs)
        return self.fc.forward(x)
    
#Medium Model 
class conv2DNet_4(nn.Module):
    def __init__(self, output_units):
        super(conv2DNet_4, self).__init__()
        
        #4*1*54 for 125 HZ 
        #4*1*42 for 100 Hz signal
        self.fc_inputs = 64*1*42        
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(1, 32, kernel_size=(1,5), dilation=2))
        self.conv.add_module("BN_1", torch.nn.BatchNorm2d(32, False))                     
        self.conv.add_module("conv_2", torch.nn.Conv2d(32, 64, kernel_size=(28,1)))
        self.conv.add_module("BN_2", torch.nn.BatchNorm2d(64, False))

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(self.fc_inputs, 64))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout(0.2))
        self.fc.add_module("fc2", torch.nn.Linear(64, output_units)) #add batch_norms
        self.fc.add_module("sig_1", torch.nn.Sigmoid()) 
        
    def forward(self,x):     
        x = self.conv.forward(x)
        x = x.view(-1, self.fc_inputs)
        return self.fc.forward(x)
    
#Heavy Model 
class conv2DNet_5(nn.Module):
    def __init__(self, output_units):
        super(conv2DNet_5, self).__init__()
        
        #4*1*54 for 125 HZ 
        #4*1*42 for 100 Hz signal
        self.fc_inputs = 128*1*42        
        self.conv = torch.nn.Sequential()
        self.conv.add_module("conv_1", torch.nn.Conv2d(1, 64, kernel_size=(1,5), dilation=2))
        self.conv.add_module("BN_1", torch.nn.BatchNorm2d(64, False))                     
        self.conv.add_module("conv_2", torch.nn.Conv2d(64, 128, kernel_size=(28,1)))
        self.conv.add_module("BN_2", torch.nn.BatchNorm2d(128, False))

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc1", torch.nn.Linear(self.fc_inputs, 128))
        self.fc.add_module("relu_3", torch.nn.ReLU())
        self.fc.add_module("dropout_3", torch.nn.Dropout(0.2))
        self.fc.add_module("fc2", torch.nn.Linear(128, output_units)) #add batch_norms
        self.fc.add_module("sig_1", torch.nn.Sigmoid()) 
        
    def forward(self,x):     
        x = self.conv.forward(x)
        x = x.view(-1, self.fc_inputs)
        return self.fc.forward(x)