# pytorch cnn for multiclass classification
import os
import torch
import numpy as np
from numpy import vstack, argmax
from pandas import read_csv
from sklearn.metrics import accuracy_score
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.nn import Conv2d, Conv3d, MaxPool3d
from torch.nn import Linear, Dropout, ReLU, Softmax, CrossEntropyLoss
from torch.nn import Module, Sequential
from torch.optim import SGD


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# model definition
class C3D(Module):
    # define model elements
    def __init__(self, n_channels):
        super(C3D, self).__init__()
        # Convolution layers
        self.conv1 = Sequential(Conv3d(n_channels, 32, (3,3,3)), ReLU())
        self.conv2 = Sequential(Conv3d(32, 32, (3,3,3)), ReLU())
        self.dropout1 = Dropout(0.2)
        self.conv3 = Sequential(Conv3d(32, 64, (3,3,3)), ReLU())
        self.conv4 = Sequential(Conv3d(64, 64, (3,3,3)), ReLU())
        self.dropout2 = Dropout(0.2)        
        # Fully connected layer
        self.fc1 = Sequential(Linear(64*24*24*2, 512), ReLU())
        # output layer
        self.fc2 = Linear(512, 3)
        self.soft = Softmax(dim=1)
 
    # forward propagate input
    def forward(self, X):
        # input to first section
        X = self.conv1(X)
        X = self.conv2(X)      
        X = self.dropout1(X)
        # second section
        X = self.conv3(X)
        X = self.conv4(X)      
        X = self.dropout2(X)
        # flatten
        X = X.view(X.size(0), -1)
        # third hidden layer
        X = self.fc1(X)
        # output layer
        X = self.fc2(X)
        X = self.soft(X)
        
        return X
 
# prepare the dataset
def prepare_data(train_data, test_data):

    # load dataset
    with np.load(train_data) as npzfile:
        X_train = npzfile["X"]
        y_train = npzfile["Y"]
    X_train = np.tile(X_train, (3))
    X_train = np.transpose(X_train, axes=[0,4,1,2,3])
    X_train = torch.from_numpy(X_train).to(device)
    #X_train = X_train.astype(np.float)
    y_train = np.argmax(y_train, axis=1)    
    y_train = torch.from_numpy(y_train).to(device)

    train = list(zip(X_train, y_train))
        
    with np.load(test_data) as npzfile:
        X_test = npzfile["X"]
        y_test = npzfile["Y"]
    X_test = np.tile(X_test, (3))
    X_test = np.transpose(X_test, axes=[0,4,1,2,3])
    X_test = torch.from_numpy(X_test).to(device)
    #X_test = X_test.astype(np.float)
    y_test = np.argmax(y_test, axis=1)    
    y_test = torch.from_numpy(y_test).to(device)

    test = list(zip(X_test, y_test))
    
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)    
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=128, shuffle=True)
    test_dl = DataLoader(test, batch_size=64, shuffle=False)
    
    return train_dl, test_dl
 
# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    running_loss = 0
    # enumerate epochs
    for epoch in range(10):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs.float())
            # calculate loss
            loss = criterion(yhat, targets)
            print(yhat.shape, targets.shape, loss)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            # update loss
            running_loss += loss.item()
        
# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs.float())
        # retrieve numpy array
        yhat = yhat.cpu().detach().numpy()
        actual = targets.cpu().numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)

        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        print(yhat.shape, actual.shape)        
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc
 
# prepare the data
umitron_path = "D:/Umitron/sw-ml-20230126-powlook/datasets"
train_data_path = "c3d_train.npz"
test_data_path  = "c3d_test.npz"
train_data = os.path.join(umitron_path, train_data_path)
test_data = os.path.join(umitron_path, test_data_path)

train_dl, test_dl = prepare_data(train_data, test_data)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
#model = C3D(1)  # n_channel
#model = C3D(1).to(device)  # n_channel
model = torchvision.models.video.r3d_18(pretrained=True).to(device)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=3).to(device)
# # train the model
train_model(train_dl, model)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)