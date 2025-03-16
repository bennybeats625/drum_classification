import numpy as np
import torch as pt
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from stopwatch import tic, toc, time_string
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from drum_sample_processing import divvy

def model_eval(model, X):
    # disable gradient processing
    with pt.no_grad():
        # evaluate the model
        Z = model(X)
        # apply the softmax layer (which was omitted in the model definition)
        Y = nn.Softmax(dim=1)(Z)
        # convert the model output back to numpy format
        yy = Y.detach().cpu().numpy()
        # select the class with the highest probability
        y = np.argmax(yy, axis=1)
    # return the result
    return y

# define a model name
model_name = "drum_classifier"

# define if a pre-trained model should be loaded
load_the_model = False

# define if the model should be trained
train_the_model = True

# define if a trained model should be saved
save_the_model = True

# define the maximum iteration number
max_epoch = 20

# define the batch size
batch_size = 128

# initialize the learning rate
alpha = 0.01

# define the dropout probability for hidden layers
dropout = 0.5

# configure the optimization documentation
print_training_updates = True
show_accuracy_curves = True

# This swaps between processing the data and using processed data
skip_process = True

folder_path = r'.\sample_data\max_drm'

train_size = 3200
dev_size = 320
test_size = 480

print("Defined variables")

# Call the divvy function from "drum_sample_processing"
X_train, y_train, X_dev, y_dev, X_test, y_test = divvy(
    folder_path, train_size, dev_size, test_size, skip=skip_process
)

print('Divvied the data') 

labels = np.unique(y_train)

print('labels = ', labels)

# Reshape into the same shape in case it makes a difference
X_train = X_train.reshape(X_train.shape[0], 2, 100, 100)
X_dev = X_dev.reshape(X_dev.shape[0], 2, 100, 100)
X_test = X_test.reshape(X_test.shape[0], 2, 100, 100)

X_train = pt.tensor(X_train, requires_grad=False, dtype=pt.float32)
y_train = pt.tensor(y_train, requires_grad=False, dtype=pt.long)

# process the testing set
X_test = pt.tensor(X_test, requires_grad=False, dtype=pt.float32)
y_test = pt.tensor(y_test, requires_grad=False, dtype=pt.long)

# process the development set
X_dev = pt.tensor(X_dev, requires_grad=False, dtype=pt.float32)
y_dev = pt.tensor(y_dev, requires_grad=False, dtype=pt.long)

# Print out shapes
print("X_train shape2:", X_train.shape)
print("y_train shape2:", y_train.shape)
print("X_dev shape2:", X_dev.shape)
print("y_dev shape2:", y_dev.shape)
print("X_test shape2:", X_test.shape)
print("y_test shape2:", y_test.shape)

# combine input training data and labels into a joint data structure
joint_data_iterator = TensorDataset(X_train, y_train)

# define the data loader and batch manager
batched_data_iterator = DataLoader(dataset=joint_data_iterator, batch_size=batch_size, drop_last=False, shuffle=True)

# Print out shapes
print("X_train shape3:", X_train.shape)
print("y_train shape3:", y_train.shape)
print("X_dev shape3:", X_dev.shape)
print("y_dev shape3:", y_dev.shape)
print("X_test shape3:", X_test.shape)
print("y_test shape3:", y_test.shape)


if load_the_model:
    # load the mode from file
    # -----------------------
    print(' ')
    print('=================')
    print('Loading the Model')
    print('=================')
    print(' ')
    model_file_name = model_name + '.pt'
    print('Model Filename: "' + model_file_name + '"')
    # reload the model
    model = pt.load(model_file_name)

else:
    # define the model structure
    # --------------------------

    # define the input size and the output size
    output_size = len(labels)

    # define the activation function for the input layer and the hidden layers
    activation = nn.ReLU()
    
    # define the model as a sequential model
    model = nn.Sequential()

    # add the first convolutional layer to the model (with pooling)
    model.add_module('Conv Layer 1', nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=2))
    model.add_module('Activation 1', activation)
    model.add_module('Pooling Layer 1', nn.MaxPool2d(kernel_size=2))

    # add the second convolutional layer to the model (with pooling)
    model.add_module('Conv Layer 2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2))
    model.add_module('Activation 2', activation)
    model.add_module('Pooling Layer 2', nn.MaxPool2d(kernel_size=2))

    # add a fully connected hidden layer to the model
    model.add_module('Flattening Layer', nn.Flatten())
    model.add_module('Linear Layer 1', nn.Linear(43264, 1024))
    model.add_module('Activation 3', activation)

    # add the output layer (prior to the Softmax layer)
    model.add_module('Dropout Layer', nn.Dropout(p=dropout))
    model.add_module('Linear Layer 2', nn.Linear(1024, output_size))
    
# define the cost function
cost_function = nn.CrossEntropyLoss()

# define the optimizer
#optimizer = pt.optim.SGD(model.parameters(), lr=alpha)
optimizer = pt.optim.Adam(model.parameters(), lr=alpha)

# initialize the cost vector
cost = []

# initialize the accuracy collection vectors
# for the training and development sets
train_acc_vec = []
dev_acc_vec = []

# check if the model should be trained
if train_the_model:

    # start the timer
    tic()

    if print_training_updates:
        print(' ')
        print('==============')
        print('Model Training')
        print('==============')

        # display information about the data type
        print(' ')
        print('Data Type: Spectrogram')

        # display information about the model
        print(' ')
        print('PyTorch Model: "' + model_name + '"')
        print(model)

    # perform the gradient descend algorithm
    for epoch in range(max_epoch):

        # process the training data batch by batch
        for X, y in batched_data_iterator:
            #print('flag6 - ', epoch)
            
            # Print out shapes
            #print("X shape:", X.shape)
            #print("y shape:", y.shape)
            
            # ======================
            # Perform a Model Update
            # ======================
            # forward evaluate the model
            y_pred = model(X)
            #print('flag7 - ', epoch)
            # compute the cost value (loss)
            loss = cost_function(y_pred, y)
            #print('flag8 - ', epoch)
            # perform backpropagation for gradient calculation
            loss.backward()
            # perform the gradient descent update
            optimizer.step()
            # set the gradient values back to zero for the
            # next backpropagation step
            optimizer.zero_grad()

        # ==========================================
        # Check the Performance of the Updated Model
        # ==========================================

        # update the cost vector from the last loss calculation
        cost.append(loss.item())

        # calculate the accuracy on the development set
        y_pred = model_eval(model, X_dev)
        dev_accuracy = accuracy_score(y_dev, y_pred)
        dev_acc_vec.append(dev_accuracy)

        # calculate the accuracy on the training set
        y_pred = model_eval(model, X_train)
        train_accuracy = accuracy_score(y_train, y_pred)
        train_acc_vec.append(train_accuracy)

        # ==================
        # Report the Results
        # ==================

        # create a time measurement
        time_str = time_string(toc())

        # print the results
        if print_training_updates:
            print(' ')
            print(f'Epoch {epoch+1} (of {max_epoch}): [ Cumulative Runtime = ' + time_str +']')
            print(f'Cost:                 {cost[-1]:.5f} (down from {cost[0]:.5f} at epoch 1)')
            print(f'Training Accuracy:    {100 * train_accuracy:.2f} % (up from {100*train_acc_vec[0]:.2f} % at epoch 1)')
            print(f'Development Accuracy: {100 * dev_accuracy:.2f} % (up from {100*dev_acc_vec[0]:.2f} % at epoch 1)')

# =========================
# Save the Model if Desired
# =========================

if save_the_model:
    print(' ')
    print('================')
    print('Saving the Model')
    print('================')
    print(' ')
    model_file_name = model_name + '.pt'
    print('Model Filename: "' + model_file_name + '"')
    # save the model
    pt.save(model, model_file_name)

# ============================
# Perform the Final Evaluation
# ============================

print(' ')
print('======================')
print('Final Model Evaluation')
print('======================')

# display information about the data type
print(' ')
print('Data Type: Spectrogram')

# display information about the model
print(' ')
print('PyTorch Model: "' + model_name + '"')
print(model)

# calculate the accuracy on the testing set
tic(); y_pred = model_eval(model, X_test)
test_time = time_string(toc())
test_accuracy = accuracy_score(y_test, y_pred)

# calculate the accuracy on the development set
tic(); y_pred = model_eval(model, X_dev)
dev_time = time_string(toc())
dev_accuracy = accuracy_score(y_dev, y_pred)

# calculate the accuracy on the training set
tic(); y_pred = model_eval(model, X_train)
train_time = time_string(toc())
train_accuracy = accuracy_score(y_train, y_pred)

print(' ')
print('Performance Results: "' + model_name + '"')
print(f'Testing Accuracy:     {100*test_accuracy:.2f} % [ ' +\
      f'{len(y_test)} Instances in {test_time}]')
print(f'Development Accuracy: {100*dev_accuracy:.2f} % [ ' + \
      f'{len(y_dev)} Instances in {dev_time}]')
print(f'Training Accuracy:    {100*train_accuracy:.2f} % [ ' + \
      f'{len(y_train)} Instances in {train_time}]')

# ====================
# create the cost plot
# ====================
if show_accuracy_curves and train_the_model:
    # create the epoch number alignment
    epoch = np.arange(1, len(cost) + 1)
    # re-normalize the cost (relative cost in %)
    cost = 100 * np.array(cost) / np.max(cost)
    # assess the range of the re-normalized cost and accuracy values
    comax = np.max(cost); comin = np.min(cost)
    minvec = [comin, 100*np.min(dev_acc_vec), 100*np.min(train_acc_vec)]
    comin = np.min(minvec)
    codiff = 0.05 * (comax - comin)
    if codiff == 0: codiff = 1.0
    # define the range values for the cost axis
    comax = comax + codiff; comin = comin - codiff
    # create a figure and an axis for the pytorch cost plot
    fig = plt.figure(); ax = plt.subplot(111)
    # plot the cost curve (pytorch)
    ax.plot(epoch, cost, label='Relative Cost in % (learning rate = ' + str(alpha) + ')')
    ax.plot(epoch, 100 * np.array(dev_acc_vec), label='Development Set Accuracy')
    ax.plot(epoch, 100 * np.array(train_acc_vec), label='Training Set Accuracy')
    # define the limits on the shown data range
    ax.set_xlim(-5, epoch[-1] + 5); ax.set_ylim(comin, comax)
    # switch on grid lines
    ax.grid(True)
    # create a legend [upper/center/lower left/center/right]
    ax.legend(loc='center right', framealpha=0.5)
    # label the axis
    ax.set_title(f'Spectrogram Classifier - {model_name}')
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel('Percentage [%]')


# plot
plt.show()







