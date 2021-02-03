# Author: Yoshimasa Kubo
# Date: 2020/03/09
# Updated: 2020/05/07
# Dataset: MNIST 
# Purpose: CHL with preClamp*(postClamped – postFreePredicted)  

import numpy as np
from keras.datasets import mnist

import torch

import os
from sklearn.preprocessing import OneHotEncoder
from sklearn import utils  
from warnings import filterwarnings
filterwarnings('ignore')
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.cuda.FloatTensor)

## just in case, please create the directory before runnning
directory = 'with_delay13_inp12_clamped_f120_c120_ada'

# list for AdaGrad
dy_squared  = []
dy_squared.append(None)
dy_squared.append(None)

## define sigmoid 
def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

def relu(X):
   return torch.maximum(0,X)


## weight initialization (Xaiver initialization for weights)
## return appended w and b
##
def initialize_weights(node_sizes):
  w, b = [],[]
  for i in range(len(node_sizes)-1):
    w.append(torch.tensor(np.random.rand(node_sizes[i],node_sizes[i+1]) * np.sqrt(6. /(node_sizes[i]*node_sizes[i+1]))).float().cuda())
    b.append(torch.tensor(np.random.normal(size=node_sizes[i+1]) - 0.5).float().cuda())
  return w, b

## preprocess for dataset 
## return training set (x and y) and testing set (x and y)
##
def preprocess_data():
  train_size = 60000
  test_size  = 10000  
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
  x_train = np.reshape(x_train, (train_size, -1)) / 255.0
  x_test  = np.reshape(x_test,  (test_size, -1)) / 255.0

  # one hot encoding
  enc = OneHotEncoder()
  train_y = enc.fit_transform(y_train[:, np.newaxis]).toarray()
  test_y = enc.fit_transform(y_test[:, np.newaxis]).toarray()

  return torch.from_numpy(x_train), torch.from_numpy(x_test), torch.from_numpy(train_y), torch.from_numpy(test_y)

## shuffle the data
## return shuffled dataset (x and y)
##
def shuffle_data(X, y):
  return utils.shuffle(X, y, random_state=1234)

## calculating dynamics
## return states of neurons at each layer
def calculate_dynamics(input, time, delay, dt, batch_size, node_sizes, gamma, w, b, target=None, binarize=False):
    # initialization of activations
    activations     = [torch.zeros((batch_size,size)) for size in node_sizes]
    activations_new = [torch.zeros((batch_size,size)) for size in node_sizes]
     
    # clamped input
    activations[0] =  input
    length = len(node_sizes)

    store_all_activations     = [torch.zeros((size, batch_size, time)) for size in node_sizes]

    # similations for the free or clamped phase start
    for t in range(time):
      length = len(node_sizes)
      if (target is not None) and (t >= delay):
        activations[-1] = target
        length -= 1
      for j in range(1,length):
        if (target is None and j == length -1) or (t < delay and target is not None and j == length -1)  : # length + 1 does not exist
            activations_new[j] =  activations[j] + dt * (- activations[j] + sigmoid(torch.mm(activations[j-1].float().cuda(), w[j-1])  + b[j-1]))
        else: 
           activations_new[j] =  activations[j] + dt * (- activations[j] + sigmoid(torch.mm(activations[j-1].float().cuda(), w[j-1])  + gamma*torch.mm(activations[j+1].float().cuda(), torch.transpose(w[j], 0, 1))  + b[j-1]))

      # t -> t +1
      for k in range(1, length):
        activations[k] = activations_new[k]
        store_all_activations[k][:,:, t] = torch.transpose(activations[k], 0, 1)

    return activations, store_all_activations


## update weights and biases
## updated date: 2020/06/02
## update rule with AdaGrad (AdaCHL)
def update_weights(w, b, learning_rate, gamma, free_act, clamped_act, length=2, batch_size=32):
    for i in range(1, length):       
      
      # AdaptiveGrad with Contrastive Hebbian Learning (AdaCHL)
      # Reffered to AdaGrad      
      dy =  (torch.mm(torch.transpose(clamped_act[i-1].float().cuda(), 0, 1), clamped_act[i].float().cuda())  - torch.mm(torch.transpose(clamped_act[i-1].float().cuda(), 0, 1), free_act[i].float().cuda()))/float(batch_size)
      global dy_squared
      if dy_squared[i-1] is None:
        dy_squared[i-1] = dy * dy
      else:
        dy_squared[i-1] += dy * dy
    
      dy_update  = dy/(torch.sqrt(dy_squared[i-1]) + 1e-7)
      
      w[i-1] += learning_rate[i-1]*(dy_update)
      b[i-1] += learning_rate[i-1]*((clamped_act[i].float().cuda() -  free_act[i].float().cuda())[0])/float(batch_size) 
    return w, b


## linear prediction
## return prediction of dynamics
## updated: 2020/05/07
def predict_dynamics(store_free_all_activations, prediction_inp_size, node_size, update_data_idx, train_ls_idx, length=5):
  #length = np.shape(store_free_all_activations)[0]
  pred_all_activations = []
  
  # for each layer
  for i in range(1,length):
    layer_l     = store_free_all_activations[i]
    node_size_l = np.shape(layer_l)[0] 
    activaitons = []
    
    # for each neuron
    for j in range(node_size_l):
      ## training data for the prediction
      one_neuron_train_data     =  layer_l[j, train_ls_idx, :prediction_inp_size].cpu().numpy()
      
      ## testing data (prediction data)
      one_neuron_test_data      = layer_l[j, update_data_idx, :prediction_inp_size].cpu().numpy()
  
      shape_train                      = np.shape(one_neuron_train_data)
      shape_test                       = np.shape(one_neuron_test_data)
      
      ## adding offset for trainig and prediction data
      one_neuron_input_offset_train    = np.ones((shape_train[0], shape_train[1]+1))
      one_neuron_input_offset_test     = np.ones((shape_test[0], shape_test[1]+1))     
      one_neuron_input_offset_train[:, :-1] = one_neuron_train_data
      one_neuron_input_offset_test[:, :-1]  = one_neuron_test_data
      
      ## targets for traininig 
      one_neuron_train_target               = layer_l[j, train_ls_idx, -1].cpu().numpy()
      
      ## training for linear regresssion
      pred_activation                       = np.linalg.lstsq(one_neuron_input_offset_train, one_neuron_train_target, rcond=None)[0]
      
      # prediction
      #pred_negative_acts              = one_neuron_input_offset_test @ pred_activation # or
      pred_negative_acts              = np.dot(one_neuron_input_offset_test,  pred_activation)
      
      # if values are negative, they will be 0
      pred_negative_acts              = np.clip(pred_negative_acts, a_min=0, a_max=None)    

      activaitons.append(torch.from_numpy(pred_negative_acts))
    pred_all_activations.append(torch.transpose(torch.stack(activaitons), 0,1))
  
  return pred_all_activations

## check accuracy
## return mean of accuracy
## 
def check_accuracy(data_x, data_y, data_size, batch_size, node_sizes, free_time, clamped_time, delay, dt, gamma, w, b, n_activations):
  accs = []
  test_size = 10000 
  index = int(test_size/float(batch_size))
  for i in range(index): 
    x = torch.reshape(data_x[i*batch_size:(i+1)*batch_size], (batch_size, node_sizes[0]))
    y = torch.reshape(data_y[i*batch_size:(i+1)*batch_size], (batch_size, node_sizes[-1]))
      
    free_act, _          = calculate_dynamics(x, free_time, delay, dt, batch_size, node_sizes, gamma, w, b)


    acc =  torch.argmax(free_act[-1].float().cuda(), dim=1) == torch.argmax(y.float().cuda(), dim=1)  
    accs.append(acc)
  return torch.mean(torch.stack(accs).float().cuda())

  

## train the model  
## return training and testing accuraices
##
def train_model(epoch, w, b, learning_rate, gamma, batch_size, minibatch_size, free_time, clamped_time, delay, dt, node_sizes, n_activations, prediction_inp_size, pred=False):
  train_accs   = []
  test_accs    = []
  train_x, test_x, train_y, test_y = preprocess_data()
  train_size = np.shape(train_x)[0]
  test_size  = np.shape(test_x)[0]
  epoch_train_size = int(train_size/batch_size)
  
  ## check accuracies before training   
  ## check training accuracy (mean)
  train_acc  = check_accuracy(train_x, train_y, train_size, batch_size, node_sizes, free_time, clamped_time, delay, dt, gamma, w, b, n_activations)      

  ## check testing accuracy (mean)
  test_acc   = check_accuracy(test_x, test_y, test_size, batch_size, node_sizes, free_time, clamped_time, delay, dt, gamma, w, b, n_activations)

  print('epoch:' + str(0))
  print('accuracy for training: ' + str(train_acc.cpu().numpy()))
  print('accuracy for testing: '  + str(test_acc.cpu().numpy()))
      
  f = None
  if os.path.isfile(directory + '/log.txt'):
    f = open(directory + '/log.txt', 'a')
  else:
    os.mkdir(directory)    
    f = open(directory + '/log.txt', 'w')

  np.save(directory + '/w_epoch_' + str(0) + '.npy', w)    
  np.save(directory + '/b_epoch_' + str(0) + '.npy', b)        
  np.save(directory + '/dy_squared_epoch_' + str(0) + '.npy', dy_squared)
        
  f.write("Epoch: " + str(0) + '\n')
  f.write("accuracy for training: " + str(train_acc.cpu().numpy()) + '\n')
  f.write("accuracy for testing: " + str(test_acc.cpu().numpy()) + '\n')
  f.close()  
  for i in range(epoch):

    start = time.time()
    
    train_x, train_y = shuffle_data(train_x, train_y)

    for j in range(epoch_train_size):
      one_x = torch.reshape(train_x[j*batch_size:(j+1)*batch_size], (batch_size, node_sizes[0]))
      one_y = torch.reshape(train_y[j*batch_size:(j+1)*batch_size], (batch_size, node_sizes[-1]))
      
      ## the free phase
      free_act,    store_free_all_activations    = calculate_dynamics(one_x, free_time, delay, dt, batch_size, node_sizes, gamma, w, b)
      
      ## randomly picked up data indices for the prediction and clamped phase data to update the weights
      update_data_idx                            = np.random.choice(batch_size, size=minibatch_size, replace=False)
      
      ## these indices are for training LS model to predict the activations  
      train_ls_idx                               = [k for k in range(batch_size) if k not in update_data_idx]

      if pred:
          ## predict the dynamics for both hidden and output
          free_pred_acts = predict_dynamics(store_free_all_activations, prediction_inp_size, node_sizes, update_data_idx, train_ls_idx, length=len(node_sizes))
          
          ## store predicted dynamics into the free phase activations
          input_act = free_act[0]
          del free_act
          
          free_act = []
          free_act.append(input_act[update_data_idx, :])  
          free_act.append(free_pred_acts[0])
          free_act.append(free_pred_acts[1])

      ## the clamped phase 
      clamped_act, store_clamped_all_activations = calculate_dynamics(one_x[update_data_idx,:], clamped_time, delay, dt, minibatch_size, node_sizes, gamma, w, b, target=one_y[update_data_idx,:])
            
      ## update the weights
      w, b = update_weights(w, b, learning_rate, gamma, free_act, clamped_act, length=len(node_sizes), batch_size=minibatch_size)

    ## after every xxx epoch, check the accuracies for training and testing  
    if i % 1 == 0:
      ## check training accuracy (mean)
      train_acc  = check_accuracy(train_x, train_y, train_size, batch_size, node_sizes, free_time, clamped_time, delay, dt, gamma, w, b, n_activations)      

      ## check testing accuracy (mean)
      test_acc   = check_accuracy(test_x, test_y, test_size, batch_size, node_sizes, free_time, clamped_time, delay, dt, gamma, w, b, n_activations)
    
      print('epoch:' + str(i+1))
      print('accuracy for training: ' + str(train_acc.cpu().numpy()))
      print('accuracy for testing: '  + str(test_acc.cpu().numpy()))

      np.save(directory + '/w_epoch_' + str(i+1) + '.npy', w)    
      np.save(directory + '/b_epoch_' + str(i+1) + '.npy', b)         
      
      f = None
      if os.path.isfile(directory + '/log.txt'):
        f = open(directory + '/log.txt', 'a')
      else:
        f = open(directory + '/log.txt', 'w')
        
      f.write("Epoch: " + str(i+1) + '\n')
      f.write("accuracy for training: " + str(train_acc.cpu().numpy()) + '\n')
      f.write("accuracy for testing: " + str(test_acc.cpu().numpy()) + '\n')
      f.close()        

      train_accs.append(train_acc)
      test_accs.append(test_acc)
      end = time.time()
      print(end - start)

  return train_accs, test_accs

## running the code
## return training and testing accuraices
##
def run():

  ## setting up the hyper parameters 
  gamma          = 1.0                    # gamma for CHL 
  free_time      = 120                    # total simulation time for CHL
  clamped_time   = 120
  dt             = 0.1                    # time step for CHL
  epoch          = 3                      # training epoch
  learning_rate  = [0.03, 0.02]           # learning rate
  n_activations  = 100                    # how many examples should be used for activation maps
  pred           = True

  batch_size          = 500               # mini-batch size 
  minibatch_size      = 10                # update mini-batch size
  hidden_size         = [1000]            # hidden size for the network
  output_size         = 10                # output size for the network
  input_size          = 784               # input size for the network 28x28 = 784 mnist
  prediction_inp_size = 12                # how many inputs should be used for predictions of dynamics on the free phase
  delay               = 13                # time delay for the clamped phase 

  print('Delay: ' + str(delay))
  print('Pred inp size: ' + str(prediction_inp_size))  

  # all of the sizes for the network
  # at first, adding the first layer
  node_sizes = [input_size] 

  # adding the hidden layers 
  for i in hidden_size:
    node_sizes.append(i)

  # adding output layers
  node_sizes.append(output_size)

  # initilization of weights and biases for the model
  w, b = initialize_weights(node_sizes)

  return train_model(epoch, w, b, learning_rate, gamma,  batch_size, minibatch_size, free_time, clamped_time, delay, dt, node_sizes, n_activations, prediction_inp_size, pred=pred)

# run the code
train_accs, test_accs = run()
