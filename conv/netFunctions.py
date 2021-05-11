from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import os, sys
import datetime
from shutil import copyfile

## linear prediction
## return prediction of dynamics
## updated: 2020/08/21
def predict_acts(store_inp_activations, neurons_phase2, update_data_idx, train_ls_idx):

  #pred_all_activations = []
  for idx in range(len(neurons_phase2)):
    # pop activations at a layer (phase1 and phase3)
    one_layer_acts     = store_inp_activations[idx]
    target_acts        = neurons_phase2[idx]

    # prepare for predicting the acitvation at phase3
    shape_         = one_layer_acts.size()

    if idx >= 1:
      one_layer_acts = torch.reshape(one_layer_acts, (shape_[0], shape_[1]*shape_[2]*shape_[3], shape_[4]))
      target_acts    = torch.reshape(target_acts, (shape_[0], shape_[1]*shape_[2]*shape_[3]))
      
    node_size      = one_layer_acts.size()[1]
    activations = []
    
    for node in range(node_size):
      ## training data for the prediction
      one_neuron_train_data               = one_layer_acts[train_ls_idx, node, :].detach().numpy()
      ## testing data (prediction data)
      one_neuron_test_data                = one_layer_acts[update_data_idx, node].detach().numpy()#.cpu().detach().numpy()
     
      
      shape_train                         = np.shape(one_neuron_train_data)
      shape_test                          = np.shape(one_neuron_test_data)      
   
      ## adding offset for trainig and prediction data
      one_neuron_input_offset_train       = np.ones((shape_train[0], shape_train[1]+1))
      one_neuron_input_offset_test        = np.ones((shape_test[0], shape_test[1]+1))     
      one_neuron_input_offset_train[:, :-1] = one_neuron_train_data
      one_neuron_input_offset_test[:, :-1]  = one_neuron_test_data    
      
      ## targets for traininig 
      one_neuron_train_target               = target_acts[train_ls_idx, node].cpu().detach().numpy()

      ## training for linear regresssion
      pred_activation                       = np.linalg.lstsq(one_neuron_input_offset_train, one_neuron_train_target, rcond=None)[0]
      
      # prediction
      pred_negative_acts                    = one_neuron_input_offset_test @ pred_activation

      # if values are negative, they will be 0
      pred_negative_acts              = np.clip(pred_negative_acts, a_min=0, a_max=1.0) 
      pred_activations = torch.from_numpy(pred_negative_acts)
      activations.append(pred_activations)

    activations  = torch.transpose(torch.stack(activations), 0,1)
    if idx >= 1:    
      activations   = torch.reshape(activations, (len(update_data_idx), shape_[1], shape_[2], shape_[3]))
      
    neurons_phase2[idx] = activations.float().cuda()
  return  neurons_phase2


def train(net, train_loader, epoch, batch_size, update_batch_size, pred_inps, delay, method): 
    if not hasattr(net, 'C_tab'):
        net.train()
        loss_tot = 0
        correct = 0
        criterion = nn.MSELoss(reduction = 'sum')
        
        if method == 'BPTT':
            for i in range(len(net.w)):
                if net.w[i] is not None:
                    net.w[i].weight.requires_grad = True
                    if net.w[i].bias is not None:
                        net.w[i].bias.requires_grad = True                                
               
        for batch_idx, (data, targets) in enumerate(train_loader):            
            s = net.initHidden(data.size(0))
            
            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(net.ns):
                    s[i] = s[i].to(net.device)
                
            if method == 'BPTT':    
                net.zero_grad()
                s = net.forward(data, s, method = 'withgrad')
                pred = s[0].data.max(1, keepdim=True)[1]          
                loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
                ###############################* BPTT *###################################              
                loss.backward()
                with torch.no_grad():                      
                    for i in range(len(net.w)):
                        if net.w[i] is not None:			
                            w_temp = net.w[i].weight
                            w_temp -= net.lr_tab[int(np.floor(i/2))]*w_temp.grad
                            if net.w[i].bias is not None:
                                w_temp = net.w[i].bias
                                w_temp -= net.lr_tab[int(np.floor(i/2))]*w_temp.grad                                   
                ##########################################################################
                
            elif method == 'eqprop':
                with torch.no_grad():
                    s = net.forward(data, s)
                    pred = s[0].data.max(1, keepdim=True)[1]
                    loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
                    ###################################* EQPROP *############################################
                    seq = []
                    for i in range(len(s)):
                        seq.append(s[i].clone())
                    s = net.forward(data, s, target = targets, beta = net.beta, method = 'nograd')    
                    net.updateWeights(net.beta, data, s, seq)
                    #########################################################################################

                       
            loss_tot += loss                     
            targets_temp = targets.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets_temp.data.view_as(pred)).cpu().sum()
                                        
            if (batch_idx + 1)% 100 == 0:
               print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                   100. * (batch_idx + 1) / len(train_loader), loss.data))
            
            
        loss_tot /= len(train_loader.dataset)
        
        
        print('\nAverage Training loss: {:.4f}, Training Error Rate: {:.2f}% ({}/{})\n'.format(
           loss_tot,100*(len(train_loader.dataset)- correct.item() )/ len(train_loader.dataset), len(train_loader.dataset)-correct.item(), len(train_loader.dataset),
           ))

        
        return 100*(len(train_loader.dataset)- correct.item())/ len(train_loader.dataset)
        
    else:        
        net.train()
        loss_tot = 0
        correct = 0
        criterion = nn.MSELoss(reduction = 'sum')

        # append it and use this for getting the activations
        pred_inps_with_delay = pred_inps
        pred_inps_with_delay.append(delay)   
        
        if method == 'BPTT':
            for i in range(len(net.fc)):
                    net.fc[i].weight.requires_grad = True
                    if net.fc[i].bias is not None:
                        net.fc[i].bias.requires_grad = True   
                                                     
            for i in range(len(net.conv)):
                    net.conv[i].weight.requires_grad = True
                    if net.conv[i].bias is not None:
                        net.conv[i].bias.requires_grad = True 
                                   
        for batch_idx, (data, targets) in enumerate(train_loader):
            # for prediction
            if data.size(0) != batch_size:
              continue
              
            s, inds = net.initHidden(data.size(0))
            
            ## randomly picked up data indices for the prediction and clamped phase data to update the weights
            update_data_idx   = np.random.choice(batch_size, size=update_batch_size, replace=False)
  
            ## these indices are for training LS model to predict the activations  
            train_ls_idx      = [k for k in range(batch_size) if k not in update_data_idx]            
            
            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(len(s)):
                    s[i] = s[i].to(net.device)
                
            if method == 'BPTT':  
                net.zero_grad()
                s, inds = net.forward(data, s, inds, method = 'withgrad')
                pred = s[0].data.max(1, keepdim=True)[1]          
                loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
                ###############################* BPTT *###################################              
                loss.backward()
                with torch.no_grad():                      
                    for i in range(len(net.fc)):			
                        w_temp = net.fc[i].weight
                        w_temp -= net.lr_tab[i]*w_temp.grad
                        if net.fc[i].bias is not None:
                            w_temp = net.fc[i].bias
                            w_temp -= net.lr_tab[i]*w_temp.grad
                    for i in range(len(net.conv)):			
                        w_temp = net.conv[i].weight
                        w_temp -= net.lr_tab[i + len(net.fc)]*w_temp.grad
                        if net.conv[i].bias is not None:
                            w_temp = net.conv[i].bias
                            w_temp -= net.lr_tab[i + len(net.fc)]*w_temp.grad                                                             
                ##########################################################################
                
            elif method == 'eqprop':
                with torch.no_grad():
                    # free phase
                    seq, indseq, all_activations = net.forward(data, s, inds, pred_inps_with_delay)
                    pred = s[0].data.max(1, keepdim=True)[1]
                    loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)

                    ###################################* EQPROP *############################################
                    seq          = []
                    indseq       = []
                    s_with_delay = []                    
                    for i in range(len(s)):    
                        seq.append(s[i].clone())
                        
                        # for delay
                        if i <=0: 
                          s_with_delay.append(all_activations[i][:, :, -1].float().cuda().clone())
                        else:                          
                          s_with_delay.append(all_activations[i][:, :, :, :, -1].float().cuda().clone())
                          
                        if inds[i] is not None:
                            indseq.append(inds[i].clone())
                        else:
                            indseq.append(None)
                            
                    # clamped phase  
                    s, inds, _  = net.forward(data, s_with_delay, inds, [0], seq, indseq, target = targets, beta = net.beta, method = 'nograd')   
                    
                     
                    update_s       = []
                    pred_inp_store = []
                    for k in range(len(s)):
                      update_s.append(s[k][update_data_idx])
                      if k <= 0: 
                        pred_inp_store.append(all_activations[k][:, :, :-1])
                      else:
                        pred_inp_store.append(all_activations[k][:, :, :, :, :-1])
                        
                    ## predicting steady state from several points at 1st phase
                    seq     = predict_acts(pred_inp_store, seq, update_data_idx, train_ls_idx )

                    net.updateWeights(net.beta, data, update_s, inds, seq, indseq, update_data_idx, epoch) 
                    #########################################################################################

                       
            loss_tot += loss                     
            targets_temp = targets.data.max(1, keepdim=True)[1]
            correct += pred.eq(targets_temp.data.view_as(pred)).cpu().sum()
                                 
            
            if (batch_idx + 1)% 100 == 0:
               print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                   epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                   100. * (batch_idx + 1) / len(train_loader), loss.data))
            
            
        loss_tot /= len(train_loader.dataset)
        
        
        print('\nAverage Training loss: {:.4f}, Training Error Rate: {:.2f}% ({}/{})\n'.format(
           loss_tot,100*(len(train_loader.dataset)- correct.item() )/ len(train_loader.dataset), len(train_loader.dataset)-correct.item(), len(train_loader.dataset),
           ))
        
        return 100*(len(train_loader.dataset)- correct.item())/ len(train_loader.dataset)
    
def evaluate(net, test_loader): 
    if not hasattr(net, 'C_tab'):
        net.eval()
        loss_tot_test = 0
        correct_test = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader): 
                s = net.initHidden(data.size(0))             
                if net.cuda:
                    data, targets = data.to(net.device), targets.to(net.device)
                    for i in range(net.ns):
                        s[i] = s[i].to(net.device)
                                      
                s = net.forward(data, s, method = 'nograd')
                 
                loss_tot_test += (1/2)*((s[0]-targets)**2).sum()                
                pred = s[0].data.max(1, keepdim = True)[1]
                targets_temp = targets.data.max(1, keepdim = True)[1]
                correct_test += pred.eq(targets_temp.data.view_as(pred)).cpu().sum()
                
        loss_tot_test = loss_tot_test / len(test_loader.dataset)
        accuracy = correct_test.item() / len(test_loader.dataset)
        print('\nAverage Test loss: {:.4f}, Test Error Rate: {:.2f}% ({}/{})\n'.format(
            loss_tot_test,100. *(len(test_loader.dataset)- correct_test.item() )/ len(test_loader.dataset), len(test_loader.dataset)-correct_test.item(), len(test_loader.dataset)))        
        return 100 *(len(test_loader.dataset)- correct_test.item() )/ len(test_loader.dataset)

    else:
        net.eval()
        loss_tot_test = 0
        correct_test = 0
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader): 
                s, inds = net.initHidden(data.size(0))             
                if net.cuda:
                    data, targets = data.to(net.device), targets.to(net.device)
                    for i in range(len(s)):
                        s[i] = s[i].to(net.device)
                                      
                s, inds, _ = net.forward(data, s, inds, method = 'nograd')
                 
                loss_tot_test += (1/2)*((s[0]-targets)**2).sum()                
                pred = s[0].data.max(1, keepdim = True)[1]
                targets_temp = targets.data.max(1, keepdim = True)[1]
                correct_test += pred.eq(targets_temp.data.view_as(pred)).cpu().sum()
                
        loss_tot_test = loss_tot_test / len(test_loader.dataset)
        accuracy = correct_test.item() / len(test_loader.dataset)
        print('\nAverage Test loss: {:.4f}, Test Error Rate: {:.2f}% ({}/{})\n'.format(
            loss_tot_test,100. *(len(test_loader.dataset)- correct_test.item() )/ len(test_loader.dataset), len(test_loader.dataset)-correct_test.item(), len(test_loader.dataset)))        
        return 100 *(len(test_loader.dataset)- correct_test.item() )/ len(test_loader.dataset)    
    
    

def compute_nSdSDT(net, data, target): 

    if not hasattr(net, 'C_tab'):
        beta = net.beta
        batch_size_temp = data.size(0)
        s = net.initHidden(batch_size_temp)    
        if net.cuda: 
            for i in range(net.ns):
                s[i] = s[i].to(net.device)
            
        net.zero_grad()
        s, nS = net.forward(data, s, target = target, method = 'nS')
        
        
        seq = []
        for i in range(len(s)):         
            seq.append(s[i].clone())
        with torch.no_grad():
            s, dS, DT = net.forward(data, s, seq, target = target, beta = beta, method = 'dSDT')

        return nS, dS, DT, seq

    else:
        beta = net.beta
        batch_size_temp = data.size(0)
        s, inds = net.initHidden(batch_size_temp)    
        if net.cuda: 
            for i in range(len(s)):
                s[i] = s[i].to(net.device)
            
        net.zero_grad()
        s, inds, nS = net.forward(data, s, inds, target = target, method = 'nS')
        
        
        seq = []
        indseq = []
        for i in range(len(s)):       
            seq.append(s[i].clone())
            if inds[i] is not None:
                indseq.append(inds[i].clone())
            else:
                indseq.append(None)
            
        with torch.no_grad():
            s, inds, dS, DT_conv, DT_fc = net.forward(data, s, inds, seq, indseq, target = target, beta = beta, method = 'dSDT')

        return nS, dS, [DT_conv, DT_fc], seq
                            

def compute_NT(net, data, target, wholeProcess = True):

    if not hasattr(net, 'C_tab'):
        batch_size_temp = data.size(0)
        
        NT = []
        for i in range(len(net.w)):
            if net.w[i] is not None:
                NT.append(torch.zeros(net.Kmax, net.w[i].weight.size(0), net.w[i].weight.size(1)))
            else:
                NT.append(None)
            
        criterion = nn.MSELoss(reduction = 'sum')
        if wholeProcess:
            for K in range(net.Kmax):
                print(K)
                s = net.initHidden(batch_size_temp)
                if net.cuda: 
                    for i in range(net.ns):
                        s[i] = s[i].to(net.device)     
                net.zero_grad()
                s = net.forward(data, s, method = 'withgrad', K = K)    
                loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
                loss.backward()
                
                for i in range(len(NT)):
                    if (net.w[i] is not None):
                        if (net.w[i].weight.grad is not None):
                            NT[i][K, :, :] = net.w[i].weight.grad.clone()
        else:
                s = net.initHidden(batch_size_temp)
                if net.cuda: 
                    for i in range(net.ns):
                        s[i] = s[i].to(net.device)     
                net.zero_grad()
                s = net.forward(data, s, method = 'withgrad', K = net.Kmax)    
                loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
                loss.backward()
                
                for i in range(len(NT)):
                    if net.w[i] is not None:
                        NT[i][-1, :, :] = net.w[i].weight.grad.clone()
                         
        return NT


    else:
        batch_size_temp = data.size(0)
        NT_fc = []
        NT_conv = []

        for i in range(len(net.fc)):
            NT_fc.append(torch.zeros(net.Kmax, net.fc[i].weight.size(0), net.fc[i].weight.size(1)))    
             
        for i in range(len(net.conv)):
            NT_conv.append(torch.zeros(net.Kmax, net.conv[i].weight.size(0), net.conv[i].weight.size(1), 
                                    net.conv[i].weight.size(2), net.conv[i].weight.size(3)))  
           
        criterion = nn.MSELoss(reduction = 'sum')
        if wholeProcess:
            for K in range(net.Kmax):
                print(K)
                s, inds = net.initHidden(batch_size_temp)
                if net.cuda: 
                    for i in range(len(s)):
                        s[i] = s[i].to(net.device)     
                net.zero_grad()
                s, inds = net.forward(data, s, inds, method = 'withgrad', K = K)    
                loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
                loss.backward()
                
                for i in range(len(NT_fc)):
                        NT_fc[i][K, :] = net.fc[i].weight.grad.clone()
                        
                for i in range(len(NT_conv)):
                        if net.conv[i].weight.grad is not None:
                            NT_conv[i][K, :] = net.conv[i].weight.grad.clone()                    
                        
        else:
                s, inds = net.initHidden(batch_size_temp)
                if net.cuda: 
                    for i in range(len(s)):
                        s[i] = s[i].to(net.device)     
                net.zero_grad()
                s, inds = net.forward(data, s, inds, method = 'withgrad', K = net.Kmax)    
                loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
                loss.backward()
                
                for i in range(len(NT_fc)):
                        NT_fc[i][-1, :] = net.fc[i].weight.grad.clone()
                        
                for i in range(len(NT_conv)):
                        NT_conv[i][-1, :] = net.conv[i].weight.grad.clone()   
                         
        return [NT_conv, NT_fc]   


def compute_nTdT(NT, DT):

    if not isinstance(NT[0], list):
        nT = []
        dT = []
        for i in range(len(NT)):
            if NT[i] is not None:
                nT.append(torch.zeros_like(NT[i]))
                dT.append(torch.zeros_like(DT[i]))
            else:
                nT.append(None)
                dT.append(None)            

        for i in range(len(NT)):
            if NT[i] is not None:
                for t in range(NT[i].size(0) - 1):
                    nT[i][t + 1, :, :] = NT[i][t + 1, :, :] - NT[i][t, :, :]
                    dT[i][t + 1, :, :] = DT[i][t + 1, :, :] - DT[i][t, :, :]       
        return nT, dT
    else:
        nT_conv = []
        nT_fc = []    
        dT_conv = []
        dT_fc = []

        NT_conv = NT[0]
        NT_fc = NT[1]
        DT_conv = DT[0]
        DT_fc = DT[1]    
        
        for i in range(len(NT_fc)):
                nT_fc.append(torch.zeros_like(NT_fc[i]))
                dT_fc.append(torch.zeros_like(DT_fc[i]))
                
        for i in range(len(NT_conv)):
                nT_conv.append(torch.zeros_like(NT_conv[i]))
                dT_conv.append(torch.zeros_like(DT_conv[i]))   

        for i in range(len(NT_fc)):
                for t in range(NT_fc[i].size(0) - 1):
                    nT_fc[i][t + 1, :, :] = NT_fc[i][t + 1, :, :] - NT_fc[i][t, :, :]
                    dT_fc[i][t + 1, :, :] = DT_fc[i][t + 1, :, :] - DT_fc[i][t, :, :]
                    
        for i in range(len(NT_conv)):
                for t in range(NT_conv[i].size(0) - 1):
                    nT_conv[i][t + 1, :, :] = NT_conv[i][t + 1, :, :] - NT_conv[i][t, :, :]
                    dT_conv[i][t + 1, :, :] = DT_conv[i][t + 1, :, :] - DT_conv[i][t, :, :]
        
        return [nT_conv, nT_fc], [dT_conv, dT_fc]


          

def receipe(net, train_loader, N_trials):

    if hasattr(net, 'conv'):        
        counter_sign_T_fc = np.zeros((N_trials, len(net.fc)))
        counter_sign_T_conv = np.zeros((N_trials, len(net.conv)))
        counter_zero_T_fc = np.zeros((N_trials, len(net.fc)))
        counter_zero_T_conv = np.zeros((N_trials, len(net.conv)))
      
        for n in range(N_trials):
            print('mini-batch {}/{}'.format(n + 1, N_trials))
            batch_idx, (data, targets) = next(enumerate(train_loader))
            batch_size = data.size(0)                                  
            s, inds = net.initHidden(batch_size)
            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(len(s)):
                    s[i] = s[i].to(net.device)
            
            #Check dS, nS, DT computation
            nS, dS, DT, _ = compute_nSdSDT(net, data, targets)
            
            DT_conv, DT_fc = DT
			
            #Check NT computation		       
            NT = compute_NT(net, data, targets, wholeProcess = False)
            NT_conv, NT_fc = NT
 

            #***************************COMPUTE PROPORTION OF SYNAPSES WHICH HAVE THE GOOD SIGN******************************#
            
            for i in range(len(NT_fc)):
                size_temp = DT_fc[i][-1, :].view(-1,).size()[0]
                counter_temp = ((torch.sign(NT_fc[i][-1, :]) == torch.sign(DT_fc[i][-1, :])) & (torch.abs(NT_fc[i][-1, :]) > 0) & (torch.abs(DT_fc[i][-1, :]) > 0)).sum().item()*100/size_temp

                counter_temp_2 = ((NT_fc[i][-1, :] == DT_fc[i][-1, :]) & (NT_fc[i][-1, :] == torch.zeros_like(NT_fc[i][-1, :]))).sum().item()*100/size_temp
                
                  
                counter_sign_T_fc[n, i] = counter_temp
                counter_zero_T_fc[n, i] = counter_temp_2
                print('fc layer {}: {:.1f}% (same sign, total), i.e. {:.1f}% (stricly non zero), {:.1f}% (both zero)'.format(i, counter_temp + counter_temp_2, counter_temp, counter_temp_2))


            for i in range(len(NT_conv)):
                size_temp = DT_conv[i][-1, :].view(-1,).size()[0]
                counter_temp = ((torch.sign(NT_conv[i][-1, :]) == torch.sign(DT_conv[i][-1, :])) & (torch.abs(NT_conv[i][-1, :]) > 0) & (torch.abs(DT_conv[i][-1, :]) > 0)).sum().item()*100/size_temp

                counter_temp_2 = ((NT_conv[i][-1, :] == DT_conv[i][-1, :]) & (NT_conv[i][-1, :] == torch.zeros_like(NT_conv[i][-1, :]))).sum().item()*100/size_temp


                counter_sign_T_conv[n, i] = counter_temp
                counter_zero_T_conv[n, i] = counter_temp_2
                print('conv layer {}: {:.1f}% (same sign, total), i.e. {:.1f}% (stricly non zero), {:.1f}% (both zero)'.format(i, counter_temp + counter_temp_2, counter_temp, counter_temp_2))
        
        #***************************************************************************************************************#

        print('************Statistics on {} trials************'.format(N_trials))
        for i in range(len(NT_fc)):
            print('average fc layer {}: {:.1f} +- {:.1f}%  (same sign, total), i.e. {:.1f} +- {:.1f}%  (stricly non zero), {:.1f} +- {:.1f}%  (both zero)'.format(i, 
                    counter_sign_T_fc[:, i].mean() + counter_zero_T_fc[:, i].mean(), 
                    counter_sign_T_fc[:, i].std() + counter_zero_T_fc[:, i].std(), 
                    counter_sign_T_fc[:, i].mean(), 
                    counter_sign_T_fc[:, i].std(), 
                    counter_zero_T_fc[:, i].mean(),
                    counter_zero_T_fc[:, i].std()))

        for i in range(len(NT_conv)):
            print('average conv layer {}: {:.1f} +- {:.1f}%  (same sign, total), i.e. {:.1f} +- {:.1f}%  (stricly non zero), {:.1f} +- {:.1f}%  (both zero)'.format(i, counter_sign_T_conv[:, i].mean() + counter_zero_T_conv[:, i].mean(), 
                counter_sign_T_conv[:, i].std() + counter_zero_T_conv[:, i].std(), 
                counter_sign_T_conv[:, i].mean(), 
                counter_sign_T_conv[:, i].std(),
                counter_zero_T_conv[:, i].mean(),
                counter_zero_T_conv[:, i].std()))
        print('***********************************************')
        print('done')


    else:
        counter_sign_T = np.zeros((N_trials, len(net.w)))
        counter_zero_T = np.zeros((N_trials, len(net.w)))
      
        for n in range(N_trials):
            print('mini-batch {}/{}'.format(n + 1, N_trials))
            batch_idx, (data, targets) = next(enumerate(train_loader))
            batch_size = data.size(0)                                  
            s = net.initHidden(batch_size)
            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(len(s)):
                    s[i] = s[i].to(net.device)
            
            #Check dS, nS, DT computation
            nS, dS, DT, _ = compute_nSdSDT(net, data, targets)

            #Check NT computation		       
            NT = compute_NT(net, data, targets, wholeProcess = False)

            #***************************COMPUTE PROPORTION OF SYNAPSES WHICH HAVE THE GOOD SIGN******************************#
            
            for i in range(len(NT)):
                if NT[i] is not None:
                    size_temp = DT[i][-1, :].view(-1,).size()[0]
                    counter_temp = ((torch.sign(NT[i][-1, :]) == torch.sign(DT[i][-1, :])) & (torch.abs(NT[i][-1, :]) > 0) & (torch.abs(DT[i][-1, :]) > 0)).sum().item()*100/size_temp

                    counter_temp_2 = ((NT[i][-1, :] == DT[i][-1, :]) & (NT[i][-1, :] == torch.zeros_like(NT[i][-1, :]))).sum().item()*100/size_temp
                                     
                    counter_sign_T[n, i] = counter_temp
                    counter_zero_T[n, i] = counter_temp_2

                    print('layer {}: {:.1f}% (same sign, total), i.e. {:.1f}% (stricly non zero), {:.1f}% (both zero)'.format(int(i/2), counter_temp + counter_temp_2, counter_temp, counter_temp_2))

        
        #***************************************************************************************************************#

        print('************Statistics on {} trials************'.format(N_trials))

        for i in range(len(NT)):
            if NT[i] is not None:
                print('average layer {}: {:.1f} +- {:.1f}%  (same sign, total), i.e. {:.1f} +- {:.1f}%  (stricly non zero), {:.1f} +- {:.1f}%  (both zero)'.format(int(i/2), 
                        counter_sign_T[:, i].mean() + counter_zero_T[:, i].mean(), 
                        counter_sign_T[:, i].std() + counter_zero_T[:, i].std(), 
                        counter_sign_T[:, i].mean(), 
                        counter_sign_T[:, i].std(), 
                        counter_zero_T[:, i].mean(),
                        counter_zero_T[:, i].std()))
        print('***********************************************')
        print('done')

              
   

def createPath(args):

    if args.action == 'train':
        BASE_PATH = os.getcwd() + '/' 

        name = 'ep'
    
        if args.conv: 
            name = name + '_conv'
        else:
            if args.discrete:
                name = name + '_disc'
            else:
                name = name + '_cont'
            if not args.toymodel:
                name = name + '_' + str(len(args.size_tab) - 2) + 'hidden'
            else:
                name = name + '_toymodel'
                    
        BASE_PATH = BASE_PATH + name

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        BASE_PATH = BASE_PATH + '/' + datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d")

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        files = os.listdir(BASE_PATH)

        if not files:
            BASE_PATH = BASE_PATH + '/' + 'Trial-1'
        else:
            tab = []
            for names in files:
                tab.append(int(names[-1]))
            BASE_PATH = BASE_PATH + '/' + 'Trial-' + str(max(tab)+1)                                
        
        os.mkdir(BASE_PATH) 
        filename = 'results'   
        
        #**********************************************************#
        copyfile('plotFunctions.py', BASE_PATH + '/plotFunctions.py')
        #**********************************************************#

        return BASE_PATH, name
    
    elif args.action == 'plotcurves':
        BASE_PATH = os.getcwd() + '/' 

        name = 'ep'


        if args.conv: 
            name = name + '_conv'
        else:
            if args.discrete:
                name = name + '_disc'
            else:
                name = name + '_cont'
            if not args.toymodel:
                name = name + '_' + str(len(args.size_tab) - 2) + 'hidden'
            else:
                name = name + '_toymodel'
                    
        BASE_PATH = BASE_PATH + name

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        files = os.listdir(BASE_PATH)

        if not files:
            BASE_PATH = BASE_PATH + '/' + 'Trial-1'
        else:
            tab = []
            for names in files:
                tab.append(int(names[-1]))
            BASE_PATH = BASE_PATH + '/' + 'Trial-' + str(max(tab)+1)                                
        
        os.mkdir(BASE_PATH) 
        filename = 'results'   
        
        #********************************************************#
        copyfile('plotFunctions.py', BASE_PATH + '/plotFunctions.py')
        #********************************************************#

        return BASE_PATH, name


def createHyperparameterfile(BASE_PATH, name, args):    

    if args.action == 'train':
        hyperparameters = open(BASE_PATH + r"/hyperparameters.txt","w+") 
        L = [" TRAINING: list of hyperparameters " + "(" + name + ", " + datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d") + ") \n",
            "- T: {}".format(args.T) + "\n",
            "- Kmax: {}".format(args.Kmax) + "\n",
            "- beta: {:.3f}".format(args.beta) + "\n", 
            "- batch size: {}".format(args.batch_size) + "\n",
            "- activation function: " + args.activation_function + "\n",
            "- number of epochs: {}".format(args.epochs) + "\n",
            "- learning rates: {}".format(args.lr_tab) + "\n"]

        if not args.discrete:
            L.append("- dt: {:.3f}".format(args.dt) + "\n")   

        if args.conv:
            L.append("- channel sizes: {}".format(args.C_tab) + "\n")
            L.append("- classifier sizes: {}".format(args.size_tab) + "\n")
            L.append("- filter size: {}".format(args.Fconv) + "\n")
            if args.padding == 1:
                L.append("- padded layers: yes !\n")
            else:
                L.append("- padded layers: no\n")     
        else:
            L.append("- layer sizes: {}".format(args.size_tab) + "\n")

        hyperparameters.writelines(L) 
        hyperparameters.close()
    
    elif args.action == 'plotcurves':        
        hyperparameters = open(BASE_PATH + r"/hyperparameters.txt","w+") 
        L = ["NABLA-DELTA CURVES: list of hyperparameters " + "(" + name + ", " + datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d") + ") \n",
            "- T: {}".format(args.T) + "\n",
            "- Kmax: {}".format(args.Kmax) + "\n",
            "- beta: {:.3f}".format(args.beta) + "\n", 
            "- batch size: {}".format(args.batch_size) + "\n",
            "- activation function: " + args.activation_function + "\n"]

        if not args.discrete:
            L.append("- dt: {:.3f}".format(args.dt) + "\n")   

        if args.conv:
            L.append("- channel sizes: {}".format(args.C_tab) + "\n")
            L.append("- classifier sizes: {}".format(args.size_tab) + "\n")
            L.append("- filter size: {}".format(args.Fconv) + "\n")
            if args.padding == 1:
                L.append("- padded layers: yes !\n")
            else:
                L.append("- padded layers: no\n")     
        else:
            L.append("- layer sizes: {}".format(args.size_tab) + "\n")

        hyperparameters.writelines(L) 
        hyperparameters.close()        

    