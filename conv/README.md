# Convolutional neural network with the predictions on CIFAR-10

These codes are based on "Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input" (https://github.com/ernoult/updatesEPgradientsBPTT)
We modified the codes for our prediction model on CIFAR-10.<br/>
If you want to run the code, go:
```
python main.py 
```
The hyper-parameters are already set up for our model in the paper. But if you want to change the hyper-parameters, please change the lines between 13 - 154 in main.py <br/> <br/>
*This python code will create directories "ep_conv/cuda0-20xx-xx-xx/Trial-x" (xx parts are year, month, and date. x part is the trial number. e.g. if you run it on 2021/05/11 first time, then it will create directories "ep_conv/cuda0-2021-05-11/Trial-1") to save the results (results) and parameters (model_epochN.pt, N is epoch number). Also, in the directory, a text file (hyperparameters.txt) will be created to tell you a summary of hyper-parameters.<br/> 

Training this network will take time. It will take around one hour to finish an epoch (in our case, it takes 1 hour and 10 min for one epoch with GPU Geforce RTX 2080 Super).<br/><br/>
If you want to check the learning curves for this model, go to the directory "ep_conv/cuda0-20xx-xx-xx/Trial-x", and go:

```
python plotFunctions.py
```
It shows you the learning curves for training and testing. Also, it shows you the best testing error so far. <br/>

This model tested on: <br/>
Python 3.7.9 <br/>
Pytorch 1.7.0 <br/>
Tensorflow 1.14.0 <br/>
Sklearn 0.22.1 <br/>
Numpy 1.18.1 <br/>
