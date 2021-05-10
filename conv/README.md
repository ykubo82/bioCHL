# Convolutional Neural Network with the predictions

These codes are based on "Updates of Equilibrium Prop Match Gradients of Backprop Through Time in an RNN with Static Input" (https://github.com/ernoult/updatesEPgradientsBPTT)
We modified the codes for our prediction model.
If you want to run the code, go:
```
python main.py 

```
The parameters are already set up for our model in the paper. But if you want to change the parameters, please change Line 12 - 154 in main.py <br/>

*This python code will create a directory "ep_conv" to save the results and parameters.

This model tested on: <br/>
Python 3.7.9 <br/>
Pytorch 1.7.0 <br/>
Tensorflow 1.14.0 <br/>
Sklearn 0.22.1 <br/>
Numpy 1.18.1 <br/>
