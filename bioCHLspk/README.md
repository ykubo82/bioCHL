# Spiking Neural Network with the predictions

To run the spiking neural networks with the predictions, go:
```
  git clone https://github.com/quva-lab/spiking-eqprop.git
  cd spiking-eqprop
  source setup.sh
  or 
  pip install -e git+https://github.com/quva-lab/spiking-eqprop.git#egg=spiking-eqprop
```
and install artemis, go:
```
  pip3 install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis 
```  
and replace the codes in the directory 'spiking-eqprop' with our codes in 'bioCHLspk'.
These implementations are based on https://github.com/quva-lab/spiking-eqprop   <br/>
*If artemis package does not work, please dowonload the latest verstion manually from: <br/>
https://github.com/QUVA-Lab/artemis/tree/ad2871fae7d986bf10580eec27aee5b7315adad5/artemis <br/>

For our experiments, we used the spkinking neural networks with Optimal Step-Size Adaptation
(OSA). If you want to run the code, go:
```
python demo_mnist_quantized_eqprop.py 

```
and when you run the code, the program ask you like "Enter command or experiment # to run (h for help) >>"  
So plese enter 17 for our model. <br/>
results_spike
*Beofre running the program, please create a directory for the model. <br/>
"results_spike" <br/>
*You can change the directory names in the python files, if you want. <br/>

This model tested on: <br/>
Python 3.7.9 <br/>
Pytorch 1.7.0 <br/>
Tensorflow 1.14.0 <br/>
Sklearn 0.22.1 <br/>
Numpy 1.18.1 <br/>
