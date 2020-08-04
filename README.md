# Neuronal learning rule based on predicting its future activity?

To run the spiking neural networks with the predictions, go:
```
  git clone https://github.com/quva-lab/spiking-eqprop.git
  cd spiking-eqprop
  source setup.sh

  pip3 install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis 
```
and replace the codes in the directory 'spiking-eqprop' with our codes.
These implementations are based on https://github.com/quva-lab/spiking-eqprop 

For our experiments, we used the spkinking neural networks with Optimal Step-Size Adaptation
(OSA). If you want to run the code, go:
```
python demo_mnist_quantized_eqprop.py 

```
and plese select the #17 model.

