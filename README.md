# Neuronal learning rule based on predicting its future activity?

To run the spiking neural networks with the predictions, go:
```
  git clone https://github.com/quva-lab/spiking-eqprop.git
  cd spiking-eqprop
  source setup.sh

  pip3 install -e git+http://github.com/QUVA-Lab/artemis.git#egg=artemis 
```
and replace the codes in the directory 'spiking-eqprop' with our codes in 'bioCHLspk'.
These implementations are based on https://github.com/quva-lab/spiking-eqprop 

For our experiments, we used the spkinking neural networks with Optimal Step-Size Adaptation
(OSA). If you want to run the code, go:
```
python demo_mnist_quantized_eqprop.py 

```
and when you run the code, the program ask you like "Enter command or experiment # to run (h for help) >>"  
So plese enter 17 for our model.
*Beofre running the program, please create the directories for each model. 
 For CHL_clamped.py, "with_delay13_inp12_clamped_f120_c120_ada"
 For CHL_inh_exc.py, "with_delay13_inp12_f120_c120_inh_exc_ada"
 For the spiking neural network, "with_delay18_inp17_pred_spike_ada_lr001001_1000_min10_skip2"
 *You can change the directory names if you want in the python files.

