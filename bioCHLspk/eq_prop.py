from abc import abstractmethod
from typing import Sequence, NamedTuple, Optional, Tuple, Callable, Iterable
import dataclasses
from dataclasses import dataclass
from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import izip_equal
import torch
import numpy as np


def rho(x):
    return torch.clamp(x, 0, 1)


def drho(x):
    return ((x>=0) * (x<=1)).float()


def last(iterable):

    gen = iter(iterable)
    x = next(gen)
    for x in gen:
        pass
    return x

## get activations for predictions 
## return inputs and targets for the predictions
## added
def get_all_potaintial(iterable, prediction_inp_size, n_negative_steps, layer_states):
  length             = len(layer_states)
  activations        = []
  target_activations = []
  all_potential      = []
  for i in range(length):
    layer                 = layer_states[i].potential
    if i == 0:
      activation          = torch.zeros((np.shape(layer)[0], np.shape(layer)[1]))
    else: 
      activation          = torch.zeros((np.shape(layer)[0], np.shape(layer)[1], prediction_inp_size)) ################
      target_activation   = torch.zeros((np.shape(layer)[0], np.shape(layer)[1]))
      target_activations.append(target_activation)

    activations.append(activation)

  gen   =iter(iterable)
  count = 0
  for x in gen:
    for idx, later_state in enumerate(x):
      if idx == 0:
         activations[idx]  = later_state.potential  
         continue
      if count < (prediction_inp_size):
        activations[idx][:, :, count] = later_state.potential
        #activations[idx][count] = later_state.potential
      elif count == (n_negative_steps-1):
        target_activations[idx-1] = later_state.potential
      all_potential.append(later_state.potential) # to return all potential just in case
    count +=1
  return activations, target_activations, x, all_potential


class PropDirectionOptions:

    FORWARD = 'forward'
    BACKWARD = 'backward'
    NEUTRAL = 'neutral'
    FASTFORWARD = 'fast-forward'
    SWAP = 'swap'


class ISimpleUpdateFunction(object):

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> Tuple['ISimpleUpdateFunction', torch.Tensor]:
        pass


class LayerParams(NamedTuple):
    w_fore: Optional[torch.Tensor] = None
    w_aft: Optional[torch.Tensor] = None
    b: Optional[torch.Tensor] = None

    @property
    def n_units(self):
        return len(self.b) if self.b is not None else self.w_fore.shape[1] if self.w_fore is not None else self.w_aft.shape[0]


@dataclass
class IDynamicLayer(object):
    params: LayerParams  # Parameters (weight, biases)
    output: torch.Tensor  # The last output produced by the layer
    potential: torch.Tensor  # The Potential (i.e. "value") of the neuron.

    @abstractmethod
    def __call__(self, x_aft=None, x_fore=None, pressure = None, clamp = None) -> 'IDynamicLayer':
        pass


@dataclass
class SimpleLayerController(IDynamicLayer):

    epsilon: float

    @staticmethod
    def get_partial_constructor(epsilon):
        def partial_constructor(n_samples: int, params: LayerParams):
            return SimpleLayerController(
                params=params,
                output=torch.zeros((n_samples, params.n_units)),
                potential = torch.zeros((n_samples, params.n_units)),
                epsilon=epsilon
            )
        return partial_constructor

    def __call__(self, x_aft=None, x_fore=None, pressure = None, clamp = None, gamma = 1.0, dt=1.0):

        if clamp is not None:
            potential = clamp
        else:
            n_samples = x_aft.shape[0] if x_aft is not None else x_fore.shape[0]

            input_pressure = torch.zeros((n_samples, self.params.n_units))
            if self.params.b is not None:
                input_pressure += self.params.b
            if x_aft is not None:
                input_pressure += x_aft @ self.params.w_aft

            if x_fore is not None:
                input_pressure += gamma * x_fore @ self.params.w_fore

            #de_ds = rho(input_pressure)
            de_ds = self.potential - drho(self.potential)*input_pressure

            #potential = self.potential - dt * de_ds  # Euler integration with clipping to possible range
            potential = torch.clamp(self.potential - self.epsilon * de_ds, 0, 1)  # Euler integration with clipping to possible range

        output = rho(potential)

        return SimpleLayerController(
            epsilon=self.epsilon,
            params = self.params,
            potential = potential,
            output = output
        )

## return next output
## return targets when it is the last hidden layer 
## for contrastive hebbian learning (CHL)
## Added date : 2020/02/11
def return_next_output(layer_states, indx, y_data):
  if (indx < len(layer_states) - 1):
    if indx ==  len(layer_states) - 2 and y_data is not None:
      return y_data
    else:
      return layer_states[indx + 1].output  
  else:
    return None

def return_clamped(layer_states, indx, y_data, x_data):
  if indx == 0:
    return x_data
  elif indx ==  len(layer_states) -1 and y_data is not None:
    return y_data
  else:
    return None

def eqprop_step(layer_states: Sequence[IDynamicLayer], x_data, beta, y_data: Optional[torch.Tensor] = None, direction ='neutral') -> Sequence[IDynamicLayer]:

    assert direction in (PropDirectionOptions.FORWARD, PropDirectionOptions.BACKWARD, PropDirectionOptions.NEUTRAL)
    layer_ix = range(len(layer_states)) if direction in (PropDirectionOptions.FORWARD, PropDirectionOptions.NEUTRAL) else range(len(layer_states))[::-1]
    layer_states = list(layer_states)
    new_layers = [None]*len(layer_states)
    for ix in layer_ix:
        new_state = layer_states[ix](
            x_aft = None if ix==0 else layer_states[ix - 1].output,
            #x_fore = layer_states[ix + 1].output if ix < len(layer_states) - 1 else None,
            x_fore = return_next_output(layer_states, ix, y_data),
            clamp = return_clamped(layer_states, ix, y_data, x_data), 
            #clamp = x_data if ix==0 else None,
            #pressure = beta * 2*(y_data - layer_states[-1].potential) if (y_data is not None and ix == len(layer_states) - 1) else None
            pressure = None
        )

        # if  ix < len(layer_states) - 1 and y_data is not None:
        #   print('len-1')
        #   print(len(layer_states) - 1)
        #   print(ix)
        #   print('pred?')
        #   test = torch.max(layer_states[ix + 1].output, 1)
        #   print(test)
        #   print('target')
        #   print(torch.max(y_data,1))

        new_layers[ix] = new_state
        if direction in (PropDirectionOptions.FORWARD, PropDirectionOptions.BACKWARD):
            layer_states[ix] = new_state

    # dbplot_collection([layer_states[0].output.reshape(-1, 28, 28)]+[s.output for s in layer_states[1:]], 'outputs')
    # dbplot_collection([layer_states[0].potential.reshape(-1, 28, 28)]+[s.potential for s in layer_states[1:]], 'outputs')
    return new_layers


def eqprop_fast_forward_step(layer_states: Sequence[IDynamicLayer], x_data):
    layer_states = list(layer_states)
    new_layers = []
    for ix, layer in enumerate(layer_states):
        new_state = layer_states[ix](
            x_aft = None if ix==0 else new_layers[-1].output,
            x_fore = None,
            clamp = x_data if ix==0 else None,
            pressure = None
        )
        # new_state = layer_states[ix](
        #     x_aft = None if ix==0 else new_layers[-1].output,
        #     x_fore = None,
        #     clamp = x_data if ix==0 else rho(new_layers[-1].output @ layer_states[ix].params.w_aft + layer_states[ix].params.b),
        #     pressure = None
        # )
        new_layers.append(new_state)
    return new_layers


def eqprop_update(negative_acts, positive_acts, ws, bs, learning_rate, beta, bidirectional, l2_loss = None, gamma=0.5, dy_squared=None):

    n_samples = negative_acts[0].shape[0]
    w_grads = [-(pa_pre.float().cuda().t() @ pa_post.float().cuda() - pa_pre.float().cuda().t() @ na_post.float().cuda())/float(n_samples) for na_pre, na_post, pa_pre, pa_post in izip_equal(negative_acts[:-1], negative_acts[1:], positive_acts[:-1], positive_acts[1:])]

    ## AdaGrad
    for i, dy in enumerate(w_grads):
      if dy_squared[i] is None:
        dy_squared[i] = dy * dy
      else:
        dy_squared[i] += dy * dy
    
      w_grads[i] = dy/(torch.sqrt(dy_squared[i]) + 1e-7)
    
    b_grads = [-torch.mean(pa_post-na_post, dim=0) for pa_post, na_post in izip_equal(positive_acts[1:], negative_acts[1:])]
    if l2_loss is not None:
        w_grads = [(1-l2_loss)*wg for wg in w_grads]
        b_grads = [(1-l2_loss)*bg for bg in b_grads]

    if not isinstance(learning_rate, (list, tuple)):
        learning_rate = [learning_rate]*len(ws)

    new_ws = [w - lr * w_grad for w, w_grad, lr in izip_equal(ws, w_grads, learning_rate)]
    new_bs = [b - lr * b_grad for b, b_grad, lr in izip_equal(bs, b_grads, learning_rate)]
    return new_ws, new_bs, dy_squared
  

def _params_vals_to_params(ws: Sequence[torch.Tensor], bs: Sequence[torch.Tensor]):
    return [LayerParams(w_aft = None if i==0 else ws[i-1], w_fore = ws[i].t() if i<len(ws) else None, b=None if i==0 else bs[i-1]) for i in range(len(ws)+1)]


def uniform(low, high, size):
    return (high-low)*torch.rand(size) + low


def initialize_params(layer_sizes: Sequence[int], initial_weight_scale=1., rng = None) -> Sequence[LayerParams]:
    rng = get_rng(rng)
    ws = [uniform(low=-initial_weight_scale*(6./(n_pre+n_post))**.5, high=(6./(n_pre+n_post))**.5, size=(n_pre, n_post))
      for n_pre, n_post in izip_equal(layer_sizes[:-1], layer_sizes[1:])]
    bs = [torch.zeros(n_post) for n_post in layer_sizes[1:]]
    return _params_vals_to_params(ws, bs)


def initialize_states(layer_constructor: Callable[[int, LayerParams], IDynamicLayer], n_samples: int, params: Sequence[LayerParams]) -> Sequence[IDynamicLayer]:
    return [layer_constructor(n_samples, p) for p in params]


def output_from_state(states: Sequence[IDynamicLayer]):
    return states[-1].potential


def run_negative_phase(x_data, layer_states: Sequence[IDynamicLayer], n_steps, prop_direction) -> Iterable[Sequence[IDynamicLayer]]:
    if prop_direction==PropDirectionOptions.SWAP:
        prop_direction = PropDirectionOptions.FORWARD
    if prop_direction==PropDirectionOptions.FASTFORWARD:
        for t in range(n_steps):
            layer_states = eqprop_fast_forward_step(layer_states=layer_states, x_data=x_data)
            yield layer_states
    else:
        for t in range(n_steps):
            layer_states = eqprop_step(layer_states=layer_states, x_data = x_data, y_data=None, beta=0, direction=prop_direction)
            yield layer_states

## added delay
## updated date : 2020/06/24
def run_positive_phase(x_data, y_data, beta, delay, layer_states: Sequence[IDynamicLayer], n_steps, prop_direction) -> Iterable[Sequence[IDynamicLayer]]:
    if prop_direction==PropDirectionOptions.SWAP:
        prop_direction = PropDirectionOptions.BACKWARD
    for t in range(n_steps):
        if t >= delay: # delay
          layer_states = eqprop_step(layer_states=layer_states, x_data = x_data, y_data=y_data, beta=beta, direction=prop_direction)
        else:
          layer_states = eqprop_step(layer_states=layer_states, x_data = x_data, y_data=None, beta=beta, direction=prop_direction)
        yield layer_states


def run_inference(x_data, states: Sequence[IDynamicLayer], n_steps: int, prop_direction=PropDirectionOptions.NEUTRAL):

    negative_states = last(run_negative_phase(x_data=x_data, layer_states=states, n_steps=n_steps, prop_direction=prop_direction))
    return output_from_state(negative_states)

## linear prediction
## return prediction of dynamics
## Added date:  2020/02/11
## Updated date: 2020/06/24
def predict_dynamics(negative_activations, negative_target_activations, layer_states, device, update_data_idx, train_ls_idx):
  length = len(layer_states)
  pred_all_activations = []
  #print(np.shape(negative_activations[1]))
  # 490,1000,12
  for i in range(length):

    activaiton_all = negative_activations[i]

    if i == 0:
      activation_test_data = activaiton_all[update_data_idx, :]
      pred_all_activations.append(activation_test_data)
      continue
    
    node_size_l = np.shape(activaiton_all)[1] 
    activations = []
    
    for j in range(node_size_l):    
      ## testing data (prediction data)
      activation_test_data  = activaiton_all[update_data_idx, j, :]
  
      ## training data for the prediction
      activation_train_data = activaiton_all[train_ls_idx, j, :].cpu().numpy()
      
      activation_test_data  = activation_test_data.cpu().numpy()
      shape_train           = np.shape(activation_train_data)
      shape_test            = np.shape(activation_test_data)
        
      ## adding offset for trainig and prediction data
      activation_input_offset_train    = np.ones((shape_train[0], shape_train[1]+1))
      activation_input_offset_test     = np.ones((shape_test[0], shape_test[1]+1)) 
      
      activation_input_offset_train[:, :-1] = activation_train_data
      activation_input_offset_test[:, :-1]  = activation_test_data
  
    
      target_all = negative_target_activations[i-1] 
      activation_target = target_all[train_ls_idx, j].cpu().numpy()
      
      ## training for linear regresssion
      pred_activation    = np.linalg.lstsq(activation_input_offset_train, activation_target, rcond=None)[0]
      
      # prediction
      pred_negative_acts = activation_input_offset_test @ pred_activation
      pred_negative_acts = np.clip(pred_negative_acts, a_min=0, a_max=1.1)
      
      activations.append(torch.from_numpy(pred_negative_acts).to(device))
    pred_all_activations.append(torch.stack(activations).T)

  
  return pred_all_activations

def run_eqprop_training_update(x_data, y_data, layer_states: Sequence[IDynamicLayer], beta: float, random_flip_beta: bool,
                               learning_rate: float, n_negative_steps: int, n_positive_steps: int, layer_constructor: Optional[Callable[[int, LayerParams], IDynamicLayer]]=None,
                               bidirectional:bool=True, l2_loss:Optional[float]=None, renew_activations:bool = True, prop_direction=PropDirectionOptions.NEUTRAL, splitstream=False, 
                               rng=None, prediction_inp_size=None, delay=None,device='cpu', epoch_check=False, epoch=None, pred=False, batch_size=500, minibatch_size=20, dy_squared=None) -> Sequence[IDynamicLayer]:

    if isinstance(prop_direction, (list, tuple)):
        negative_prop_direction, positive_prop_direction = prop_direction
    else:
        negative_prop_direction, positive_prop_direction = prop_direction, prop_direction

    rng = get_rng(rng)
    this_beta = beta*(torch.randint(2, size=()).float()*2-1) if random_flip_beta else beta
    
    ## randomly picked up data indices for the prediction and clamped phase data to update the weights
    update_data_idx                            = np.random.choice(batch_size, size=minibatch_size, replace=False)
      
    ## these indices are for training LS model to predict the activations  
    train_ls_idx                               = [k for k in range(batch_size) if k not in update_data_idx]
      
    if pred:
      all_negative_states = run_negative_phase(x_data=x_data, layer_states=layer_states, n_steps=n_negative_steps, prop_direction=negative_prop_direction)
      negative_activations, negative_target_activations, negative_states, all_potential = get_all_potaintial(all_negative_states, prediction_inp_size, n_negative_steps, layer_states)
    else:
      negative_states = last(run_negative_phase(x_data=x_data, layer_states=layer_states, n_steps=n_negative_steps, prop_direction=negative_prop_direction))

    #positive_states = last(run_positive_phase(x_data=x_data, layer_states=negative_states, beta=this_beta, delay=delay, y_data=y_data, n_steps=n_positive_steps, prop_direction=positive_prop_direction))
    
    positive_states = last(run_positive_phase(x_data=x_data, layer_states=layer_states, beta=this_beta, delay=delay, y_data=y_data, n_steps=n_positive_steps, prop_direction=positive_prop_direction))
    if splitstream:
        negative_states = last(run_negative_phase(x_data=x_data, layer_states=negative_states, n_steps=n_positive_steps, prop_direction=positive_prop_direction))

    ws, bs = zip(*((s.params.w_aft, s.params.b) for s in layer_states[1:]))
    
    if pred:
    
      _, pos_acts = [[ls.potential for ls in later_state] for later_state in (negative_states, positive_states)]

      pos_acts = [pos_act[update_data_idx, :] for pos_act in pos_acts]
      
      neg_act_layer_1 = negative_activations[1]
      neg_act_layer_2 = negative_activations[2]

      negative_activations[1] = neg_act_layer_1[:, :, ::2]
      negative_activations[2] = neg_act_layer_2[:, :, ::2]      
      #print(np.shape(negative_activations[1]))
      # linear regression prediction 
      neg_acts = predict_dynamics(negative_activations, negative_target_activations, layer_states, device, update_data_idx, train_ls_idx)
    
      #print(neg_acts[1])
#      if epoch > 1.0:
#        np.save('C:/Users/yoshi/work/01_python/01_bioplausible/spiking-eqprop/spiking_eqprop/with_delay19_inp18_pred_spike_ada_lr00030002_b500_m10/negative_states_epoch_' + str(epoch) + '.npy', negative_activations)
    else:
      neg_acts, pos_acts = [[ls.potential for ls in later_state] for later_state in (negative_states, positive_states)]

    new_ws, new_bs, dy_squared = eqprop_update(
        negative_acts=neg_acts,
        positive_acts=pos_acts,
        ws=ws,
        bs=bs,
        learning_rate=learning_rate,
        beta=this_beta,
        bidirectional=bidirectional,
        l2_loss=l2_loss,
        dy_squared=dy_squared
    )
    new_params = _params_vals_to_params(new_ws, new_bs)
    if renew_activations:
        assert layer_constructor is not None, 'If you choose renew_activations true, you must provide a layer constructor.'
        new_states = initialize_states(n_samples=x_data.shape[0], params=new_params, layer_constructor=layer_constructor)
    else:
        new_states = [dataclasses.replace(s, params=p) for s, p in izip_equal(positive_states, new_params)]
    return new_states, dy_squared