from skopt.space import Real
#import sys
#sys.path.append('../src/artemis/')

from artemis.experiments.ui import browse_experiments
#from spiking_eqprop.demo_mnist_eqprop import experiment_mnist_eqprop_torch, settings, X_1hid, \
from demo_mnist_eqprop import experiment_mnist_eqprop_torch, settings, X_1hid, \
    X_3hid, set_use_cuda, X_1hid_vanilla
#from spiking_eqprop.quantized_eqprop import EncodingDecodingNeuronLayer, create_quantizer, \
from quantized_eqprop import EncodingDecodingNeuronLayer, create_quantizer, \
    IdentityFunction, KestonsStepSizer, OptimalStepSizer, PredictiveDecoder, PredictiveEncoder, create_step_sizer
from warnings import filterwarnings
filterwarnings('ignore')
"""
Here we add quantized experiments
"""

for X in list(X_1hid.get_all_variants()) + list(X_3hid.get_all_variants()):
    X.tag('vanilla')


X_1hid_quantized_scheduled = X_1hid.add_config_root_variant('quantized', layer_constructor = lambda epsilons, quantizer, lambdas=None: EncodingDecodingNeuronLayer.get_simple_constructor(epsilons=epsilons, lambdas=lambdas, quantizer=quantizer))
X_3hid_quantized_scheduled = X_3hid.add_config_root_variant('quantized', layer_constructor = lambda epsilons, quantizer, lambdas=None: EncodingDecodingNeuronLayer.get_simple_constructor(epsilons=epsilons, lambdas=lambdas, quantizer=quantizer))

N_PARAMETER_SEARCH_EPOCHS = 1

# ======================================================================================================================
# Try best parameters from demo_mnist_quantized_find_best_params

X = X_1hid_quantized_scheduled.add_config_root_variant('poly_schedule',
    epsilons = lambda eps_init, eps_exp: f'{eps_init}/t**{eps_exp}', lambdas = lambda lambda_init, lambda_exp: f'{lambda_init}/t**{lambda_exp}')
X.add_root_variant(n_epochs=N_PARAMETER_SEARCH_EPOCHS, epoch_checkpoint_period=None, quantizer='sigma_delta').add_parameter_search(
    space = dict(eps_init = Real(0, 1, 'uniform'), eps_exp = Real(0, 1, 'uniform'), lambda_init = Real(0, 1, 'uniform'), lambda_exp = Real(0, 1, 'uniform')),
    scalar_func=lambda result: result[-1, 'val_error'],
    n_calls=500
    )
X = X_3hid_quantized_scheduled.add_config_root_variant('poly_schedule',
    epsilons = lambda eps_init, eps_exp: f'{eps_init}/t**{eps_exp}', lambdas = lambda lambda_init, lambda_exp: f'{lambda_init}/t**{lambda_exp}')
X_3hid_paramsearch_base = X.add_root_variant(n_epochs=N_PARAMETER_SEARCH_EPOCHS, epoch_checkpoint_period=None, quantizer='sigma_delta')
for X in (X_3hid_paramsearch_base, X_3hid_paramsearch_base.add_root_variant(n_negative_steps = 100, n_positive_steps = 50)):
    X.add_parameter_search(
    space = dict(eps_init = Real(0, 1, 'uniform'), eps_exp = Real(0, 1, 'uniform'), lambda_init = Real(0, 1, 'uniform'), lambda_exp = Real(0, 1, 'uniform')),
    scalar_func=lambda result: result[-1, 'val_error'],
    n_calls=500
    )

# Results from demo_mnist_quantized_find_best_params (note: this are now irrelevant because of the splitstream thing)
X.add_variant('5epoch_params', quantizer='sigma_delta', eps_init=0.592, eps_exp=0, lambda_init=0.632, lambda_exp=0).tag('dead')
X.get_variant('5epoch_params').add_variant(splitstream=True).tag('dead')
"""
Was the splitstream things essential the whole time?
- Well it does help a LOT but we can still train without it, it just requires well-tuned hyperparams or longer 
  convergence times.
"""
X.add_variant('gp_params', quantizer='sigma_delta', eps_init=0.354, eps_exp=0, lambda_init=0.503, lambda_exp=0.292)  # Found in parameter search


# In final version
X_polyscheduled_longer = X_1hid_quantized_scheduled.get_variant('poly_schedule').add_root_variant('longer', n_negative_steps = 100, n_positive_steps = 50)

X_polyscheduled_longer_psearch = X_polyscheduled_longer.add_parameter_search(
    fixed_args=dict(n_epochs=N_PARAMETER_SEARCH_EPOCHS, epoch_checkpoint_period=None, quantizer='sigma_delta'),
    space = dict(eps_init = Real(0, 1, 'uniform'), eps_exp = Real(0, 1, 'uniform'), lambda_init = Real(0, 1, 'uniform'), lambda_exp = Real(0, 1, 'uniform')),
    scalar_func=lambda result: result[-1, 'val_error'],
    n_calls=500
    )



# X.add_variant('25epoch_params', quantizer='sigma_delta', eps_init=0.726, eps_exp=0, lambda_init=0.614, lambda_exp=0.797).tag('dead')
# X = X_3hid_quantized_scheduled.add_config_root_variant('poly_schedule', epsilons = lambda eps_init, eps_exp: f'{eps_init}/t**{eps_exp}', lambdas = lambda lambda_init, lambda_exp: f'{lambda_init}/t**{lambda_exp}')
# X.add_variant('5epoch_params', quantizer='sigma_delta', eps_init=0.987, eps_exp=0, lambda_init=1, lambda_exp=0.818).tag('dead')
# X.add_variant('25epoch_params', quantizer='sigma_delta', eps_init=0.86, eps_exp=0, lambda_init=1, lambda_exp=0.876).tag('dead')


# ======================================================================================================================

# Try adaptive: Kesten's rule

X_adaptive_1hid, X_adaptive_3hid = (X.add_config_root_variant('adaptive_quantized',
    layer_constructor = lambda stepper, quantizer='sigma_delta', lambdas=None:
        EncodingDecodingNeuronLayer.get_partial_constructor(
            encoder = create_quantizer(quantizer) if lambdas is None else PredictiveEncoder(create_step_sizer(lambdas), quantizer=create_quantizer(quantizer)),
            decoder = IdentityFunction() if lambdas is None else PredictiveDecoder(create_step_sizer(lambdas)),
            stepper = stepper
            ),
    ) for X in (X_1hid, X_3hid))

# X_kestens = X_adaptive_1hid.add_config_variant('kestons', stepper = lambda a=10, b=10: KestonsStepSizer(a=a, b=b)).tag('dead')
# X_kestens.add_root_variant(n_epochs=N_PARAMETER_SEARCH_EPOCHS, epoch_checkpoint_period=None).add_parameter_search(
#     space=dict(a = Real(1, 100, 'log-uniform'), b = Real(1, 100, 'log-uniform')),
#     scalar_func=lambda result: result[-1, 'val_error'],
#     )
# X_kestens.add_variant(n_negative_steps = 100, n_positive_steps = 20).tag('dead')

"""
Results seem to indicate that kestens is worse than both scheduled and OSA.  See table at bottom
"""


# ======================================================================================================================
X_osa_1hid, X_osa_3hid = (X.add_config_variant('optimal', stepper = lambda error_stepsize_target = .01, epsilon=1e-15: OptimalStepSizer(error_stepsize_target=error_stepsize_target, epsilon=epsilon))
#X_osa_1hid, X_osa_3hid = (X.add_config_variant('optimal', stepper = lambda error_stepsize_target = .01, epsilon=0.1: OptimalStepSizer(error_stepsize_target=error_stepsize_target, epsilon=epsilon))
 
                          for X in (X_adaptive_1hid, X_adaptive_3hid))
#X_osa_1hid_longer = X_osa_1hid.add_variant(n_negative_steps = 100, n_positive_steps = 50)
X_osa_1hid_longer = X_osa_1hid.add_variant(n_negative_steps = 120, n_positive_steps = 120)
#X_osa_1hid_longer = X_osa_1hid.add_variant(n_negative_steps = 30, n_positive_steps = 30)
# X_osa_1hid.add_variant(n_negative_steps = 100, n_positive_steps = 20)
# X_osa_1hid.add_variant(n_negative_steps = 50, n_positive_steps = 20)
"""
We find that the extra convergence time really does help here.

  n_negative_steps  n_positive_steps  Result
  20                4                 Epoch: 25, Test Error: 6.2%, Train Error: 6.31
  100               50                Epoch: 25, Test Error: 2.58%, Train Error: 0.34
  100               20                Epoch: 25, Test Error: 3.25%, Train Error: 1.67
  50                20                Epoch: 25, Test Error: 4.68%, Train Error: 2.86
"""
# X_osa_1hid.add_variant(error_stepsize_target=0.332, lambdas=0.244)  # Found in hyperparameter search
# X_osa_1hid.add_variant(n_negative_steps=100, n_positive_steps=50, error_stepsize_target=0.0311, lambdas=0.818)  # Found in hyperparameter search
# X_osa_3hid.add_root_variant(n_negative_steps = 100, n_positive_steps = 50).add_variant(error_stepsize_target=0.0686, lambdas=0.771)  # Found in EXTREMELY SMALL hyperparameter search (3 points so far)
# X_osa_3hid.get_variant(n_negative_steps = 100, n_positive_steps = 50).add_variant(error_stepsize_target=0.332, lambdas=0.244)  # Found by taking the parameters of the large search with the single-layer.
# X_osa_3hid.get_variant(n_negative_steps = 100, n_positive_steps = 50).add_variant(error_stepsize_target=0.0788, lambdas=0.158)  # More up-to-date value on current search

# ======================================================================================================================

X_osa_1hid_psearch, X_osa_longer_1hid_psearch = [X.add_root_variant(n_epochs=N_PARAMETER_SEARCH_EPOCHS, epoch_checkpoint_period=None).add_parameter_search(
        space = dict(
            error_stepsize_target = Real(0.001, 1, 'log-uniform'),
            lambdas = Real(0, 1, 'uniform'),
        ),
        scalar_func=lambda result: result[-1, 'val_error'],
        ) for X in (X_osa_1hid, X_osa_1hid_longer)]



# X_osa_3hid.add_root_variant(n_epochs=5, epoch_checkpoint_period=None, n_negative_steps = 100, n_positive_steps = 50).add_parameter_search(
#         space = dict(
#             error_stepsize_target = Real(0.001, 1, 'log-uniform'),
#             lambdas = Real(0, 1, 'uniform'),
#         ),
#         scalar_func=lambda result: result[-1, 'val_error'],
#         )
# X_osa_3hid.add_root_variant(n_epochs=1, epoch_checkpoint_period=None, n_negative_steps = 100, n_positive_steps = 50).add_parameter_search(
#         space = dict(
#             error_stepsize_target = Real(0.001, 1, 'log-uniform'),
#             lambdas = Real(0, 1, 'uniform'),
#         ),
#         scalar_func=lambda result: result[-1, 'val_error'],
#         )

# found these params are the best for the spiking neural network with the predictions
X_osa_longer_gp_params = X_osa_1hid_longer.add_variant('gp_params', error_stepsize_target=0.001, lambdas=0.5919087455309799)

#X_osa_longer_gp_params = X_osa_1hid_longer.add_variant('gp_params', error_stepsize_target=0.0346, lambdas=0.275)  # Latest ang greatest
#X_osa_longer_gp_params = X_osa_1hid_longer.add_variant('gp_params', error_stepsize_target= 0.40836081507586197, lambdas=0.15274531694718158)  # for CHL
#X_osa_longer_gp_params = X_osa_1hid_longer.add_variant('gp_params', error_stepsize_target= 0.381952462327752, lambdas=0.15063696560609532)  # for CHL for 824
#'error_stepsize_target': 0.381952462327752, 'lambdas': 0.15063696560609532 for 648 and 0.2?
#'error_stepsize_target': 0.381952462327752, 'lambdas': 0.15063696560609532 for 648 and 0.3?
#error_stepsize_target': 0.38040498196082184, 'lambdas': 0.1535797163205437 for 648 and 0.3?
#error_stepsize_target': 0.40836081507586197, 'lambdas': 0.15274531694718158
"""
It seems that optimial Kestens worse than both optima17l scheduled and optimal OSA
Algorithm       Best Validation score after 1 epoch     Params
scheduled       6.32                                    eps_init=0.354, eps_exp=0, lambda_init=0.503, lambda_exp=0.292
kestens         10.2                                    a=4.02, b=5.28
OSA             6.35                                    error_stepsize_target=0.332, lambdas=0.244, Score = 6.35
"""


if __name__ == '__main__':

    set_use_cuda(True)

    # X_osa_1hid.get_variant(error_stepsize_target=0.332, lambdas=0.244).call()
    # experiment_mnist_eqprop_torch.browse(filterexp='has:1_hid')
    experiment_mnist_eqprop_torch.browse()
    # experiment_mnist_eqprop_torch.browse(filterexp='tag:psearch')
    # experiment_mnist_eqprop_torch.browse(filterexp='~tag:dead&~tag:vanilla')
    # experiment_mnist_eqprop_torch.browse(filterexp='has:3_hid', raise_display_errors=False)

    # browse_experiments([X_1hid_vanilla, X_polyscheduled_longer_psearch, X_osa_longer_1hid_psearch, X_osa_longer_gp_params])