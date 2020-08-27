import math
from abc import abstractmethod
from typing import Tuple, Union

import torch
from dataclasses import dataclass

from artemis.general.should_be_builtins import bad_value
from spiking_eqprop.eq_prop import ISimpleUpdateFunction, IDynamicLayer, LayerParams, \
    drho, rho


@dataclass
class StochasticQuantizer(ISimpleUpdateFunction):

    # rng: np.random.RandomState = np.random.RandomState()  # Note... this is not quite correct because rng is stateful

    def __call__(self, x: torch.Tensor) -> Tuple['StochasticQuantizer', torch.Tensor]:
        q = (x>torch.rand(*x.shape)).float()

        return StochasticQuantizer(), q


@dataclass
class ThresholdQuantizer(ISimpleUpdateFunction):

    thresh: float = 0.5

    def __call__(self, x: torch.Tensor) -> Tuple['SigmaDeltaQuantizer', torch.Tensor]:
        q = (x>self.thresh).float()
        return ThresholdQuantizer(self.thresh), q


@dataclass
class SigmaDeltaQuantizer(ISimpleUpdateFunction):

    phi: torch.Tensor = 0.

    def __call__(self, x: torch.Tensor) -> Tuple['SigmaDeltaQuantizer', torch.Tensor]:
        phi_prime = x + self.phi
        q = (phi_prime>0.5).float()
        new_phi = phi_prime - q
        return SigmaDeltaQuantizer(new_phi), q


@dataclass
class SecondOrderSigmaDeltaQuantizer(ISimpleUpdateFunction):

    phi1: torch.Tensor = 0.
    phi2: torch.Tensor = 0.

    def __call__(self, x: torch.Tensor) -> Tuple['SecondOrderSigmaDeltaQuantizer', torch.Tensor]:
        phi_1_ = self.phi_1 + x
        phi_2_ = self.phi_2 + phi_1_
        q = (phi_2_ > 0.5).float()
        return SecondOrderSigmaDeltaQuantizer(phi1=phi_1_-q, phi2=phi_2_-q), q


@dataclass
class IdentityFunction(ISimpleUpdateFunction):

    def __call__(self, x):
        return self, x


class IStepSizer(object):

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> Tuple['IStepSizer', float]:
        pass


@dataclass
class ConstantStepSizer(IStepSizer):

    step_size: float

    def __call__(self, x):
        return self, self.step_size


@dataclass
class ScheduledStepSizer(IStepSizer):

    schedule: str
    t: int = 1

    def __call__(self, x):
        step_size = eval(self.schedule, {'exp': math.exp, 'sqrt': math.sqrt}, {'t': self.t})
        return ScheduledStepSizer(schedule=self.schedule, t=self.t+1), step_size


def create_step_sizer(step_schedule):
    return ConstantStepSizer(step_schedule) if isinstance(step_schedule, (int, float)) else \
        ScheduledStepSizer(step_schedule) if isinstance(step_schedule, str) else \
        step_schedule if callable(step_schedule) else \
        bad_value(step_schedule)


def create_quantizer(quantizer):
    return SigmaDeltaQuantizer() if quantizer == 'sigma_delta' else \
        StochasticQuantizer() if quantizer == 'stochastic' else \
        ThresholdQuantizer() if quantizer == 'threshold' else \
        SecondOrderSigmaDeltaQuantizer() if quantizer == 'second_order_sd' else \
        bad_value(quantizer)


@dataclass
class KestonsStepSizer(IStepSizer):
    """
    As described in:
    https://link.springer.com/article/10.1007/s10994-006-8365-9

    """

    a: float
    b: float
    k: int = 1.
    last_err: torch.Tensor = None
    initial_step: float = 1.
    avg: torch.Tensor = 0.

    def __call__(self, x):
        if isinstance(self.k, (int, float)) and self.k<3:  # todo: Is this ok with torch?
            err = None if self.k==1 else x - self.avg
            k = self.k+1
        else:
            err = x - self.avg
            k = self.k + (self.last_err*err < 0).float()
        step_size = self.initial_step*(self.a/(self.b+k))
        avg = (1-step_size)*self.avg + step_size*x
        new_obj = KestonsStepSizer(initial_step=self.initial_step, last_err=err, k=k, avg=avg, a=self.a, b=self.b)
        return new_obj, step_size


@dataclass
class MirozamedovStepSizer(IStepSizer):
    """
    Mirozahmedov & Uryasev's step-sizer, as described in:
    As described in:
    https://link.springer.com/article/10.1007/s10994-006-8365-9

    Unless I'm mistaken, it seems like this can fail or at least be really bad at converging sometimes.
    """

    a: float
    delta: float

    step_size: float = 1.
    last_err: torch.Tensor = None
    avg: torch.Tensor = 0.

    def __call__(self, x):

        err = x - self.avg
        if self.last_err is None:
            step_size = self.step_size
        else:
            step_size = self.step_size * torch.exp((self.a*(self.last_err*err)-self.delta)*self.step_size)
        avg = (1-step_size)*self.avg + step_size*x
        new_obj = MirozamedovStepSizer(step_size=step_size, last_err=err, avg=avg, a=self.a, delta = self.delta)
        return new_obj, step_size


@dataclass
class OptimalStepSizer(IStepSizer):
    """
    As described in:
    https://link.springer.com/article/10.1007/s10994-006-8365-9
    """
    error_stepsize: float = 1  # (nu)
    error_stepsize_target: float = 0.001  # (nu-bar)
    step_size: Union[torch.Tensor, float] = 1.  # (a)
    avg: Union[torch.Tensor, float] = 0  # (theta)
    beta: Union[torch.Tensor, float] = 0.
    delta: Union[torch.Tensor, float] = 0.
    lambdaa: Union[torch.Tensor, float] = 0.
    epsilon: float = 0.
    first_iter: bool = True

    # def __call__(self, x):
    #     error = x-self.avg
    #     error_stepsize = self.error_stepsize / (1 + self.error_stepsize - self.error_stepsize_target)
    #     beta = (1-error_stepsize) * self.beta + error_stepsize * error
    #     delta = (1-error_stepsize) * self.delta + error_stepsize * error**2
    #     sigma_sq = (delta-beta**2)/(1+self.lambdaa)
    #     # step_size = 1 - sigma_sq / delta
    #     step_size = 1 - (sigma_sq+self.epsilon) / (delta+self.epsilon)
    #     lambdaa = (1-step_size)**2* self.lambdaa + step_size**2  # TODO: Test: Should it be (1-step_size**2) ??
    #     # lambdaa = (1-step_size**2)*self.lambdaa + step_size**2  # TODO: Test: Should it be (1-step_size**2) ??
    #     avg = (1-step_size) * self.avg + step_size * x
    #     new_obj = OptimalStepSizer(error_stepsize=error_stepsize, error_stepsize_target=self.error_stepsize_target,
    #         step_size=step_size, avg=avg, beta=beta, delta = delta, lambdaa=lambdaa, epsilon=self.epsilon)
    #
    #     if np.any(np.isnan(avg)):
    #         raise Exception()
    #     return new_obj, step_size

    def __call__(self, x):
        error = x-self.avg
        error_stepsize = self.error_stepsize / (1 + self.error_stepsize - self.error_stepsize_target)
        beta = (1-error_stepsize) * self.beta + error_stepsize * error
        delta = (1-error_stepsize) * self.delta + error_stepsize * error**2
        sigma_sq = (delta-beta**2)/(1+self.lambdaa)
        step_size = torch.tensor(1.) if self.first_iter else 1 - (sigma_sq+self.epsilon) / (delta+self.epsilon)
        # step_size = 1 - (sigma_sq+self.epsilon) / (delta+self.epsilon)
        lambdaa = (1-step_size)**2* self.lambdaa + step_size**2  # TODO: Test: Should it be (1-step_size**2) ??
        avg = (1-step_size) * self.avg + step_size * x
        new_obj = OptimalStepSizer(error_stepsize=error_stepsize, error_stepsize_target=self.error_stepsize_target,
            step_size=step_size, avg=avg, beta=beta, delta = delta, lambdaa=lambdaa, epsilon=self.epsilon, first_iter=False)

        if torch.any(torch.isnan(avg)):
            raise Exception()
        return new_obj, step_size


@dataclass
class PredictiveEncoder(ISimpleUpdateFunction):

    lambda_stepper: IStepSizer
    quantizer: ISimpleUpdateFunction
    last_input: torch.Tensor = 0.

    def __call__(self, x):
        new_lambda_state, lambdaa = self.lambda_stepper(x)
        prediction_error = x - (1-lambdaa)*self.last_input
        pre_code = prediction_error/lambdaa
        new_quantizer, q = self.quantizer(pre_code)
        new_encoder =PredictiveEncoder(lambda_stepper=new_lambda_state, quantizer=new_quantizer, last_input=x, )
        return new_encoder, q


@dataclass
class PredictiveDecoder(ISimpleUpdateFunction):

    lambda_stepper: IStepSizer
    last_reconstruction: torch.Tensor = 0.

    def __call__(self, x):
        lambdaa_stepper, lambdaa = self.lambda_stepper(x)
        reconstruction = (1-lambdaa)*self.last_reconstruction + lambdaa*x
        return PredictiveDecoder(lambda_stepper=lambdaa_stepper, last_reconstruction=reconstruction), reconstruction


@dataclass
class EncodingDecodingNeuronLayer(IDynamicLayer):

    encoder: ISimpleUpdateFunction
    decoder: ISimpleUpdateFunction
    stepper: IStepSizer

    @classmethod
    def get_partial_constructor(cls, encoder, decoder, stepper):
        def partial_constructor(n_samples: int, params: LayerParams):
            return EncodingDecodingNeuronLayer(
                params=params,
                output=torch.zeros((n_samples, params.n_units)),
                potential = torch.zeros((n_samples, params.n_units)),
                encoder=encoder,
                decoder=decoder,
                stepper=stepper
            )
        return partial_constructor

    @classmethod
    def get_simple_constructor(cls, epsilons, quantizer, lambdas=None):

        stepper = create_step_sizer(epsilons)

        quantizer = create_quantizer(quantizer)

        if lambdas is None:
            encoder = quantizer
            decoder = IdentityFunction()
        else:
            lambda_stepper = create_step_sizer(lambdas)
            encoder = PredictiveEncoder(lambda_stepper=lambda_stepper, quantizer=quantizer)
            decoder = PredictiveDecoder(lambda_stepper=lambda_stepper)

        return cls.get_partial_constructor(encoder=encoder, decoder=decoder, stepper=stepper)

    def __call__(self, x_aft=None, x_fore=None, pressure = None, clamp = None, gamma=1.0) -> 'EncodingDecodingNeuronLayer':

        # Compare to SimpleLayerController.__call__
        if clamp is not None:
            potential = clamp
            decoder = self.decoder
            stepper = self.stepper
        else:
            n_samples = x_aft.shape[0] if x_aft is not None else x_fore.shape[0]
            incoming_signal = torch.zeros((n_samples, self.params.n_units))
            if x_aft is not None:
                incoming_signal += x_aft @ self.params.w_aft
            if x_fore is not None:
                incoming_signal += gamma * x_fore @ self.params.w_fore
            decoder, input_pressure = self.decoder(incoming_signal)
            if self.params.b is not None:
                input_pressure = input_pressure + self.params.b

            total_pressure = drho(self.potential)*input_pressure
            stepper, eps = self.stepper(total_pressure)
            potential = torch.clamp((1-eps)*self.potential + eps * total_pressure, 0, 1)  # Euler integration with clipping to possible range
            
        post_potential = rho(potential)

        encoder, output = self.encoder(post_potential)

        return EncodingDecodingNeuronLayer(
            potential=potential,
            params=self.params,
            output = output,
            encoder=encoder,
            decoder=decoder,
            stepper=stepper
        )

    # def __call__(self, x_aft=None, x_fore=None, pressure = None, clamp = None) -> 'EncodingDecodingNeuronLayer':
    #
    #     if clamp is not None:
    #         potential = clamp
    #         decoder = self.decoder
    #         stepper = self.stepper
    #     else:
    #         n_samples = x_aft.shape[0] if x_aft is not None else x_fore.shape[0]
    #         u = np.zeros((n_samples, self.params.n_units))
    #         if x_aft is not None:
    #             u += x_aft @ self.params.w_aft
    #         if x_fore is not None:
    #             u += x_fore @ self.params.w_fore
    #         decoder, v = self.decoder(u)
    #         if pressure is not None:
    #             v += pressure
    #         if self.params.b is not None:
    #             v += self.params.b
    #         stepper, eps = self.stepper(u)
    #         potential = np.clip(self.potential - eps * drho(self.potential)* (self.potential - v), 0, 1)
    #     post_potential = rho(potential)
    #
    #     encoder, output = self.encoder(post_potential)
    #
    #     return EncodingDecodingNeuronLayer(
    #         potential=potential,
    #         params=self.params,
    #         output = output,
    #         encoder=encoder,
    #         decoder=decoder,
    #         stepper=stepper
    #     )
