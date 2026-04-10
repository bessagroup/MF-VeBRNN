# ------------------ Beginning of Reference Python Module ---------------------
""" This module contains the class VeBRNNTrainer, which is a wrapper class for
training different RNN architectures with Variance estimation Bayesian neural
networks.

Class:
    VeBRNNTrainer: Wrapper class for training different RNN architectures for
    deterministic RNN training.

"""
#
#                                                                       Modules
# =============================================================================

from VeBNN.methods import SGMCMCTrainer
from VeBNN.networks import MeanNet, GammaVarNet

import torch
import copy


class VeBRNNTrainer(SGMCMCTrainer):
    """VeBRNN trainer, with supporting different RNN architectures."""

    def __init__(
        self,
        mean_net: MeanNet,
        var_net: GammaVarNet,
        device: torch.device = torch.device("cpu"),
        job_id: int = 1,
    ) -> None:
        """Variance estimation Bayesian neural network training using
        Stochastic

        Parameters
        ----------
        mean_net : MeanNet
            mean neural network for the VeBNN
        var_net : GammaVarNet
            variance estimation neural network for VeBNN
        device : torch.device, optional
            device for training, by default torch.device("cpu")
        job_id : int, optional
            job ID for the training, by default 1
        """
        super().__init__(
            mean_net=mean_net,
            var_net=var_net,
            device=device,
            job_id=job_id,
        )

    def recurrent_forward(self,
                          x: torch.Tensor,
                          return_var: bool = False) -> torch.Tensor:
        """forward of low-fidelity RNN, such that hidden states are returned

        Parameters
        ----------
        x : torch.Tensor
            input data for the forward pass
        return_var : bool, optional
            whether to return the variance estimation, by default False

        Returns
        -------
        torch.Tensor
            output data for the forward pass
        """

        hidden_states = []
        for state_dict in self.mean_nets:
            # Create a fresh model instance
            temp_model = copy.deepcopy(self.un_trained_mean_net)
            temp_model.load_state_dict(state_dict)
            with torch.no_grad():
                # get the hidden state from the first
                hidden_state, _ = list(temp_model.net.children())[0](x)
            hidden_states.append(hidden_state)
        hidden_states = torch.stack(hidden_states, dim=1)
        if return_var:
            hidden_states_mean = torch.mean(hidden_states, dim=1)
            hidden_states_var = torch.var(hidden_states, dim=1)
            return hidden_states_mean.detach(), hidden_states_var.detach()
        else:
            return torch.mean(hidden_states, dim=1)
