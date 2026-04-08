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
        super().__init__(
            mean_net=mean_net,
            var_net=var_net,
            device=device,
            job_id=job_id,
        )

    def recurrent_forward(self,
                          x: torch.Tensor,
                          return_var: bool = False) -> torch.Tensor:
        """forward of low-fidelity RNN, such that hidden states are returned"""

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
