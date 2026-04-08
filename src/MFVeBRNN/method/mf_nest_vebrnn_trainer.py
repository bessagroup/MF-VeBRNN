# ------------------ Beginning of Reference Python Module ---------------------
""" This module contains the class MFResidualRNNTrainer, which is a wrapper
for multi-fidelity recurrent neural networks deterministically with a linear
transfer learning and non-linear residual network.
Class:
    MFResidualRNNTrainer: A wrapper class for multi-fidelity recurrent neural
    networks deterministically with a linear transfer learning and non-linear
    residual network.

"""
#
#                                                                       Modules
# =============================================================================

import torch
from torch import Tensor

from typing import Tuple
from VeBNN.networks.mean_nets import MeanNet
from VeBNN.networks.variance_nets import GammaVarNet
from MFVeBRNN.method.rnn_trainer import RNNTrainer
from MFVeBRNN.method.vebnn_trainer import VeBRNNTrainer
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class MFNestVeBRNNTrainer:

    def __init__(self,
                 mean_net: MeanNet,
                 var_net: GammaVarNet,
                 pre_trained_lf_model: RNNTrainer|VeBRNNTrainer,
                 device: torch.device = torch.device("cpu"),
                 job_id: int = 0,
                 nest_option: str = "hidden",
                 ) -> None:
        """initialize the trainer for the high-fidelity model
        """
        # define the device
        self.device = device
        # load the pre-trained low-fidelity model
        self.lf_model: RNNTrainer | VeBRNNTrainer  = pre_trained_lf_model
        # load the pre-trained low-fidelity model to the device
        self.lf_model.device = self.device
        if isinstance(self.lf_model, RNNTrainer):
            self.lf_model.best_net = self.lf_model.best_net.to(self.device)
        elif isinstance(self.lf_model, VeBRNNTrainer):
            self.lf_model.mean_net = self.lf_model.mean_net.to(self.device)
            self.lf_model.var_net = self.lf_model.var_net.to(self.device)

        # get the mean and variance network architecture
        self.un_trained_mean_net = mean_net.to(self.device)
        self.un_trained_var_net = var_net.to(self.device)
        # seed of the model
        self.job_id = job_id
        # the nested option
        self.nest_option = nest_option

        # init the high-fidelity trainer
        self.hf_vebrnn_trainer = self._init_hf_trainer()


    def cooperative_train(self,
                          x_train: Tensor,
                          y_train: Tensor,
                          iteration: int,
                          init_config = {
                                    "loss_name": "MSE",
                                    "optimizer_name": "Adam",
                                    "lr": 1e-3,
                                    "weight_decay": 1e-6,
                                    "num_epochs": 1000,
                                    "batch_size": 200,
                                    "verbose": False,
                                    "print_iter": 50,
                                    "split_ratio": 0.8,
                                },
                        var_config = {
                            "optimizer_name": "Adam",
                            "lr": 1e-3,
                            "num_epochs": 1000,
                            "batch_size": 200,
                            "verbose": True,
                            "print_iter": 50,
                            "early_stopping": False,
                            "early_stopping_iter": 100,
                            "early_stopping_tol": 1e-4,
                        },
                        sampler_config = {
                            "sampler": "pSGLD",
                            "lr": 1e-3,
                            "gamma": 0.9999,
                            "num_epochs": 2000,    # SGMCMC epochs
                            "mix_epochs": 10,     # thinning interval
                            "burn_in_epochs": 500,
                            "batch_size": 200,
                            "verbose": False,
                            "print_iter": 100,
                        },
                        delete_model_raw_data=True,
                          ) -> None:
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)

            # re-arrange the input data
            x_train = self._re_arrange_input(x_train)

            # train the residual model with the VeBNN trainer
            self.hf_vebrnn_trainer.cooperative_train(
                x_train=x_train,
                y_train=y_train,
                iteration=iteration,
                init_config=init_config,
                var_config=var_config,
                sampler_config=sampler_config,
                delete_model_raw_data=delete_model_raw_data,
            )

    def hf_bayes_predict(
        self,
        x: torch.Tensor,
        save_ppd: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Predict the mean and variance of the output at the scaled data.

        Parameters
        ----------
        X : torch.Tensor
            Test data points.
        save_ppd : bool, optional
            Whether to save ppd or not (default is False).

        Returns
        -------
        Tuple[Tensor, Tensor]
            Predicted mean and variance at the scaled space.
        """
        x = x.to(self.device)

        x = self._re_arrange_input(x)

        # get the prediction from the residual model
        y_pred_mean, y_pred_var = self.hf_vebrnn_trainer.bayes_predict(x,save_ppd=save_ppd)

        if save_ppd:
            self.responses = self.hf_vebrnn_trainer.responses

        return y_pred_mean.detach(), y_pred_var.detach()

    def lf_predict(self, x: Tensor, return_var: bool = False) -> Tensor:
        """predict the output of the network

        Parameters
        ----------
        x : Tensor
            input data

        Returns
        -------
        Tensor
            predicted output data
        """

        if isinstance(self.lf_model, RNNTrainer):
            y = self.lf_model.predict(x.to(self.device))
        elif isinstance(self.lf_model, VeBRNNTrainer):
            y, var_epistemic = self.lf_model.bayes_predict(
                x.to(self.device))
            var_aleatoric = self.lf_model.aleatoric_variance_predict(
                x.to(self.device))
            if return_var:
                return y, var_aleatoric, var_epistemic

        return y

    def _re_arrange_input(self,
                          x: Tensor) -> Tensor:
        """re-arrange the input data for the training process

        Parameters
        ----------
        x : Tensor
            input data for the training process or prediction

        Returns
        -------
        Tensor
            the re-arranged input data

        Raises
        ------
        ValueError
            Undefined nest option
        """

        if self.nest_option == "output":

            # predict the output of the low-fidelity model
            re_hx_input = self.lf_model.predict(x)
            # concatenate the data
            x = torch.cat((x, re_hx_input), dim=2)

            return x

        elif self.nest_option == "hidden":

            # predict the output of the low-fidelity model
            re_hx_input = self.lf_model.recurrent_forward(x)
            re_hx_input = re_hx_input.detach()
            # concatenate the data
            x = torch.cat((x, re_hx_input), dim=-1)

            return x

        else:
            raise ValueError("Undefined nest option")


    def _init_hf_trainer(self) -> VeBRNNTrainer:
        vebrnn_trainer = VeBRNNTrainer(
            mean_net=self.un_trained_mean_net,
            var_net=self.un_trained_var_net,
            device=self.device,
            job_id=self.job_id,
        )
        return vebrnn_trainer
