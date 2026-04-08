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
from torch.utils.data import DataLoader
import numpy as np
import copy
from MFVeBRNN.method.rnn_trainer import RNNTrainer
from BayesMFM.method.sf_cuq_rnn_trainer import CUQRNNBayesTrainer as LFRNNBayes

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================



class MFResidualRNNTrainer:

    def __init__(
        self,
        net: torch.nn.Module,
        pre_trained_lf_model: RNNTrainer | LFRNNBayes ,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        nest_option: str = "hidden",
    ) -> None:
        """initialize the trainer for the high-fidelity model

        Parameters
        ----------
        net : torch.nn.Module
            _description_
        dataset : MFDeterDataset
            _description_
        pre_trained_lf_model : RNNTrainer | LFRNNBayes
            _description_
        device : torch.device, optional
            _description_, by default torch.device("cpu")
        seed : int, optional
            _description_, by default 0
        nest_option : str, optional
            nested option, by default "hidden" or "output"
        """
        self.device = device
        # load the net to the device
        self.net = net.to(self.device)
        # load the pre-trained low-fidelity model to the device
        self.lf_model: RNNTrainer | LFRNNBayes  = pre_trained_lf_model

        # load to the device
        self.lf_model.device = self.device
        if isinstance(self.lf_model, RNNTrainer ):
            self.lf_model.best_net = self.lf_model.best_net.to(self.device)
        elif isinstance(self.lf_model, LFRNNBayes):
            self.lf_model.mean_net = self.lf_model.mean_net.to(self.device)
            self.lf_model.var_net = self.lf_model.var_net.to(self.device)

        # set the seed and nest option
        self.seed = seed
        self.nest_option = nest_option

    def configure_optimizer_info(self,
                                 optimizer_name: str = "Adam",
                                 lr: float = 0.001,
                                 weight_decay: float = 0.0):

        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            self.optimizer = torch.optim.SGD(
                self.net.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("Optimizer not supported")

    def configure_loss_function(self, loss_name: str = "MSE") -> None:
        """configure the loss function for the training process

        Parameters
        ----------
        loss_name : str, optional
            name of the loss function, by default "MSE"

        Raises
        ------
        ValueError
            Undefined loss function
        """
        if loss_name == "MSE":
            self.loss_function = torch.nn.MSELoss()
        elif loss_name == "MAE":
            self.loss_function = torch.nn.L1Loss()
        else:
            raise ValueError("Undefined loss function")

    def train(self,
              hx_train: Tensor,
              hy_train: Tensor,
              num_epochs: int,
              batch_size: int = None,
              hx_val: Tensor = None,
              hy_val: Tensor = None,
              verbose: bool = True,
              print_iter: int = 100,) -> None:

        # set the data to the device
        hx_train = hx_train.to(self.device)
        hy_train = hy_train.to(self.device)
        if hx_val is not None:
            hx_val = hx_val.to(self.device)
            hy_val = hy_val.to(self.device)

        # re-arrange the training and validation data for output
        hy_train = self._calculate_residual(hx_train, hy_train)
        if hx_val is not None:
            hy_val = self._calculate_residual(hx_val, hy_val)

        # re-arrange the training and validation data
        hx_train = self._re_arrange_input(hx_train)
        if hx_val is not None:
            hx_val = self._re_arrange_input(hx_val)
        min_loss = np.Inf
        # loader for mini-batch
        if batch_size is None:
            self.batch_size = hx_train.shape[0]
            self.num_scale = 1.0
        else:
            self.batch_size = batch_size
            self.num_scale = hx_train.shape[0] / self.batch_size

        loader = DataLoader(
            dataset=list(zip(hx_train, hy_train)),
            batch_size=self.batch_size,
            shuffle=True
        )

        # begin the training process
        for epoch in range(num_epochs):

            # set the network to training mode
            self.net.train()
            # running loss for the training data
            running_loss_train = 0
            for X_batch, y_batch in loader:
                # set gradient of params to zero
                self.optimizer.zero_grad()
                # get prediction from network
                pred = self.net.forward(X_batch)
                # calculate the loss value for the batch
                loss = self.loss_function(pred, y_batch)
                # back propagation
                loss.backward()
                # update the weights
                self.optimizer.step()
                # accumulate the loss value
                running_loss_train += loss.item() * X_batch.size(0)

            # average the loss value
            loss_train = running_loss_train / hx_train.size(0)
            if hx_val is not None:
                self.net.eval()
                y_val_pred = self.net.forward(hx_val)
                loss_val = self.loss_function(hy_val, y_val_pred)
                if loss_val.item() < min_loss:
                    # save the model with the best validation loss
                    min_loss = loss_val.item()
                    best_epoch = epoch
                    self.best_net = copy.deepcopy(self.net)

            else:
                loss_val = torch.tensor(0.0)
                self.best_net = copy.deepcopy(self.net)
                min_loss = loss_val.item()
                best_epoch = epoch

            if verbose and epoch % print_iter == 0:
                self._print(epoch, num_epochs,
                            loss_train, loss_val.item())

        return min_loss, best_epoch

    def hf_predict(self, x: Tensor) -> Tensor:
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
        x = x.to(self.device)
        y_lf = self.lf_predict(x)
        # get the re-arranged input data
        x = self._re_arrange_input(x)
        self.best_net.eval()
        diff = self.best_net.forward(x)

        y = y_lf + diff

        return y.detach()

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

        elif isinstance(self.lf_model, LFRNNBayes):
            y, var_epistemic = self.lf_model.bayes_predict(
                x.to(self.device))
            var_aleatoric = self.lf_model.aleatoric_variance_predict(
                x.to(self.device))
            if return_var:
                return y, var_aleatoric, var_epistemic
        else:
            raise ValueError("Undefined low-fidelity model")

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

        if self.nest_option == "original":

            return x

        elif self.nest_option == "hidden":

            # predict the output of the low-fidelity model
            re_hx_input = self.lf_model.recurrent_forward(x)
            # re_hx_input = torch.stack(re_hx_input, dim=1)
            re_hx_input = re_hx_input.detach()
            # concatenate the data
            x = torch.cat((x, re_hx_input), dim=-1)

            return x

        else:
            raise ValueError("Undefined nest option")

    def _calculate_residual(self,
                            x: Tensor,
                            y: Tensor) -> Tensor:
        """calculate residual between the predicted output and the true output

          Parameters
          ----------
          x : Tensor
                input data
          y : Tensor
                true output data

          Returns
          -------
          Tensor
                residual between the predicted output and the true output
          """

        y_pred = self.lf_predict(x)
        residual = y - y_pred

        return residual.detach()

    def _print(
        self,
        epoch: int,
        num_epoch: int,
        loss_train: float,
        loss_val: float
    ) -> None:
        """print the loss values during training at certain epochs

            Parameters
            ----------
            epoch : int
                the current epoch
            num_epoch : int
                total number of epochs
            loss_train : float
                training loss value at the current epoch
            loss_val : float
                validation loss value at the current epoch
            """

        print(
            "Epoch/Total: %d/%d, Train Loss: %.3e, Val Loss: %.3e"
            % (
                epoch,
                num_epoch,
                loss_train,
                loss_val,
            )
        )
