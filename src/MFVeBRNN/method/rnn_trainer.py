# ------------------ Beginning of Reference Python Module ---------------------
""" This module contains the class RNNTrainer, which is a wrapper class for
training different RNN architectures for deterministic RNN training.

Class:
    RNNTrainer: Wrapper class for training different RNN architectures for
    deterministic RNN training.

"""
#
#                                                                       Modules
# =============================================================================
import copy
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'J.Yi@tudelft.nl'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================


class RNNTrainer:
    """RNN trainer, with supporting different RNN architectures."""

    def __init__(
        self,
        net: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
    ) -> None:
        """initialize the RNN with linear transfer layer and BNN for residual
        learning
        """
        # device
        self.device = device
        # set seed for all components
        self.seed = seed
        # define the network and move it to the device
        self.net = net.to(self.device)

    def configure_optimizer_info(
        self,
        optimizer_name: str = "Adam",
        lr: float = 1e-3,
        weight_decay: float = 1e-6
    ) -> None:
        """define optimizer of the low-fidelity deterministic RNN"""
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == "SGD":
            self.lf_optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError("Undefined optimizer")

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

    def train(
        self,
        x_train: Tensor,
        y_train: Tensor,
        num_epochs: int,
        batch_size: int = None,
        x_val: Tensor = None,
        y_val: Tensor = None,
        verbose: bool = True,
        print_iter: int = 100,
    ) -> Tuple[float, int]:
        """train the network

        Parameters
        ----------
        x_train : Tensor
            input training data
        y_train : Tensor
            output training data
        num_epochs : int
            number of epochs for training
        batch_size : int, optional
            batch size for mini-batch training, by default None
        x_val : Tensor, optional
            input validation data, by default None
        y_val : Tensor, optional
            output validation data, by default None
        verbose : bool, optional
            print the training information, by default True
        print_iter : int, optional
            print the training information every certain epochs, by default 100
        save_best_model : bool, optional
            save the best model during training, by default False
        name : str, optional
            name of the best model, by default "best_val_net.pth"

        Returns
        -------
        Tuple[float, int]
            minimum loss value and the epoch number
        """
        # move the training and validation data to the device
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        if x_val is not None:
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)
        # record the minimum loss of the validation data
        min_loss = np.inf
        # loader for mini-batch
        if batch_size is None:
            self.batch_size = x_train.shape[0]
            self.num_scale = 1.0
        else:
            self.batch_size = batch_size
            self.num_scale = x_train.shape[0] / self.batch_size

        loader = DataLoader(
            dataset=list(zip(x_train, y_train)),
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
                # back-propagation
                loss.backward()
                # update the weights
                self.optimizer.step()
                # accumulate the loss value
                running_loss_train += loss.item() * X_batch.size(0)

            # average the loss value
            loss_train = running_loss_train / x_train.size(0)
            if x_val is not None:
                self.net.eval()
                y_val_pred = self.net.forward(x_val)
                loss_val = self.loss_function(y_val, y_val_pred)
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

    def predict(self, x: Tensor) -> Tensor:
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
        self.best_net.eval()
        y = self.best_net.forward(x.to(self.device))

        return y.detach()

    def recurrent_forward(self,
                          x: torch.Tensor,
                          hx: torch.Tensor = None) -> List:
        """forward of low-fidelity RNN, such that hidden states are returned.
        Then, the hidden states are used as the inputs for the transfer layer
        and also used for residual BNN training

        Parameters
        ----------
        x : torch.Tensor
            input data (scaled low-fidelity data or high-fidelity data in
            low-fidelity scale)
        hx : torch.Tensor, optional
            hidden state, by default None

        Returns
        -------
        List
            hidden states of the RNN at each time step
        """

        self.best_net.eval()
        with torch.no_grad():
            outs, _ = self.best_net.gru(x.to(self.device), hx)

        return outs

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
