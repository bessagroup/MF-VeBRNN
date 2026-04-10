"""

This package contains the implementation of the multi-fidelity recurrent neural
networks with a linear transfer learning and non-linear residual network,
trained with VeBNN.

Author: Jiaxiang Yi (J.Yi@tudelft.nl)
"""

#                                                                       Modules
# =============================================================================


#                                                        Authorship and Credits
# =============================================================================
__author__ = 'Jiaxiang Yi (J.Yi@tudelft.nl)'
__credits__ = ['Jiaxiang Yi']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

from .dataset.load_dataset import SingleFidelityDataset, MultiFidelityDataset
from .method.rnn_trainer import RNNTrainer
from .method.vebnn_trainer import VeBRNNTrainer
from .method.mf_nest_rnn_trainer import MFNestRNNTrainer
from .method.mf_residual_rnn_trainer import MFResidualRNNTrainer
from .method.mf_nest_vebrnn_trainer import MFNestVeBRNNTrainer
from .method.mf_residual_vebrnn_trainer import MFResidualVeBRNNTrainer

