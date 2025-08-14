import torch.nn as nn
import numpy as np
from abc import abstractmethod


class ModelBase(nn.Module):
    """
        Base class for all models
        """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = list(filter(lambda p: p.requires_grad, self.parameters()))
        # we need to wrap filter by a list, because filter function return an iteration
        # once somewhere consumes this iteration then it won't be reused, because it will be empty
        # for p in model_parameters:
        #     print(p.size())
        # self.parameters() does not return every single weight and bias unit,
        # instead it returns a tensor like (a, b, c)
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
