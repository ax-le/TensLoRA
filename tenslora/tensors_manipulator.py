import math

import numpy as np
import tensorly as tl
import tltorch
import torch
from torch import nn


def get_list_of_available_tensor_methods():
    """
    Returns the list of all available tensor methods for TensLoRA.
    To be updated with developments.
    """
    return ["att", "qkv", "depth", "att_qkv", "att_depth", "qkv_depth", "att_qkv_depth"]


def get_list_of_available_tensor_factorizations():
    """
    Returns the list of all available tensor factorization methods for TensLoRA.
    To be updated with developments.
    """
    return ["tucker", "cp"]


class MotherTensorLoRA(nn.Module):
    """
    Mother class for all tensor LoRA adapters.
    Is used to regroup function common to all tensor LoRA adapters.
    This class should not be used directly, but inherited from.
    It could/should be made abstract in the future. TODO.
    """
    def __init__(self, dropout_prob=0):
        super(MotherTensorLoRA, self).__init__()
        self.dropout_prob = dropout_prob  # Default dropout probability, can be changed later

    def fold_tensor(self):
        """
        Folds the tltorch factorized tensor into a full tensor.
        """
        return self.tltorch_tensor.to_tensor()

    def get_update_matrix(self, qkv, layer):
        """
        Returns the update matrix from the tltorch tensor.
        """
        tensor = self.fold_tensor()
        return self.tensor_to_update_matrix(tensor, qkv=qkv, layer=layer)

    def tensor_to_update_matrix(self, tensor, qkv=None, layer=None):
        """
        Converts the tltorch tensor to an update matrix.
        This method is used to get the update matrix from the tltorch tensor.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses. It depends on the type of tensor LoRA being used.",
        )

    def make_tltorch_tensor(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses. It depends on the type of tensor factorization being used.",
        )

    def _check_dropout_applied(self):
        cond_empty_list = self.tltorch_tensor._forward_hooks == {}
        if cond_empty_list:
            return False
        cond_tensor_dropout = any(
            isinstance(hook, tltorch.tensor_hooks._tensor_dropout.TensorDropout)
            for hook in self.tltorch_tensor._forward_hooks.values()
        )
        return cond_tensor_dropout

    def add_dropout(self): # Add dropout to the tltorch tensor
        if not self._check_dropout_applied():  # Checking that dropout has not already been applied
            self.tltorch_tensor = tltorch.tensor_hooks.tensor_dropout(self.tltorch_tensor, p=self.dropout_prob, min_dim=4)

        assert self._check_dropout_applied(), (
            "Dropout has not been applied to the tltorch tensor, when it should have been. "
            "This is probably because the dropout from tltorch is not correctly called, check this."
        )

    def remove_dropout(self): # Remove dropout from the tltorch tensor
        if self._check_dropout_applied():
            self.tltorch_tensor = tltorch.tensor_hooks.remove_tensor_dropout(self.tltorch_tensor)

        assert not self._check_dropout_applied(), (
            "Dropout has been applied to the tltorch tensor, when it should not have been applied. "
            "This is probably because the dropout from tltorch is not correctly removed, check this."
        )


class TuckerLoRA(MotherTensorLoRA):
    """
    Standard class for Tucker TensLoRA adapters.
    Inherits from MotherTensorLoRA.
    Handles the Tucker Factorization, independently of the tensor method.
    This class should not be used directly, but inherited from.
    It could/should be made abstract in the future. TODO.
    """
    def __init__(self, number_modes, dimensions, n_components, dropout_prob=0, init="orthogonal", seed=0):
        MotherTensorLoRA.__init__(self, dropout_prob)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        if type(n_components) == int:
            n_components = [n_components for i in range(number_modes)]

        self._validate_args(number_modes, dimensions, n_components)

        self.n_components = n_components
        self.tucker_core = nn.Parameter(torch.zeros(n_components))
        self.tucker_factors = nn.ParameterList()

        if init == "normal": # Init from Tensorly
            r = np.prod([math.sqrt(r) for r in n_components])
            std_factors = (1 / r) ** (1 / (number_modes + 1))

        for i in range(number_modes):
            factor = nn.Parameter(torch.zeros(dimensions[i], n_components[i]))

            if init == "orthogonal":  # HOSVD like, seems to work well for Tucker
                nn.init.orthogonal_(factor)
            elif init == "kaiming_uniform":  # Kaiming uniform initialization, follows standard LoRA init
                nn.init.kaiming_uniform_(factor, a=math.sqrt(5))
            elif init == "normal":  # Init from Tensorly
                nn.init.normal_(factor, mean=0.0, std=std_factors)
            else:
                raise ValueError(
                    f"Unsupported initialization method: {init}. "
                    "Supported methods are 'orthogonal', 'kaiming_uniform', and 'normal'.",
                )

            self.tucker_factors.append(factor)

        self.tltorch_tensor = self.make_tltorch_tensor()  # Store the tltorch tensor for later use

    def make_tltorch_tensor(self):
        """Creates the tltorch tensor for the Tucker decomposition."""
        return tltorch.TuckerTensor(core=self.tucker_core, factors=self.tucker_factors)

    def _validate_args(self, number_modes, dimensions, n_components):
        # Number modes
        assert isinstance(number_modes, int), "Number of modes must be an integer"
        assert number_modes > 0, "Number of modes must be positive"

        # Dimensions
        assert number_modes == len(dimensions), "Number of modes must match the number of dimensions"

        # n_components
        assert len(n_components) == len(dimensions), "Number of n_components must match the number of dimensions"
        assert all(n_components > 0 for n_components in n_components), "n_components must be positive"
        assert all(isinstance(n_components, int) for n_components in n_components), "n_components must be integers"
        # assert (n_components <= dimensions).all(), "n_components must be less than or equal to dimensions"
        # # Do I really want this? It may be too restrictive, especially if one may want to compare LoRA r=8 with Tucker r=8


class CPLoRA(MotherTensorLoRA):
    """
    Standard class for CP TensLoRA adapters.
    Inherits from MotherTensorLoRA.
    Handles the CP Factorization, independently of the tensor method.
    This class should not be used directly, but inherited from.
    It could/should be made abstract in the future. TODO.
    """
    def __init__(self, number_modes, dimensions, n_components, dropout_prob=0, init="kaiming_uniform", seed=0):
        MotherTensorLoRA.__init__(self, dropout_prob)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self._validate_args(number_modes, dimensions, n_components)

        self.number_modes = number_modes
        self.n_components = n_components
        self.cp_factors = nn.ParameterList()
        self.init = init

        # Init from Tensorly
        std_factors = (1 / math.sqrt(self.n_components)) ** (1 / self.number_modes)

        for i in range(number_modes):
            factor = nn.Parameter(torch.zeros(dimensions[i], n_components))
            if i == 0:
                nn.init.zeros_(factor)
            elif init == "orthogonal":  # HOSVD like, makes sense for CP?
                nn.init.orthogonal_(factor)
            elif init == "kaiming_uniform":  # Kaiming uniform initialization, follows standard LoRA init
                nn.init.kaiming_uniform_(factor, a=math.sqrt(5))
            elif init == "normal":  # Init from Tensorly
                nn.init.normal_(factor, mean=0.0, std=std_factors)
            else:
                raise ValueError(
                    f"Unsupported initialization method: {init}. "
                    "Supported methods are 'orthogonal', 'normal', 'kaiming_uniform'.",
                )

            self.cp_factors.append(factor)

        self.tltorch_tensor = self.make_tltorch_tensor()  # Store the tltorch tensor for later use

    def make_tltorch_tensor(self):
        """Creates the tltorch tensor for the CP decomposition."""
        return tltorch.CPTensor(weights=torch.ones(self.n_components), factors=self.cp_factors)

    def _validate_args(self, number_modes, dimensions, n_components):
        # Number modes
        assert isinstance(number_modes, int), "Number of modes must be an integer"
        assert number_modes > 0, "Number of modes must be positive"

        # Dimensions
        assert number_modes == len(dimensions), "Number of modes must match the number of dimensions"

        # n_components
        assert n_components > 0, "n_components must be a positive integer"
        assert isinstance(n_components, int), "n_components must be an integer"


class AttTensorLoRA(TuckerLoRA, CPLoRA):
    """
    Standard class for Attention TensLoRA adapters. Used for the "att" tensor method.
    Inherits from MotherTensorLoRA, TuckerLoRA and CPLoRA.
    Handles both the Tucker and CP Factorizations, depending on the tensor_fac argument.
    """
    def __init__(self, tensor_fac, number_modes, dimensions, n_components, dropout_prob=0, init="orthogonal", seed=0):
        if tensor_fac == "tucker":
            TuckerLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        elif tensor_fac == "cp":
            CPLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        else:
            raise ValueError(f"Unsupported tensor factorization method: {tensor_fac}. Supported methods are 'tucker' and 'cp'.")

    def tensor_to_update_matrix(self, tensor, qkv, layer):
        """Converts the input tensor to the update matrix. In this setting, qkv and layer are not used."""
        # return tltorch.unfold(tensor, mode=0)
        return tl.unfold(tensor, mode=0)


class QKVTensorLoRA(TuckerLoRA, CPLoRA):
    """
    Standard class for QKV TensLoRA adapters. Used for the "qkv" tensor method.
    Inherits from MotherTensorLoRA, TuckerLoRA and CPLoRA.
    Handles both the Tucker and CP Factorizations, depending on the tensor_fac argument.
    """
    def __init__(self, tensor_fac, number_modes, dimensions, n_components, dropout_prob=0, init="orthogonal", seed=0):
        if tensor_fac == "tucker":
            TuckerLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        elif tensor_fac == "cp":
            CPLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        else:
            raise ValueError(f"Unsupported tensor factorization method: {tensor_fac}. Supported methods are 'tucker' and 'cp'.")

    def tensor_to_update_matrix(self, tensor, qkv, layer):
        """Converts the input tensor to the update matrix. In this setting, layer is not used."""
        # return tensor[:, :, qkv]
        return tensor.permute(2, 0, 1)[qkv] # Magic fix to avoid in large loss of computation time


class DepthTensorLoRA(TuckerLoRA, CPLoRA):
    """
    Standard class for Depth TensLoRA adapters. Used for the "depth" tensor method.
    Inherits from MotherTensorLoRA, TuckerLoRA and CPLoRA.
    Handles both the Tucker and CP Factorizations, depending on the tensor_fac argument.
    """
    def __init__(self, tensor_fac, number_modes, dimensions, n_components, dropout_prob=0, init="orthogonal", seed=0):
        if tensor_fac == "tucker":
            TuckerLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        elif tensor_fac == "cp":
            CPLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        else:
            raise ValueError(f"Unsupported tensor factorization method: {tensor_fac}. Supported methods are 'tucker' and 'cp'.")

    def tensor_to_update_matrix(self, tensor, qkv, layer):
        """Converts the input tensor to the update matrix. In this setting, qkv is not used."""
        # return tensor[:, :, layer]
        return tensor.permute(2, 0, 1)[layer] # Magic fix to avoid in large loss of computation time


class AttQKVTensorLoRA(TuckerLoRA, CPLoRA):
    """
    Standard class for AttQKV TensLoRA adapters. Used for the "att_qkv" tensor method.
    Inherits from MotherTensorLoRA, TuckerLoRA and CPLoRA.
    Handles both the Tucker and CP Factorizations, depending on the tensor_fac argument.
    """
    def __init__(self, tensor_fac, number_modes, dimensions, n_components, dropout_prob=0, init="orthogonal", seed=0):
        if tensor_fac == "tucker":
            TuckerLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        elif tensor_fac == "cp":
            CPLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        else:
            raise ValueError(f"Unsupported tensor factorization method: {tensor_fac}. Supported methods are 'tucker' and 'cp'.")

    def tensor_to_update_matrix(self, tensor, qkv, layer):
        """Converts the input tensor to the update matrix. In this setting, layer is not used."""
        # att_tensor_this_update = tensor[:, :, :, qkv]
        att_tensor_this_update = tensor.permute(3, 0, 1, 2)[qkv] # Magic fix to avoid in large loss of computation time
        return tl.unfold(att_tensor_this_update, mode=0)


class AttDepthTensorLoRA(TuckerLoRA, CPLoRA):
    """
    Standard class for AttDepth TensLoRA adapters. Used for the "att_depth" tensor method.
    Inherits from MotherTensorLoRA, TuckerLoRA and CPLoRA.
    Handles both the Tucker and CP Factorizations, depending on the tensor_fac argument.
    """
    def __init__(self, tensor_fac, number_modes, dimensions, n_components, dropout_prob=0, init="orthogonal", seed=0):
        if tensor_fac == "tucker":
            TuckerLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        elif tensor_fac == "cp":
            CPLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        else:
            raise ValueError(f"Unsupported tensor factorization method: {tensor_fac}. Supported methods are 'tucker' and 'cp'.")

    def tensor_to_update_matrix(self, tensor, qkv, layer):
        """Converts the input tensor to the update matrix. In this setting, qkv is not used."""
        # att_tensor_this_update = tensor[:, :, :, layer]
        att_tensor_this_update = tensor.permute(3, 0, 1, 2)[layer] # Magic fix to avoid in large loss of computation time
        return tl.unfold(att_tensor_this_update, mode=0)


class QKVDepthTensorLoRA(TuckerLoRA, CPLoRA):
    """
    Standard class for QKVDepth TensLoRA adapters. Used for the "qkv_depth" tensor method.
    Inherits from MotherTensorLoRA, TuckerLoRA and CPLoRA.
    Handles both the Tucker and CP Factorizations, depending on the tensor_fac argument.
    """
    def __init__(self, tensor_fac, number_modes, dimensions, n_components, dropout_prob=0, init="orthogonal", seed=0):
        if tensor_fac == "tucker":
            TuckerLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        elif tensor_fac == "cp":
            CPLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        else:
            raise ValueError("Unsupported tensor factorization method: tensor_fac. Supported methods are 'tucker' and 'cp'.")

    def tensor_to_update_matrix(self, tensor, qkv, layer):
        """Converts the input tensor to the update matrix. In this setting, both qkv and layer are used."""
        # return tensor[:, :, qkv, layer]
        return tensor.permute(2, 3, 0, 1)[qkv, layer] # Magic fix to avoid in large loss of computation time


class AttQKVDepthTensorLoRA(TuckerLoRA, CPLoRA):
    """
    Standard class for AttQKVDepth TensLoRA adapters. Used for the "att_qkv_depth" tensor method.
    Inherits from MotherTensorLoRA, TuckerLoRA and CPLoRA.
    Handles both the Tucker and CP Factorizations, depending on the tensor_fac argument.
    """
    def __init__(self, tensor_fac, number_modes, dimensions, n_components, dropout_prob=0, init="orthogonal", seed=0):
        if tensor_fac == "tucker":
            TuckerLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        elif tensor_fac == "cp":
            CPLoRA.__init__(self, number_modes, dimensions, n_components, dropout_prob, init=init, seed=seed)
        else:
            raise ValueError(f"Unsupported tensor factorization method: {tensor_fac}. Supported methods are 'tucker' and 'cp'.")

    def tensor_to_update_matrix(self, tensor, qkv, layer):
        """Converts the input tensor to the update matrix. In this setting, both qkv and layer are used."""
        # att_tensor_this_update = tensor[:, :, :, qkv, layer]
        att_tensor_this_update = tensor.permute(3, 4, 0, 1, 2)[qkv, layer] # Magic fix to avoid in large loss of computation time
        return tl.unfold(att_tensor_this_update, mode=0)
