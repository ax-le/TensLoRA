import numpy as np
from torch import nn

import tenslora.tensors.tensors_manipulator as tensors_manipulator

# The adapter module for TensLoRA.
# It may be too complex, TODO: simplify the hierarchy of functions.
class TensLoRA_adapter(nn.Module):
    """
    The TensLoRA adapter module.
    It contains the tensor decomposition and the logic to apply it to the input tensor.
    
    Args:
        input_dim (int): The input dimension of the adapter.
        output_dim (int): The output dimension of the adapter.
        n_components (int or list of int): The number of components for the tensor decomposition.
            - If int, it applies the same number of components to all modes.
            - If list of int, it applies the specified number of components to each mode.
            The length of the list must be equal to the number of modes of the tensor.
        tensor_method (str): The tensor method to use.
            - Options are "att", "qkv", "depth", "att_qkv", "att_depth", "qkv_depth", and "att_qkv_depth".
            - See tensors_manipulator for more details.
        tensor_fac (str): The tensor factorization to use.
            - Options are "tucker" and "cp".
        dropout_prob (float, optional): The dropout probability for the adapter. Default is 0.1.
        tensor_init (str, optional): The initialization method for the tensor decomposition.
            - Options are "orthogonal", "normal", and "uniform". Default is "orthogonal".
        scaling (float, optional): The scaling factor for the adapter. Default is 1.
        seed (int, optional): The random seed for the initialization. Default is 0.
        *args, **kwargs: Additional arguments for the tensor decomposition.

    """
    def __init__(
        self,
        input_dim,
        output_dim,
        n_components,
        tensor_method,
        tensor_fac,
        dropout_prob=0.1,
        tensor_init="orthogonal",
        scaling=1,
        seed=0,
        *args,
        **kwargs,
    ):
        # The code implicitly assumes that input_dim == output_dim
        # This is not a problem for the moment, but it may be in the future
        super(TensLoRA_adapter, self).__init__()

        self._validate_tensor_method(tensor_method, input_dim, output_dim, n_components, *args, **kwargs)

        # Tensor parameters
        self.tensor_fac = tensor_fac
        self.tensor_method = tensor_method
        self.tensor_init = tensor_init
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_components = n_components
        self.dropout_prob = dropout_prob
        self.scaling = scaling
        self.seed = seed

        # Actually construct the tensors
        self.tensor_lora_module = self._create_actual_tensor(
            tensor_method,
            tensor_fac,
            input_dim,
            output_dim,
            n_components,
            tensor_init,
            seed=seed,
            *args,
            **kwargs,
        )

    # Define the forward pass for the TensLoRA adapter.
    def forward(self, x, qkv=None, layer=None):
        self._validate_forward_tenslora(None, qkv, layer)

        # Transforms the tensor into a matrix specialized for the current input.
        # Hence, it depends on the tensor_method, the qkv index, and the layer index.
        # The function get_update_matrix is one of the core functions of the TensLoRA adapter.
        matrix_this_update = self.get_update_matrix(qkv=qkv, layer=layer).to(x.device, dtype=x.dtype)

        return x @ (matrix_this_update * self.scaling)  # Assumes that both matrices are square

    # Encapsulation of get_update_matrix, with dropout handling and a test.
    def get_update_matrix(self, qkv, layer):
        """
        Returns the update matrix for the current tensor.
        Is the core function of the TensLoRA adapter, because it returns the matrix that will be used to update the input tensor.
        """
        if self.training and self.dropout_prob > 0:
            self.tensor_lora_module.add_dropout()
        else:
            self.tensor_lora_module.remove_dropout()

        matrix_this_update = self.tensor_lora_module.get_update_matrix(qkv=qkv, layer=layer)

        self._validate_matrix_this_update(matrix_this_update)

        return matrix_this_update

    # Actually creates the tensor parameters.
    def _create_actual_tensor(
        self,
        tensor_method,
        tensor_fac,
        input_dim,
        output_dim,
        n_components,
        tensor_init,
        seed=0,
        *args,
        **kwargs,
    ):
        """
        Creates the actual tensor based on the specified method and factorization.
        
        Ideally, this function should be added inside the specialized TensorLoRA classes,
        for easier maintenance (in particular adding a new tensor method),
        but I couldn't think of a appropriate design pattern to do so.
        So may be TODO.

        Args:
            tensor_method (str): The tensor method to use.
                - Options are "att", "qkv", "depth", "att_qkv", "att_depth", "qkv_depth", and "att_qkv_depth".
                - See tensors_manipulator for more details.
            tensor_fac (str): The tensor factorization to use.
                - Options are "tucker" and "cp".
            input_dim (int): The input dimension of the adapter.
            output_dim (int): The output dimension of the adapter.
            n_components (int or list of int): The number of components for the tensor decomposition.
                - If int, it applies the same number of components to all modes.
                - If list of int, it applies the specified number of components to each mode.
                The length of the list must be equal to the number of modes of the tensor.
            tensor_init (str): The initialization method for the tensor decomposition.
                - Options are "orthogonal", "normal", and "uniform".
            seed (int): The random seed for the initialization.
            *args, **kwargs: Additional arguments for the tensor decomposition.
        """
        match tensor_method: # The way to create the tensor depends on the method.
            case "att":
                self.number_heads = kwargs["number_heads"]
                return tensors_manipulator.AttTensorLoRA(
                    tensor_fac=tensor_fac,
                    number_modes=3,
                    dimensions=[input_dim, int(output_dim / self.number_heads), self.number_heads],
                    n_components=n_components,
                    init=tensor_init,
                    seed=seed,
                )

            case "qkv":
                return tensors_manipulator.QKVTensorLoRA(
                    tensor_fac=tensor_fac,
                    number_modes=3,
                    dimensions=[input_dim, output_dim, 3],
                    n_components=n_components,
                    init=tensor_init,
                    seed=seed,
                )

            case "depth":
                self.number_attention_layers = kwargs["number_attention_layers"]
                return tensors_manipulator.DepthTensorLoRA(
                    tensor_fac=tensor_fac,
                    number_modes=3,
                    dimensions=[input_dim, output_dim, self.number_attention_layers],
                    n_components=n_components,
                    init=tensor_init,
                    seed=seed,
                )

            case "att_qkv":
                self.number_heads = kwargs["number_heads"]
                return tensors_manipulator.AttQKVTensorLoRA(
                    tensor_fac=tensor_fac,
                    number_modes=4,
                    dimensions=[input_dim, int(output_dim / self.number_heads), self.number_heads, 3],
                    n_components=n_components,
                    init=tensor_init,
                    seed=seed,
                )

            case "att_depth":
                self.number_heads = kwargs["number_heads"]
                self.number_attention_layers = kwargs["number_attention_layers"]
                return tensors_manipulator.AttDepthTensorLoRA(
                    tensor_fac=tensor_fac,
                    number_modes=4,
                    dimensions=[input_dim, int(output_dim / self.number_heads), self.number_heads, self.number_attention_layers],
                    n_components=n_components,
                    init=tensor_init,
                    seed=seed,
                )

            case "qkv_depth":
                self.number_attention_layers = kwargs["number_attention_layers"]
                return tensors_manipulator.QKVDepthTensorLoRA(
                    tensor_fac=tensor_fac,
                    number_modes=4,
                    dimensions=[input_dim, output_dim, 3, self.number_attention_layers],
                    n_components=n_components,
                    init=tensor_init,
                    seed=seed,
                )

            case "att_qkv_depth":
                self.number_heads = kwargs["number_heads"]
                self.number_attention_layers = kwargs["number_attention_layers"]
                return tensors_manipulator.AttQKVDepthTensorLoRA(
                    tensor_fac=tensor_fac,
                    number_modes=5,
                    dimensions=[
                        input_dim,
                        int(output_dim / self.number_heads),
                        self.number_heads,
                        3,
                        self.number_attention_layers,
                    ],
                    n_components=n_components,
                    init=tensor_init,
                    seed=seed,
                )

        # Other cases can be added here as needed. The unknown case will raise an error in "_validate_tensor_method".
        #

    def _validate_tensor_method(self, tensor_method, input_dim, output_dim, n_components, *args, **kwargs):
        """
        A checker for the tensor method and its parameters.
        May complicate maintenance, and increase execution time, but it is useful to catch errors early.
        """
        def _validate_att_parameters():
            assert kwargs.get("number_heads") is not None, (
                "The number of attention heads (number_heads) must be provided for 'att' method"
            )
            num_heads = kwargs.get("number_heads")
            assert num_heads > 0, "The number of attention heads (number_heads) must be positive"
            assert isinstance(num_heads, int), "The number of attention heads (number_heads) must be an integer"
            assert output_dim % num_heads == 0, (
                "The output dimension (2nd dimension of the matrix) must be divisible by the number of attention heads"
            )

        def _validate_qkv_parameters():
            pass

        def _validate_depth_parameters():
            assert kwargs.get("number_attention_layers") is not None, (
                "The number of attention layers must be provided for 'depth' method"
            )
            number_attention_layers = kwargs.get("number_attention_layers")
            assert number_attention_layers > 0, "number_attention_layers must be positive"
            assert isinstance(number_attention_layers, int), "number_attention_layers must be an integer"

        match tensor_method:
            case "att":
                _validate_att_parameters()

            case "qkv":
                _validate_qkv_parameters()

            case "depth":
                _validate_depth_parameters()

            case "att_qkv":
                # Att
                _validate_att_parameters()
                # QKV
                _validate_qkv_parameters()

            case "att_depth":
                # Att
                _validate_att_parameters()
                # Depth
                _validate_depth_parameters()

            case "qkv_depth":
                # QKV
                _validate_qkv_parameters()
                # Depth
                _validate_depth_parameters()

            case "att_qkv_depth":
                # Att
                _validate_att_parameters()
                # QKV
                _validate_qkv_parameters()
                # Depth
                _validate_depth_parameters()

            case _:
                raise ValueError(
                    f"Unsupported tensor method: {tensor_method}",
                )  # Should have been triggered by the _validate_tenslora_method function

    def _validate_forward_tenslora(self, x, qkv, layer):
        # Check if the input tensor has the correct shape
        ## Removed as it complicates the use of the adapter in some cases, and did not seem particularly useful.
        # assert x.ndim == 2, "Input tensor must be 2D" 
        # assert x.shape[-2] == self.input_dim, f"Input tensor's first dimension must be {self.input_dim}, but got {x.shape[-2]}"
        # assert x.shape[-1] == self.output_dim, f"Input tensor's second dimension must be {self.output_dim}, but got {x.shape[-1]}"

        # Check if qkv and layer are provided when necessary
        if self.tensor_method in ["qkv", "att_qkv", "qkv_depth", "att_qkv_depth"]:
            assert qkv is not None, "qkv must be provided for this tensor method"
            assert isinstance(qkv, int), "qkv must be an integer"
            assert 0 <= qkv < 3, "qkv must be between 0 and 2 (inclusive) for this tensor method"

        if self.tensor_method in ["depth", "att_depth", "qkv_depth", "att_qkv_depth"]:
            assert layer is not None, "layer must be provided for this tensor method"
            assert isinstance(layer, int), "layer must be an integer"
            assert 0 <= layer < self.number_attention_layers, (
                "layer must be between 0 and number_attention_layers (exclusive) for this tensor method"
            )

    def _validate_matrix_this_update(self, matrix_this_update):
        # Check if the matrix_this_update tensor has the correct shape
        # The code implicitly assumes that input_dim == output_dim
        # This is not a problem for the moment, but it may be in the future
        # assert matrix_this_update.ndim == 2, "matrix_this_update must be 2D" # Doesn't work with batches.
        assert matrix_this_update.shape[-2] == self.input_dim, (
            f"matrix_this_update's first dimension must be {self.input_dim}, but got {matrix_this_update.shape[-2]}"
        )
        assert matrix_this_update.shape[-1] == self.output_dim, (
            f"matrix_this_update's second dimension must be {self.output_dim}, but got {matrix_this_update.shape[-1]}"
        )

def select_tensor_among_adapters(tensors, tensor_method, qkv, layer):
    """
    Selects the appropriate tensor adapter based on the tensor method, qkv index, and layer.

    Ideally, this function should be added inside the specialized TensorLoRA classes,
    for easier maintenance (in particular adding a new tensor method),
    but I couldn't think of a appropriate design pattern to do so.
    So may be TODO.

    Args:
        tensors: The tensor adapters created by create_all_tensor_adapters. Is already specialized for the tensor method.
        tensor_method (str): The tensor method to use.
            - Options are "att", "qkv", "depth", "att_qkv", "att_depth", "qkv_depth", and "att_qkv_depth".
            - See tensors_manipulator for more details.
        qkv (int): An index for the query, key, and value tensors for TensLoRA.
            - query corresponds to 0, key to 1, and value to 2.
        layer (int): The layer index for the tensor. Should be between 0 and number_attention_layers - 1.
    Returns:
        The appropriate tensor adapter.
    """
    match tensor_method:
        case "att":
            return tensors[qkv][layer]
        case "qkv":
            return tensors[layer]
        case "depth":
            return tensors[qkv]
        case "att_qkv":
            return tensors[layer]
        case "att_depth":
            return tensors[qkv]
        case "qkv_depth":
            return tensors
        case "att_qkv_depth":
            return tensors


def create_all_tensor_adapters(
    input_dim,
    output_dim,
    number_attention_layers,
    n_components,
    tensor_method,
    tensor_fac,
    dropout_prob=0.1,
    tensor_init="orthogonal",
    scaling=1,
    seed=0,
    *args,
    **kwargs,
):
    """
    Creates all tensor adapters based on the specified tensor method and factorization.
    Should be used to create the tensors for TensLoRA.

    Ideally, this function should be added inside the specialized TensorLoRA classes,
    for easier maintenance (in particular adding a new tensor method),
    but I couldn't think of a appropriate design pattern to do so.
    So may be TODO.

    Args:
        input_dim (int): The input dimension of the adapter.
        output_dim (int): The output dimension of the adapter.
        number_attention_layers (int): The number of attention layers in the model.
        n_components (int or list of int): The number of components for the tensor decomposition.
            - If int, it applies the same number of components to all modes.
            - If list of int, it applies the specified number of components to each mode.
            The length of the list must be equal to the number of modes of the tensor.
        tensor_method (str): The tensor method to use.
            - Options are "att", "qkv", "depth", "att_qkv", "att_depth", "qkv_depth", and "att_qkv_depth".
            - See tensors_manipulator for more details.
        tensor_fac (str): The tensor factorization to use.
            - Options are "tucker" and "cp".
        dropout_prob (float, optional): The dropout probability for the adapter. Default is 0.1.
        tensor_init (str, optional): The initialization method for the tensor decomposition.
            - Options are "orthogonal", "normal", and "uniform". Default is "orthogonal".
        scaling (float, optional): The scaling factor for the adapter. Default is 1.
        seed (int, optional): The random seed for the initialization. Default is 0.
        *args, **kwargs: Additional arguments for the tensor decomposition.
    """

    def _create_one_tensor():
        """
        Create a single tensor.
        """
        return TensLoRA_adapter(
            input_dim,
            output_dim,
            n_components,
            tensor_method,
            tensor_fac,
            dropout_prob,
            number_attention_layers=number_attention_layers,
            tensor_init=tensor_init,
            scaling=scaling,
            seed=seed,
            *args,
            **kwargs,
        )

    match tensor_method:
        case "att":
            all_tensors = nn.ParameterList()
            for idx_qkv in range(3):
                tens_one_line = nn.ParameterList()
                for idx_depth in range(number_attention_layers):
                    tens_one_line.append(_create_one_tensor())
                all_tensors.append(tens_one_line)  # Ugly tbh, but I didn't thought of a more elegant way to do it

        case "qkv":
            all_tensors = nn.ParameterList()
            for idx_depth in range(number_attention_layers):
                all_tensors.append(_create_one_tensor())

        case "depth":
            all_tensors = nn.ParameterList()
            for idx_qkv in range(3):
                all_tensors.append(_create_one_tensor())

        case "att_qkv":
            all_tensors = nn.ParameterList()
            for idx_depth in range(number_attention_layers):
                all_tensors.append(_create_one_tensor())

        case "att_depth":
            all_tensors = nn.ParameterList()
            for idx_qkv in range(3):
                all_tensors.append(_create_one_tensor())

        case "qkv_depth":
            all_tensors = _create_one_tensor()

        case "att_qkv_depth":
            all_tensors = _create_one_tensor()

        # Other cases can be added here as needed. The unknown case will raise an error in "_validate_tensor_method".

    _validate_all_tensor_shapes(
        all_tensors,
        input_dim,
        output_dim,
        number_attention_layers,
        n_components,
        tensor_method,
        tensor_fac,
        *args,
        **kwargs,
    )

    return all_tensors



def _validate_all_tensor_shapes(
    tensors,
    input_dim,
    output_dim,
    number_attention_layers,
    n_components,
    tensor_method,
    tensor_fac,
    *args,
    **kwargs,
):
    match tensor_method:
        case "att":
            number_heads = kwargs["number_heads"]
            assert isinstance(tensors, nn.ParameterList), "The parameter _tensors_ should be a list of tensors."
            assert len(tensors) == 3 and len(tensors[0]) == number_attention_layers, (
                f"Expected {3 * number_attention_layers} unique tensors for 'att' method, got {len(tensors) * len(tensors[0])}."
            )

            tensor_shape = tensors[0][0].tensor_lora_module.fold_tensor().size()
            assert tensor_shape == (input_dim, int(output_dim / number_heads), number_heads), (
                f"Expected (individual) tensor shape for 'att' method: ({input_dim}, {int(output_dim / number_heads)}, {number_heads}), got {tensor_shape}"
            )

        case "qkv":
            assert isinstance(tensors, nn.ParameterList), "The parameter _tensors_ should be a list of tensors."
            assert len(tensors) == number_attention_layers, (
                f"Expected {number_attention_layers} unique tensors for 'qkv' method, got {len(tensors)}"
            )

            tensor_shape = tensors[0].tensor_lora_module.fold_tensor().size()
            assert tensor_shape == (input_dim, output_dim, 3), (
                f"Expected (individual) tensor shape for 'qkv' method: ({input_dim}, {output_dim}, 3), got {tensor_shape}"
            )

        case "depth":
            assert isinstance(tensors, nn.ParameterList), "The parameter _tensors_ should be a list of tensors."
            assert len(tensors) == 3, f"Expected 3 unique tensors for 'depth' method, got {len(tensors)}"

            tensor_shape = tensors[0].tensor_lora_module.fold_tensor().size()
            assert tensor_shape == (input_dim, output_dim, number_attention_layers), (
                f"Expected (individual) tensor shape for 'depth' method: ({input_dim}, {output_dim}, {number_attention_layers}), got {tensor_shape}"
            )

        case "att_qkv":
            assert isinstance(tensors, nn.ParameterList), "The parameter _tensors_ should be a list of tensors."
            number_heads = kwargs["number_heads"]
            assert len(tensors) == number_attention_layers, (
                f"Expected {number_attention_layers} unique tensors for 'att_qkv' method, got {len(tensors)}"
            )

            tensor_shape = tensors[0].tensor_lora_module.fold_tensor().size()
            assert tensor_shape == (input_dim, int(output_dim / number_heads), number_heads, 3), (
                f"Expected (individual) tensor shape for 'att_qkv' method: ({input_dim}, {int(output_dim / number_heads)}, {number_heads}, 3), got {tensor_shape}"
            )

        case "att_depth":
            assert isinstance(tensors, nn.ParameterList), "The parameter _tensors_ should be a list of tensors."
            number_heads = kwargs["number_heads"]
            assert len(tensors) == 3, f"Expected 3 unique tensors for 'att_depth' method, got {len(tensors)}"

            tensor_shape = tensors[0].tensor_lora_module.fold_tensor().size()
            assert tensor_shape == (input_dim, int(output_dim / number_heads), number_heads, number_attention_layers), (
                f"Expected tensor shape for 'att_depth' method: ({input_dim}, {int(output_dim / number_heads)}, {number_heads}, {number_attention_layers}), got {tensor_shape}"
            )

        case "qkv_depth":
            assert isinstance(tensors, TensLoRA_adapter), "The parameter _tensors_ should be a TensLoRA_adapter."
            tensor_shape = tensors.tensor_lora_module.fold_tensor().size()
            assert tensor_shape == (input_dim, output_dim, 3, number_attention_layers), (
                f"Expected tensor shape for 'qkv_depth' method: ({input_dim}, {output_dim}, 3, {number_attention_layers}), got {tensor_shape}"
            )

        case "att_qkv_depth":
            assert isinstance(tensors, TensLoRA_adapter), "The parameter _tensors_ should be a TensLoRA_adapter."
            number_heads = kwargs["number_heads"]
            tensor_shape = tensors.tensor_lora_module.fold_tensor().size()
            assert tensor_shape == (input_dim, int(output_dim / number_heads), number_heads, 3, number_attention_layers), (
                f"Expected tensor shape for 'att_qkv_depth' method: ({input_dim}, {int(output_dim / number_heads)}, {number_heads}, 3, {number_attention_layers}), got {tensor_shape}"
            )


def _validate_tenslora_method(tensor_method, tensor_fac):
    """
    Keeps track of the valid tensor methods and factorization.
    Just a parameter check.
    Raises an error if the method is not supported.
    """
    valid_methods = tensors_manipulator.get_list_of_available_tensor_methods()
    if tensor_method not in valid_methods:
        raise ValueError(f"Unsupported tensor method: {tensor_method}. Supported methods are: {valid_methods}")

    valid_fac = tensors_manipulator.get_list_of_available_tensor_factorizations()
    if tensor_fac not in valid_fac:
        raise ValueError(f"Unsupported tensor factorization method: {tensor_fac}. Supported methods are: {valid_fac}")
