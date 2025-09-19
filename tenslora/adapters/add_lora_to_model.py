import os
from copy import deepcopy
from typing import Union

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import GemmaForCausalLM, LlamaForCausalLM, RobertaForSequenceClassification, ViTForImageClassification
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import tenslora.adapters.custom_lora_adapter as custom_lora_adapter
import tenslora.adapters.tenslora_adapter as tenslora_adapter


def lora_router(
    model,
    lora_type,
    n_components,
    model_type="vit",
    input_dim=768,
    output_dim=768,
    scaling=1,
    dropout_prob=0.0,
    init_from_saved_tensors=False,
    tensor_path=None,
    tensor_persisted_name=None,
    seed=0,
    *args,
    **kwargs,
):
    """
    Routes the model to the appropriate LoRA implementation based on the lora_type.
    Supported types are:
    - "lora_hf": Uses HuggingFace's PEFT library for LoRA
    - "custom_lora": Uses a custom LoRA implementation
    - "tenslora": Uses TensLoRA for tensor-based LoRA
    Args:
        model: The model to which LoRA will be applied.
        lora_type: The type of LoRA to apply. Options are "lora_hf", "custom_lora", and "tenslora".
        n_components: The number of components (rank) for the LoRA adapter.
        dropout_prob: The dropout probability for the LoRA layers.
        init_from_saved_tensors: Whether to initialize TensLoRA from saved tensors.
        tensor_path: The path to load/save tensors for TensLoRA.
        tensor_persisted_name: The name to load/save tensors for TensLoRA.
        seed: The random seed for reproducibility.
        model_type: The type of model. Currently only "vit", "llama", "roberta", and "gemma" are supported.
        input_dim: The input dimension (hidden dimension) for the LoRA adapter.
        output_dim: The output dimension (hidden dimension) for the LoRA adapter.
        scaling: The scaling factor for the adapters.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    Returns:
        lora_model: The model with the adapters applied.
    Raises:
        ValueError: If the lora_type is not supported.
    """
    # autoconvert to int if possible
    if isinstance(n_components, str) and n_components.isdecimal():
        n_components = int(n_components)
    # autoconvert to float if possible
    if isinstance(scaling, str) and scaling.isdecimal():
        scaling = float(scaling)
    # Baseline using HugingFace
    if lora_type == "lora_hf":
        lora_model = get_lora_hf_model(
            model,
            n_components=n_components,
            model_type=model_type,
            dropout_prob=dropout_prob,
            seed=seed,
            scaling=scaling,
        )

    # Custom LoRA and TensLoRA implementations
    elif lora_type in ["custom_lora", "tenslora"]:
        if lora_type == "tenslora":
            assert "tensor_method" in kwargs, "tensor_method must be specified for tenslora"
            assert "tensor_fac" in kwargs, "tensor_fac must be specified for tenslora"
            tenslora_adapter._validate_tenslora_method(kwargs["tensor_method"], kwargs["tensor_fac"])

        lora_model = add_adapters_to_model(
            lora_type,
            model,
            input_dim,
            output_dim,
            n_components,
            model_type=model_type,
            dropout_prob=dropout_prob,
            init_from_saved_tensors=init_from_saved_tensors,
            tensor_path=tensor_path,
            tensor_persisted_name=tensor_persisted_name,
            scaling=scaling,
            seed=seed,
            *args,
            **kwargs,
        )

    # Otherwise, raise an error
    else:
        raise ValueError(f"Unsupported LoRA type: {lora_type}. Supported types are 'lora_hf', 'custom_lora', and 'tenslora'.")

    return lora_model

# Adding new models can be made here:
def add_adapters_to_model(
    lora_type,
    model: Union[nn.Module, LlamaForCausalLM, ViTForImageClassification, RobertaForSequenceClassification],
    input_dim,
    output_dim,
    n_components,
    model_type="vit",
    dropout_prob=0.1,
    init_from_saved_tensors=False,
    tensor_path=None,
    tensor_persisted_name=None,
    scaling=1,
    seed=0,
    *args,
    **kwargs,
):
    """
    Adds LoRA adapters to the model.
    Handles both custom LoRA and TensLoRA implementations.
    New models can be added by extending this function.
    Args:
        lora_type: The type of LoRA to apply. Options are "lora_hf", "custom_lora", and "tenslora".
        model: The model to which LoRA will be applied.
        input_dim: The input dimension (hidden dimension) for the adapters.
        output_dim: The output dimension (hidden dimension) for the adapters.
        n_components: The number of components (rank) for the LoRA adapter.
            - For tenslora, this is the number of components in the tensor decomposition.
            In the Tucker decomposition, n_components can be a list.
            If it's an int, it applies the same number of components to all modes.
        model_type: The type of model. Currently only "vit", "llama", "roberta", and "gemma" are supported.
        dropout_prob: The dropout probability for the adapters.
        init_from_saved_tensors: Whether to initialize TensLoRA from saved tensors.
        tensor_path: The path to load/save tensors for TensLoRA.
        tensor_persisted_name: The name to load/save tensors for TensLoRA.
        scaling: The scaling factor for the adapters.
        seed: The random seed for reproducibility.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    Returns:
        model: The model with the adapters applied.
    Raises:
        NotImplementedError: If the model_type is not supported.
        ValueError: If the lora_type is not supported the tenslora/tensor factorization method is not supported or specified.
    """

    assert isinstance(model, (LlamaForCausalLM, ViTForImageClassification, RobertaForSequenceClassification, GemmaForCausalLM)), (
        "Model must be LlamaForCausalLM or ViTForImageClassification or RobertaForSequenceClassification or GemmaForCausalLM."
    )

    assert lora_type in ["custom_lora", "tenslora"], (
        "Currently only custom_lora and tenslora are supported. "
        "For HuggingFace LoRA, use the get_lora_hf_model function."
        "You can access to HF LoRA by using the lora_router function."
    )

    # Freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    print("Adding adapters to model:", model_type, "with LoRA type:", lora_type)

    if lora_type == "tenslora":
        tensor_method = kwargs.pop("tensor_method")
        tensor_fac = kwargs.pop("tensor_fac")
        tensor_init = kwargs.pop("tensor_init")

        # Access to the number of attention layers and heads
        if model_type == "vit" and isinstance(model, ViTForImageClassification):
            number_attention_layers = len(model.vit.encoder.layer)
            number_heads = model.vit.config.num_attention_heads  # Should be 12
        elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
            number_attention_layers = len(model.model.layers)
            number_heads = model.model.config.num_attention_heads
        elif model_type == "roberta" and isinstance(model, RobertaForSequenceClassification):
            number_attention_layers = len(model.roberta.encoder.layer)
            number_heads = model.roberta.config.num_attention_heads
        elif model_type == "gemma" and isinstance(model, GemmaForCausalLM):
            number_attention_layers = len(model.model.layers)
            number_heads = model.model.config.num_attention_heads
        else:
            raise NotImplementedError(f"Model type {model_type} is not supported (yet).")

        if init_from_saved_tensors:
            assert tensor_path is not None, "Path to load tensors must be provided."
            assert tensor_persisted_name is not None, "Title for loading tensors must be provided."
            tensors = load_tensors_from_path(tensor_path, tensor_persisted_name)

        else:  # Initialize tensors from scratch
            tensors = tenslora_adapter.create_all_tensor_adapters(
                input_dim=input_dim,
                output_dim=output_dim,
                number_attention_layers=number_attention_layers,
                n_components=n_components,
                tensor_method=tensor_method,
                tensor_fac=tensor_fac,
                tensor_init=tensor_init,
                dropout_prob=dropout_prob,
                number_heads=number_heads,
                scaling=scaling,
                seed=seed,
                *args,
                **kwargs,
            )

        # Validates tensor shapes
        tenslora_adapter._validate_all_tensor_shapes(
            tensors,
            input_dim,
            output_dim,
            number_attention_layers,
            n_components,
            tensor_method=tensor_method,
            tensor_fac=tensor_fac,
            number_heads=number_heads,
            *args,
            **kwargs,
        )

    elif lora_type == "custom_lora":
        tensor_method = None  # Custom LoRA does not use tensors, so we set it to None
        tensors = None  # Custom LoRA does not use tensors, so we set it to None

    else:
        raise NotImplementedError(
            f"LoRA type {lora_type} is not supported (yet). (Jonathan: you should use the function lora_router instead.)",
        )

    # Add adapters to the model. Depends on the model, on its type of attention layers.
    if model_type == "vit" and isinstance(model, ViTForImageClassification):
        for i in range(len(model.vit.encoder.layer)):  # Add adapters to each layer
            # Get the original layer
            original_layer = model.vit.encoder.layer[i]

            # Create a new layer with the adapter
            # Note: We use deepcopy to avoid modifying the original layer in place
            new_layer = deepcopy(original_layer)
            new_layer.attention.attention.query = AttentionWithAdapter(
                lora_type,
                original_layer.attention.attention.query,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=0,
                layer=i,
                seed=seed,
                scaling=scaling,
            )
            new_layer.attention.attention.key = AttentionWithAdapter(
                lora_type,
                original_layer.attention.attention.key,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=1,
                layer=i,
                seed=seed,
                scaling=scaling,
            )
            new_layer.attention.attention.value = AttentionWithAdapter(
                lora_type,
                original_layer.attention.attention.value,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=2,
                layer=i,
                seed=seed,
                scaling=scaling,
            )

            # Replace the original layer with the new layer
            model.vit.encoder.layer[i] = new_layer

    elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
        for i in range(len(model.model.layers)):
            original_layer: LlamaDecoderLayer = model.model.layers[i]

            new_layer: LlamaDecoderLayer = deepcopy(original_layer)
            new_layer.self_attn.q_proj = AttentionWithAdapter(
                lora_type,
                original_layer.self_attn.q_proj,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=0,
                layer=i,
                seed=seed,
                scaling=scaling,
            )
            new_layer.self_attn.k_proj = AttentionWithAdapter(
                lora_type,
                original_layer.self_attn.k_proj,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=1,
                layer=i,
                n_repeats=4,  # gqa with 4 repeats
                seed=seed,
                scaling=scaling,
            )
            new_layer.self_attn.v_proj = AttentionWithAdapter(
                lora_type,
                original_layer.self_attn.v_proj,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=2,
                layer=i,
                n_repeats=4,  # gqa with 4 repeats
                seed=seed,
                scaling=scaling,
            )

            # Replace the original layer with the new layer
            model.model.layers[i] = new_layer

    elif model_type == "roberta" and isinstance(model, RobertaForSequenceClassification):
        for i in range(len(model.roberta.encoder.layer)):
            original_layer = model.roberta.encoder.layer[i]

            new_layer = deepcopy(original_layer)
            new_layer.attention.self.query = AttentionWithAdapter(
                lora_type,
                original_layer.attention.self.query,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=0,
                layer=i,
                seed=seed,
                scaling=scaling,
            )
            new_layer.attention.self.key = AttentionWithAdapter(
                lora_type,
                original_layer.attention.self.key,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=1,
                layer=i,
                seed=seed,
                scaling=scaling,
            )
            new_layer.attention.self.value = AttentionWithAdapter(
                lora_type,
                original_layer.attention.self.value,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=2,
                layer=i,
                seed=seed,
                scaling=scaling,
            )

            # Replace the original layer with the new layer
            model.roberta.encoder.layer[i] = new_layer

    elif model_type == "gemma" and isinstance(model, GemmaForCausalLM):
        for i in range(len(model.model.layers)):
            original_layer: GemmaDecoderLayer = model.model.layers[i]

            new_layer: GemmaDecoderLayer = deepcopy(original_layer)
            new_layer.self_attn.q_proj = AttentionWithAdapter(
                lora_type,
                original_layer.self_attn.q_proj,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=0,
                layer=i,
                seed=seed,
                scaling=scaling,
            )
            new_layer.self_attn.k_proj = AttentionWithAdapter(
                lora_type,
                original_layer.self_attn.k_proj,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=1,
                layer=i,
                seed=seed,
                scaling=scaling,
            )
            new_layer.self_attn.v_proj = AttentionWithAdapter(
                lora_type,
                original_layer.self_attn.v_proj,
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                tensor_method=tensor_method,
                tensor_adapters=tensors,
                qkv=2,
                layer=i,
                seed=seed,
                scaling=scaling,
            )

            # Replace the original layer with the new layer
            model.model.layers[i] = new_layer

    else:
        raise NotImplementedError(f"Model type {model_type} is not supported (yet).")

    return model


# Getter for the HuggingFace PEFT LoRA implementation
def get_lora_hf_model(
    model: Union[LlamaForCausalLM, ViTForImageClassification, GemmaForCausalLM, RobertaForSequenceClassification],
    n_components,
    model_type,
    dropout_prob=0.1,
    scaling=1,
    seed=0, # Useless actually
):
    if model_type == "vit" and isinstance(model, ViTForImageClassification):
        config = LoraConfig(
            r=n_components,
            lora_alpha=n_components*scaling,
            target_modules=["query", "key", "value"],
            lora_dropout=dropout_prob,
            bias="none",
            # modules_to_save=["classifier"],
        )
    elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
        # For Llama, we use the HuggingFace PEFT library
        config = LoraConfig(
            r=n_components,
            lora_alpha=n_components*scaling,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=dropout_prob,
            bias="none",
        )
    elif model_type == "gemma" and isinstance(model, GemmaForCausalLM):
        config = LoraConfig(
            r=n_components,
            lora_alpha=n_components*scaling,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=dropout_prob,
            bias="none",
        )
    elif model_type == "roberta" and isinstance(model, RobertaForSequenceClassification):
        config = LoraConfig(
            r=n_components,
            lora_alpha=n_components*scaling,
            target_modules=["query", "key", "value"],
            lora_dropout=dropout_prob,
            bias="none",
        )
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported (yet).")

    return get_peft_model(model, config)


class AttentionWithAdapter(nn.Module):
    """
    A wrapper for the original attention block that adds a TensLoRA adapter.
    This class is used to apply TensLoRA adapters to the attention blocks of a model.
    It can handle both custom LoRA implementations and TensLoRA tensor-based adapters.
    Args:
        lora_type: The type of LoRA to apply. Options are "custom_lora", and "tenslora".
        original_attention_block: The original attention block to which the LoRA adapter will be applied.
        input_dim: The input dimension for the LoRA adapter.
        output_dim: The output dimension for the LoRA adapter.
        n_components: The number of components (rank) for the LoRA/TensLoRA adapter.
            - For tenslora, this is the number of components in the tensor decomposition.
            In the Tucker decomposition, n_components can be a list.
            If it's an int, it applies the same number of components to all modes.
        dropout_prob: The dropout probability for the adapters.
        tensor_method: Optional; Only for tenslora.
            - the TensLoRA tensor method to use.
            - Options are "att", "qkv", "depth", "att_qkv", "att_depth", "qkv_depth", and "att_qkv_depth".
            See tensors_manipulator for more details.
        qkv: Optional; Only for tenslora.
            - an index for the query, key, and value tensors for TensLoRA.
            query corresponds to 0, key to 1, and value to 2.
            - It is used to select the appropriate tensor from the TensLoRA tensors (that are constructed at the level
            of the model, and not at the level of the attention layer).
        layer: Optional; Only for tenslora.
            - the layer index for TensLoRA.
            - It is used to select the appropriate tensor from the TensLoRA tensors (that are constructed at the level
            of the model, and not at the level of the attention layer).
        tensor_adapter: Optional; Only for tenslora.
            - the TensLoRA tensor adapter to use.
            Tensors are created at the model level, and passed to each attention layer.
            Hence this function assumes that tensors are already created.
        n_repeats: For GQA. TODO: Ask Jonathan for details on this part.
        seed: The random seed for reproducibility. Only for custom lora here because,
            for TensLoRA, tensors are already created (with a seed), and passed as arguments.
        scaling: The scaling factor for the adapters. Only for custom lora here because,
            for TensLoRA, scaling is applied at the tensor level.
    """

    def __init__(
        self,
        lora_type,
        original_attention_block,
        input_dim,
        output_dim,
        n_components,
        dropout_prob=0.1,
        tensor_method=None,
        qkv=None,
        layer=None,
        tensor_adapters=None,
        n_repeats=1,
        seed=0,  # Only for custom lora here
        scaling=1,  # Only for custom lora here
    ):
        super(AttentionWithAdapter, self).__init__()
        self.original_attention_block = original_attention_block

        # Creates adapters for each layer in the model
        if lora_type == "custom_lora":
            self.adapter = custom_lora_adapter.LoRA_adapter(
                input_dim,
                output_dim,
                n_components,
                dropout_prob,
                seed=seed,
                scaling=scaling,
            )
        # Checks that arguments are ok.
        elif lora_type == "tenslora":
            assert tensor_adapters is not None, "For TensLoRA, the tensor must be provided."
            assert qkv is not None, "For TensLoRA, the qkv index must be provided."
            assert layer is not None, "For TensLoRA, the layer index must be provided."
            # Depending on the tensor method, select the appropriate tensor adapter
            self.adapter = tenslora_adapter.select_tensor_among_adapters(tensor_adapters, tensor_method, qkv=qkv, layer=layer)
        # Otherwise, raise an error. Should have been caught earlier.
        else:
            raise ValueError(f"Unsupported LoRA type: {lora_type}. Supported types are 'custom_lora', and 'tenslora'.")

        # Useless for LoRA, but important for TensLoRA
        self.qkv = qkv
        self.layer = layer
        self.repeats = n_repeats

        if n_repeats > 1:
            # If the attention block has multiple repeats (e.g., GQA), we need to handle it differently
            self.forward = self._forward_with_repeats
        else:
            # If the attention block has no repeats, we can use the standard forward pass
            self.forward = self._forward_no_repeats

    # Actually define the forward pass with adapters
    def _forward_no_repeats(self, x):
        original_attn_output = self.original_attention_block(x)
        adapter_output = self.adapter(x, qkv=self.qkv, layer=self.layer)
        output = original_attn_output + adapter_output
        return output

    # Actually define the forward pass with adapters, but with repetitions (e.g., GQA)
    def _forward_with_repeats(self, x):
        """
        Forward pass through the attention block with the LoRA adapter.
        This method handles the case where the attention block has multiple repeats (e.g., GQA).
        """
        original_attn_output = self.original_attention_block(x)
        adapter_output = self.adapter(x, qkv=self.qkv, layer=self.layer)

        # fixing the shape mismatch in gqa
        B, T, D = adapter_output.shape
        adapter_output = adapter_output.view(B, T, self.repeats, D // self.repeats).mean(dim=2)

        output = original_attn_output + adapter_output
        return output

# %% Saving and loading tensors learned in TensLoRA
def load_tensors_from_path(path, tensor_persisted_name):
    tensors = torch.load(f"{path}/{tensor_persisted_name}_tensors.pth", weights_only=False)
    return tensors


def save_tensors_from_model(  # noqa: C901, PLR0912, PLR0915
    lora_type,
    tensor_method,
    model: Union[nn.Module, LlamaForCausalLM, ViTForImageClassification],
    model_type="vit",
    tensor_path=None,
    tensor_persisted_name=None,
):
    assert isinstance(model, (LlamaForCausalLM, ViTForImageClassification, GemmaForCausalLM)), (
        "Model must be either LlamaForCausalLM or ViTForImageClassification or GemmaForCausalLM."
    )

    assert lora_type == "tenslora", "This function is only for TensLoRA."

    match tensor_method:
        case "att":
            tensors = nn.ParameterList()
            if model_type == "vit" and isinstance(model, ViTForImageClassification):
                query_tensors = nn.ParameterList()
                for i in range(len(model.vit.encoder.layer)):
                    query_tensors.append(model.vit.encoder.layer[i].attention.attention.query.adapter)
                tensors.append(query_tensors)

                key_tensors = nn.ParameterList()
                for i in range(len(model.vit.encoder.layer)):
                    key_tensors.append(model.vit.encoder.layer[i].attention.attention.key.adapter)
                tensors.append(key_tensors)

                value_tensors = nn.ParameterList()
                for i in range(len(model.vit.encoder.layer)):
                    value_tensors.append(model.vit.encoder.layer[i].attention.attention.value.adapter)
                tensors.append(value_tensors)

            elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
                query_tensors = nn.ParameterList()
                for i in range(len(model.model.layers)):
                    query_tensors.append(model.model.layers[i].self_attn.q_proj.adapter)
                tensors.append(query_tensors)

                key_tensors = nn.ParameterList()
                for i in range(len(model.model.layers)):
                    key_tensors.append(model.model.layers[i].self_attn.k_proj.adapter)
                tensors.append(key_tensors)

                value_tensors = nn.ParameterList()
                for i in range(len(model.model.layers)):
                    value_tensors.append(model.model.layers[i].self_attn.v_proj.adapter)
                tensors.append(value_tensors)

        case "qkv":
            tensors = nn.ParameterList()
            if model_type == "vit" and isinstance(model, ViTForImageClassification):
                for i in range(len(model.vit.encoder.layer)):
                    tensors.append(model.vit.encoder.layer[i].attention.attention.query.adapter)
            elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
                for i in range(len(model.model.layers)):
                    tensors.append(model.model.layers[i].self_attn.q_proj.adapter)

        case "depth":
            tensors = nn.ParameterList()
            if model_type == "vit" and isinstance(model, ViTForImageClassification):
                tensors.append(model.vit.encoder.layer[0].attention.attention.query.adapter)
                tensors.append(model.vit.encoder.layer[0].attention.attention.key.adapter)
                tensors.append(model.vit.encoder.layer[0].attention.attention.value.adapter)
            elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
                tensors.append(model.model.layers[0].self_attn.q_proj.adapter)
                tensors.append(model.model.layers[0].self_attn.k_proj.adapter)
                tensors.append(model.model.layers[0].self_attn.v_proj.adapter)

        case "att_qkv":
            tensors = nn.ParameterList()
            if model_type == "vit" and isinstance(model, ViTForImageClassification):
                for i in range(len(model.vit.encoder.layer)):
                    tensors.append(model.vit.encoder.layer[i].attention.attention.query.adapter)
            elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
                for i in range(len(model.model.layers)):
                    tensors.append(model.model.layers[i].self_attn.q_proj.adapter)

        case "att_depth":
            tensors = nn.ParameterList()
            if model_type == "vit" and isinstance(model, ViTForImageClassification):
                tensors.append(model.vit.encoder.layer[0].attention.attention.query.adapter)
                tensors.append(model.vit.encoder.layer[0].attention.attention.key.adapter)
                tensors.append(model.vit.encoder.layer[0].attention.attention.value.adapter)
            elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
                tensors.append(model.model.layers[0].self_attn.q_proj.adapter)
                tensors.append(model.model.layers[0].self_attn.k_proj.adapter)
                tensors.append(model.model.layers[0].self_attn.v_proj.adapter)

        case "qkv_depth":
            if model_type == "vit" and isinstance(model, ViTForImageClassification):
                tensors = model.vit.encoder.layer[0].attention.attention.query.adapter
            elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
                tensors = model.model.layers[0].self_attn.q_proj.adapter

        case "att_qkv_depth":
            if model_type == "vit" and isinstance(model, ViTForImageClassification):
                tensors = model.vit.encoder.layer[0].attention.attention.query.adapter
            elif model_type == "llama" and isinstance(model, LlamaForCausalLM):
                tensors = model.model.layers[0].self_attn.q_proj.adapter

        case _:
            raise ValueError(f"Unsupported tensor method: {tensor_method}.")

    os.makedirs(tensor_path, exist_ok=True)
    save_path = f"{tensor_path}/{tensor_persisted_name}_tensors.pth"
    if os.path.exists(save_path):
        print(f"Warning: {save_path} already exists. incrementing the name to avoid overwriting.")
        i = 1
        while os.path.exists(f"{tensor_path}/{tensor_persisted_name}_tensors_{i}.pth"):
            i += 1
        save_path = f"{tensor_path}/{tensor_persisted_name}_tensors_{i}.pth"
    torch.save(tensors, save_path)
