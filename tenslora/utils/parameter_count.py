def count_trainable_parameters(model):
    adapter_params = 0
    trainable_params = 0
    all_param = 0
    for param_name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "lora" in param_name:
                adapter_params += param.numel()
    return adapter_params, trainable_params, all_param


def predict_lora_parameters(lora_rank, hidden_dim=768, layer=12):
    return 2 * lora_rank * hidden_dim * layer * 3


def predict_tenslora_parameters(method, tenslora_set_ranks, hidden_dim=768, layer=12, num_heads=12):
    if type(tenslora_set_ranks) == int:
        if method in ["att", "qkv", "depth"]:
            tenslora_set_ranks = [tenslora_set_ranks] * 3
        elif method in ["att_qkv", "att_depth", "qkv_depth"]:
            tenslora_set_ranks = [tenslora_set_ranks] * 4
        elif method == "att_qkv_depth":
            tenslora_set_ranks = [tenslora_set_ranks] * 5
        else:
            raise ValueError(f"Unsupported method: {method}")

    if len(tenslora_set_ranks) != 3 and len(tenslora_set_ranks) != 4 and len(tenslora_set_ranks) != 5:
        raise ValueError("tenslora_set_ranks must have 3, 4, or 5 elements.")

    if method == "att":
        assert len(tenslora_set_ranks) == 3, "For 'att' method, tenslora_set_ranks should have 3 elements."
        tenslora_count = (
            3
            * layer
            * (
                tenslora_set_ranks[0] * hidden_dim
                + tenslora_set_ranks[1] * int(hidden_dim / num_heads)
                + tenslora_set_ranks[2] * num_heads
                + tenslora_set_ranks[0] * tenslora_set_ranks[1] * tenslora_set_ranks[2]
            )
        )
    elif method == "qkv":
        assert len(tenslora_set_ranks) == 3, "For 'qkv' method, tenslora_set_ranks should have 3 elements."
        tenslora_count = layer * (
            tenslora_set_ranks[0] * hidden_dim
            + tenslora_set_ranks[1] * hidden_dim
            + tenslora_set_ranks[2] * 3
            + tenslora_set_ranks[0] * tenslora_set_ranks[1] * tenslora_set_ranks[2]
        )
    elif method == "depth":
        assert len(tenslora_set_ranks) == 3, "For 'depth' method, tenslora_set_ranks should have 3 elements."
        tenslora_count = 3 * (
            tenslora_set_ranks[0] * hidden_dim
            + tenslora_set_ranks[1] * hidden_dim
            + tenslora_set_ranks[2] * layer
            + tenslora_set_ranks[0] * tenslora_set_ranks[1] * tenslora_set_ranks[2]
        )
    elif method == "att_qkv":
        assert len(tenslora_set_ranks) == 4, "For 'att_qkv' method, tenslora_set_ranks should have 4 elements."
        tenslora_count = layer * (
            tenslora_set_ranks[0] * hidden_dim
            + tenslora_set_ranks[1] * int(hidden_dim / num_heads)
            + tenslora_set_ranks[2] * num_heads
            + tenslora_set_ranks[3] * 3
            + tenslora_set_ranks[0] * tenslora_set_ranks[1] * tenslora_set_ranks[2] * tenslora_set_ranks[3]
        )
    elif method == "att_depth":
        assert len(tenslora_set_ranks) == 4, "For 'att_depth' method, tenslora_set_ranks should have 4 elements."
        tenslora_count = 3 * (
            tenslora_set_ranks[0] * hidden_dim
            + tenslora_set_ranks[1] * int(hidden_dim / num_heads)
            + tenslora_set_ranks[2] * num_heads
            + tenslora_set_ranks[3] * layer
            + tenslora_set_ranks[0] * tenslora_set_ranks[1] * tenslora_set_ranks[2] * tenslora_set_ranks[3]
        )
    elif method == "qkv_depth":
        assert len(tenslora_set_ranks) == 4, "For 'qkv_depth' method, tenslora_set_ranks should have 4 elements."
        tenslora_count = (
            tenslora_set_ranks[0] * hidden_dim
            + tenslora_set_ranks[1] * hidden_dim
            + tenslora_set_ranks[2] * 3
            + tenslora_set_ranks[3] * layer
            + tenslora_set_ranks[0] * tenslora_set_ranks[1] * tenslora_set_ranks[2] * tenslora_set_ranks[3]
        )
    elif method == "att_qkv_depth":
        assert len(tenslora_set_ranks) == 5, "For 'att_qkv_depth' method, tenslora_set_ranks should have 5 elements."
        tenslora_count = (
            tenslora_set_ranks[0] * hidden_dim
            + tenslora_set_ranks[1] * int(hidden_dim / num_heads)
            + tenslora_set_ranks[2] * num_heads
            + tenslora_set_ranks[3] * 3
            + tenslora_set_ranks[4] * layer
            + tenslora_set_ranks[0]
            * tenslora_set_ranks[1]
            * tenslora_set_ranks[2]
            * tenslora_set_ranks[3]
            * tenslora_set_ranks[4]
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    return tenslora_count
