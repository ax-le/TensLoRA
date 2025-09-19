import math

import torch
from torch import nn

class LoRA_adapter(nn.Module):
    """ LoRA adapter module. Custom implementation, to be compared with Tensors."""
    def __init__(self, input_dim, output_dim, n_components, dropout_prob=0.1, seed=0, scaling=1):
        """
        Initialize the LoRA adapter.
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            n_components (int): Rank of the LoRA decomposition.
            dropout_prob (float): Dropout probability.
            seed (int): Random seed for reproducibility.
            scaling (float): Scaling factor for the LoRA output.
        """
        super(LoRA_adapter, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.lora_A = nn.Linear(input_dim, n_components, bias=False)
        self.lora_B = nn.Linear(n_components, output_dim, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        self.scaling = scaling
        self.lora_dropout = nn.Dropout(p=dropout_prob)

    # Define the forward pass for the adapter
    def forward(self, x, qkv=None, layer=None):
        # qkv and layer are useless for LoRA, but may be useful for TensLoRA,
        # so they are included for compatibility in the higher levels.
        # Dropout is applied on the input only
        if self.training:
            return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        else:
            return self.lora_B(self.lora_A(x)) * self.scaling

    # Function to get the update matrix (LoRA weight update)
    # Useless for LoRA, but may useful for TensLoRA, so included for compatibility.
    # TODO: can it be deleted?
    def get_update_matrix(self, qkv=None, layer=None):
        update_matrix = torch.matmul(self.lora_B.weight, self.lora_A.weight)
        return update_matrix * self.scaling
