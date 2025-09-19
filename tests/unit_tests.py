import unittest

import torch

# %% Tensor creation
import tenslora.tensors.tensors_manipulator as tens_man


class TestTensorCreation(unittest.TestCase):
    def test_create_tensor(self):
        cp_tensor = tens_man.CPLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=2)
        tensor = cp_tensor.fold_tensor()
        self.assertEqual(tensor.shape, (4, 5, 6))

        tucker_tensor = tens_man.TuckerLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=[2, 3, 4])
        tensor = tucker_tensor.fold_tensor()
        self.assertEqual(tensor.shape, (4, 5, 6))

    def test_invalid_number_of_modes(self):
        # Negative number of modes
        with self.assertRaises(AssertionError):
            tens_man.CPLoRA(number_modes=-1, dimensions=[4, 5], n_components=2)
        with self.assertRaises(AssertionError):
            tens_man.TuckerLoRA(number_modes=-1, dimensions=[4, 5], n_components=[2, 3])

        # Zero number of modes
        with self.assertRaises(AssertionError):
            tens_man.CPLoRA(number_modes=0, dimensions=[4, 5], n_components=2)
        with self.assertRaises(AssertionError):
            tens_man.TuckerLoRA(number_modes=0, dimensions=[4, 5], n_components=[2, 3])

        # Real valued number of modes
        with self.assertRaises(AssertionError):
            tens_man.CPLoRA(number_modes=2.1, dimensions=[4, 5], n_components=2)
        with self.assertRaises(AssertionError):
            tens_man.TuckerLoRA(number_modes=2.1, dimensions=[4, 5], n_components=[2, 3])

    def test_invalid_tensor_dimensions(self):
        with self.assertRaises(AssertionError):
            tens_man.CPLoRA(number_modes=3, dimensions=[4, 5], n_components=2)
        with self.assertRaises(AssertionError):
            tens_man.TuckerLoRA(number_modes=3, dimensions=[4, 5], n_components=[2, 3])

    def test_invalid_tensor_n_components_length(self):
        with self.assertRaises(AssertionError):
            tens_man.TuckerLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=[2, 3])
        with self.assertRaises(AssertionError):
            tens_man.TuckerLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=[2, 3, 4, 5])

    def test_invalid_tensor_n_components(self):
        # Negative n_components
        with self.assertRaises(AssertionError):
            tens_man.CPLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=-1)
        with self.assertRaises(AssertionError):
            tens_man.TuckerLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=[2, 3, -1])

        # Zero n_components
        with self.assertRaises(AssertionError):
            tens_man.CPLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=0)
        with self.assertRaises(AssertionError):
            tens_man.TuckerLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=[0, 3, 4])

        # Real valued n_components
        with self.assertRaises(AssertionError):
            tens_man.CPLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=0.2)
        with self.assertRaises(AssertionError):
            tens_man.TuckerLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=[0.5, 3, 4])

    # def test_invalid_tensor_n_components_greater_than_dimensions(self): # Decided it was too restrictive in the end
    #     # n_components greater than dimensions
    #     with self.assertRaises(AssertionError):
    #         tensor_creator.CPLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=10)
    #     with self.assertRaises(AssertionError):
    #         tensor_creator.TuckerLoRA(number_modes=3, dimensions=[4, 5, 6], n_components=[10, 3, 4])


# %% Tensor adapter
import tenslora.adapters.tenslora_adapter as tenslora_adapter


class TestTensorAdapterInit(unittest.TestCase):
    def test_invalid_tensor_method(self):
        with self.assertRaises(ValueError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=2,
                tensor_method="invalid_method",
                tensor_fac="tucker",
            )
        with self.assertRaises(ValueError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=2,
                tensor_method="invalid_method",
                tensor_fac="cp",
            )

    def test_invalid_tensor_factorization(self):
        with self.assertRaises(ValueError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=2,
                tensor_method="qkv",
                tensor_fac="invalid_fac",
            )

    def test_validate_tensor_method_att(self):
        tenslora_adapter.TensLoRA_adapter(
            input_dim=4,
            output_dim=4,
            n_components=[2, 2, 2],
            tensor_method="att",
            number_heads=2,
            tensor_fac="tucker",
        )

    def test_validate_tensor_method_att_invalid_heads(self):
        # Missing the number_heads parameter
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2],
                tensor_method="att",
                tensor_fac="tucker",
            )

        # Invalid number of heads
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2],
                tensor_method="att",
                number_heads=0,
                tensor_fac="tucker",
            )

        # Number of heads not divisible by output_dim
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2],
                tensor_method="att",
                number_heads=3,
                tensor_fac="tucker",
            )

    def test_validate_tensor_method_qkv(self):
        tenslora_adapter.TensLoRA_adapter(
            input_dim=4,
            output_dim=4,
            n_components=[2, 2, 2],
            tensor_method="qkv",
            tensor_fac="tucker",
        )

    def test_validate_tensor_method_depth(self):
        tenslora_adapter.TensLoRA_adapter(
            input_dim=4,
            output_dim=4,
            n_components=[2, 2, 2],
            tensor_method="depth",
            number_attention_layers=10,
            tensor_fac="tucker",
        )

    def test_validate_tensor_method_depth_invalid_number_attention_layers(self):
        # Missing the number_attention_layers parameter
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2],
                tensor_method="depth",
                tensor_fac="tucker",
            )

        # Invalid number_attention_layers
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2],
                tensor_method="depth",
                number_attention_layers=-1,
                tensor_fac="tucker",
            )

    def test_validate_tensor_method_att_qkv(self):
        tenslora_adapter.TensLoRA_adapter(
            input_dim=4,
            output_dim=4,
            n_components=[2, 2, 2, 2],
            tensor_method="att_qkv",
            number_heads=2,
            tensor_fac="tucker",
        )

    def test_validate_tensor_method_att_qkv_invalid_heads(self):
        # Missing the number_heads parameter
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="att_qkv",
                tensor_fac="tucker",
            )

        # Invalid number of heads
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="att_qkv",
                number_heads=-1,
                tensor_fac="tucker",
            )

        # Number of heads not divisible by output_dim
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="att_qkv",
                number_heads=3,
                tensor_fac="tucker",
            )

    def test_validate_tensor_method_att_depth(self):
        tenslora_adapter.TensLoRA_adapter(
            input_dim=4,
            output_dim=4,
            n_components=[2, 2, 2, 2],
            tensor_method="att_qkv",
            number_heads=2,
            number_attention_layers=3,
            tensor_fac="tucker",
        )

    def test_validate_tensor_method_att_depth_invalid_numheads_or_depth(self):
        # Missing the number_heads parameter
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="att_depth",
                number_attention_layers=3,
                tensor_fac="tucker",
            )

        # Invalid number of heads
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="att_depth",
                number_heads=0,
                number_attention_layers=3,
                tensor_fac="tucker",
            )

        # Number of heads not divisible by output_dim
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="att_depth",
                number_heads=3,
                number_attention_layers=3,
                tensor_fac="tucker",
            )

        # Missing the number_attention_layers parameter
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="att_depth",
                number_heads=2,
                tensor_fac="tucker",
            )

        # Invalid number_attention_layers
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="att_depth",
                number_attention_layers=-1,
                number_heads=2,
                tensor_fac="tucker",
            )

        # Missing both number_heads and number_attention_layers parameters
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="att_depth",
                tensor_fac="tucker",
            )

    def test_validate_tensor_method_qkv_depth(self):
        tenslora_adapter.TensLoRA_adapter(
            input_dim=4,
            output_dim=4,
            n_components=[2, 2, 2, 2],
            tensor_method="qkv_depth",
            number_attention_layers=3,
            tensor_fac="tucker",
        )

    def test_validate_tensor_method_qkv_depth_invalid_number_attention_layers(self):
        # Missing the number_attention_layers parameter
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="qkv_depth",
                tensor_fac="tucker",
            )

        # Invalid number_attention_layers
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2],
                tensor_method="qkv_depth",
                number_attention_layers=0,
                tensor_fac="tucker",
            )

    def test_validate_tensor_method_att_qkv_depth(self):
        tenslora_adapter.TensLoRA_adapter(
            input_dim=4,
            output_dim=4,
            n_components=[2, 2, 2, 2, 2],
            tensor_method="att_qkv_depth",
            number_heads=2,
            number_attention_layers=3,
            tensor_fac="tucker",
        )

    def test_validate_tensor_method_att_qkv_depth_invalid_numheads_or_depth(self):
        # Missing the number_heads parameter
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2, 2],
                tensor_method="att_qkv_depth",
                number_attention_layers=3,
                tensor_fac="tucker",
            )

        # Invalid number of heads
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2, 2],
                tensor_method="att_qkv_depth",
                number_heads=0,
                number_attention_layers=3,
                tensor_fac="tucker",
            )

        # Number of heads not divisible by output_dim
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2, 2],
                tensor_method="att_qkv_depth",
                number_heads=3,
                number_attention_layers=3,
                tensor_fac="tucker",
            )

        # Missing the number_attention_layers parameter
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2, 2],
                tensor_method="att_qkv_depth",
                number_heads=2,
                tensor_fac="tucker",
            )

        # Invalid number_attention_layers
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2, 2],
                tensor_method="att_qkv_depth",
                number_attention_layers=-1,
                number_heads=2,
                tensor_fac="tucker",
            )

        # Missing both number_heads and number_attention_layers parameters
        with self.assertRaises(AssertionError):
            tenslora_adapter.TensLoRA_adapter(
                input_dim=4,
                output_dim=4,
                n_components=[2, 2, 2, 2, 2],
                tensor_method="att_qkv_depth",
                tensor_fac="tucker",
            )


from tenslora.adapters.tenslora_adapter import create_all_tensor_adapters


class TestCreateAllTensors(unittest.TestCase):
    def setUp(self):
        self.input_dim = 16
        self.output_dim = 16
        self.number_attention_layers = 5
        self.number_heads = 4

    def test_create_all_tensors_att(self):
        self.n_components = [2, 2, 2]
        tensors = create_all_tensor_adapters(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="att",
            tensor_fac="tucker",
            number_heads=self.number_heads,
        )

        tenslora_adapter._validate_all_tensor_shapes(
            tensors,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="att",
            tensor_fac="tucker",
            number_heads=self.number_heads,
        )

        # With a wrong tensor_method
        with self.assertRaises(AssertionError):
            tenslora_adapter._validate_all_tensor_shapes(
                tensors,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                number_attention_layers=self.number_attention_layers,
                n_components=self.n_components,
                tensor_method="att_qkv",
                tensor_fac="tucker",
                number_heads=self.number_heads,
            )

    def test_create_all_tensors_qkv(self):
        self.n_components = [2, 2, 2]
        tensors = create_all_tensor_adapters(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="qkv",
            tensor_fac="tucker",
        )

        tenslora_adapter._validate_all_tensor_shapes(
            tensors,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="qkv",
            tensor_fac="tucker",
        )

        # With a wrong tensor_method
        with self.assertRaises(AssertionError):
            tenslora_adapter._validate_all_tensor_shapes(
                tensors,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                number_attention_layers=self.number_attention_layers,
                n_components=self.n_components,
                tensor_method="depth",
                tensor_fac="tucker",
                number_heads=self.number_heads,
            )

    def test_create_all_tensors_depth(self):
        self.n_components = [2, 2, 2]

        tensors = create_all_tensor_adapters(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="depth",
            tensor_fac="tucker",
        )
        tensors_shape = tensors[0].tensor_lora_module.fold_tensor().size()
        self.assertEqual(
            tensors_shape,
            (self.input_dim, self.output_dim, self.number_attention_layers),
            f"Expected shape: ({self.input_dim}, {self.output_dim}, {self.number_attention_layers}), got {tensors_shape}",
        )

        tenslora_adapter._validate_all_tensor_shapes(
            tensors,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="depth",
            tensor_fac="tucker",
        )

        # With a wrong tensor_method
        with self.assertRaises(AssertionError):
            tenslora_adapter._validate_all_tensor_shapes(
                tensors,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                number_attention_layers=self.number_attention_layers,
                n_components=self.n_components,
                tensor_method="qkv",
                tensor_fac="tucker",
                number_heads=self.number_heads,
            )

    def test_create_all_tensors_att_qkv(self):
        self.n_components = [2, 2, 2, 2]
        tensors = create_all_tensor_adapters(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="att_qkv",
            tensor_fac="tucker",
            number_heads=self.number_heads,
        )
        tensors_shape = tensors[0].tensor_lora_module.fold_tensor().shape
        self.assertEqual(
            tensors_shape,
            (self.input_dim, self.output_dim // self.number_heads, self.number_heads, 3),
            f"Expected shape: ({self.input_dim}, {self.output_dim // self.number_heads}, {self.number_heads}, 3), got {tensors_shape}",
        )

        tenslora_adapter._validate_all_tensor_shapes(
            tensors,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="att_qkv",
            tensor_fac="tucker",
            number_heads=self.number_heads,
        )

        # With a wrong tensor_method
        with self.assertRaises(AssertionError):
            tenslora_adapter._validate_all_tensor_shapes(
                tensors,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                number_attention_layers=self.number_attention_layers,
                n_components=self.n_components,
                tensor_method="att_depth",
                tensor_fac="tucker",
                number_heads=self.number_heads,
            )

    def test_create_all_tensors_att_depth(self):
        self.n_components = [2, 2, 2, 2]

        tensors = create_all_tensor_adapters(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="att_depth",
            tensor_fac="tucker",
            number_heads=self.number_heads,
        )
        tensors_shape = tensors[0].tensor_lora_module.fold_tensor().shape
        self.assertEqual(
            tensors_shape,
            (self.input_dim, self.output_dim // self.number_heads, self.number_heads, self.number_attention_layers),
            f"Expected shape: ({self.input_dim}, {self.output_dim // self.number_heads}, {self.number_heads}, {self.number_attention_layers}), got {tensors_shape}",
        )

        tenslora_adapter._validate_all_tensor_shapes(
            tensors,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="att_depth",
            tensor_fac="tucker",
            number_heads=self.number_heads,
        )

        # With a wrong tensor_method
        with self.assertRaises(AssertionError):
            tenslora_adapter._validate_all_tensor_shapes(
                tensors,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                number_attention_layers=self.number_attention_layers,
                n_components=self.n_components,
                tensor_method="att_qkv",
                tensor_fac="tucker",
                number_heads=self.number_heads,
            )

    def test_create_all_tensors_qkv_depth(self):
        self.n_components = [2, 2, 2, 2]
        tensors = create_all_tensor_adapters(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="qkv_depth",
            tensor_fac="tucker",
        )
        tensors_shape = tensors.tensor_lora_module.fold_tensor().shape

        self.assertEqual(
            tensors_shape,
            (self.input_dim, self.output_dim, 3, self.number_attention_layers),
            f"Expected shape: ({self.input_dim}, {self.output_dim}, 3, {self.number_attention_layers}), got {tensors_shape}",
        )

        tenslora_adapter._validate_all_tensor_shapes(
            tensors,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="qkv_depth",
            tensor_fac="tucker",
        )

        # With a wrong tensor_method
        with self.assertRaises(AssertionError):
            tenslora_adapter._validate_all_tensor_shapes(
                tensors,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                number_attention_layers=self.number_attention_layers,
                n_components=[2, 2, 2, 2, 2],
                tensor_method="att_qkv_depth",
                tensor_fac="tucker",
                number_heads=self.number_heads,
            )

    def test_create_all_tensors_att_qkv_depth(self):
        self.n_components = [2, 2, 2, 2, 2]
        tensors = create_all_tensor_adapters(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="att_qkv_depth",
            tensor_fac="tucker",
            number_heads=self.number_heads,
        )
        tensors_shape = tensors.tensor_lora_module.fold_tensor().shape
        self.assertEqual(
            tensors_shape,
            (self.input_dim, self.output_dim // self.number_heads, self.number_heads, 3, self.number_attention_layers),
            f"Expected shape: ({self.input_dim}, {self.output_dim // self.number_heads}, {self.number_heads}, 3, {self.number_attention_layers}), got {tensors_shape}",
        )

        tenslora_adapter._validate_all_tensor_shapes(
            tensors,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            number_attention_layers=self.number_attention_layers,
            n_components=self.n_components,
            tensor_method="att_qkv_depth",
            tensor_fac="tucker",
            number_heads=self.number_heads,
        )

        # With a wrong tensor_method
        with self.assertRaises(AssertionError):
            tenslora_adapter._validate_all_tensor_shapes(
                tensors,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                number_attention_layers=self.number_attention_layers,
                n_components=[2, 2, 2],
                tensor_method="att",
                tensor_fac="tucker",
                number_heads=self.number_heads,
            )


class TestTensLoRADropout(unittest.TestCase):
    def setUp(self):
        self.input_dim = 16
        self.output_dim = 16
        self.n_components = [2, 2, 2]
        self.tensor_method = "att"
        self.tensor_fac = "tucker"
        self.dropout_prob = 0.1
        self.number_heads = 4
        self.number_attention_layers = 3

        # Create an instance of TensLoRA_adapter
        self.adapter = tenslora_adapter.TensLoRA_adapter(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            n_components=self.n_components,
            tensor_method=self.tensor_method,
            tensor_fac=self.tensor_fac,
            dropout_prob=self.dropout_prob,
            number_heads=self.number_heads,
            number_attention_layers=self.number_attention_layers,
        )

    def test_dropout_applied_in_training_mode(self):
        # Set the adapter to training mode
        self.adapter.train()
        self.adapter.forward(torch.ones((self.input_dim, self.output_dim)))

        self.assertTrue(self.adapter.tensor_lora_module._check_dropout_applied(), "Dropout should be applied in training mode.")

    def test_dropout_removed_in_eval_mode(self):
        # Set the adapter to training mode and apply dropout
        self.adapter.train()
        self.adapter.forward(torch.ones((self.input_dim, self.output_dim)))

        self.assertTrue(self.adapter.tensor_lora_module._check_dropout_applied(), "Dropout should be applied in training mode.")

        # Switch to evaluation mode and remove dropout
        self.adapter.eval()
        self.adapter.forward(torch.ones((self.input_dim, self.output_dim)))

        # self.adapter.factorized_tensor = tltorch.tensor_hooks.remove_tensor_dropout(self.adapter.factorized_tensor)
        self.assertFalse(
            self.adapter.tensor_lora_module._check_dropout_applied(),
            "Dropout should be removed in evaluation mode.",
        )

    def test_dropout_not_applied_in_eval_mode(self):
        # Set the adapter to evaluation mode
        self.adapter.eval()
        self.adapter.forward(torch.ones((self.input_dim, self.output_dim)))

        # Check that dropout is not applied
        self.assertFalse(
            self.adapter.tensor_lora_module._check_dropout_applied(),
            "Dropout should not be applied in evaluation mode.",
        )

    def test_dropout_applied_and_removed_correctly(self):
        # Set the adapter to training mode and apply dropout
        self.adapter.train()
        self.adapter.forward(torch.ones((self.input_dim, self.output_dim)))

        self.assertTrue(self.adapter.tensor_lora_module._check_dropout_applied(), "Dropout should be applied in training mode.")

        # Switch to evaluation mode and remove dropout
        self.adapter.eval()
        self.adapter.forward(torch.ones((self.input_dim, self.output_dim)))

        self.assertFalse(
            self.adapter.tensor_lora_module._check_dropout_applied(),
            "Dropout should be removed in evaluation mode.",
        )

        # Switch back to training mode and reapply dropout
        self.adapter.train()
        self.adapter.forward(torch.ones((self.input_dim, self.output_dim)))

        self.assertTrue(self.adapter.tensor_lora_module._check_dropout_applied(), "Dropout should be reapplied in training mode.")


class TestTensorAdapterForward(unittest.TestCase):
    def test_validate_forward_tenslora_valid_att_none(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=512,
            output_dim=512,
            tensor_fac="tucker",
            n_components=[2, 2, 2],
            tensor_method="att",
            number_heads=8,
        )

        x = torch.zeros((self.mock_tenslora.input_dim, self.mock_tenslora.output_dim))
        self.mock_tenslora._validate_forward_tenslora(x, qkv=None, layer=None)

    def test_validate_forward_tenslora_valid_qkv_set(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=512,
            output_dim=512,
            tensor_fac="tucker",
            n_components=[2, 2, 2],
            tensor_method="qkv",
        )

        x = torch.zeros((self.mock_tenslora.input_dim, self.mock_tenslora.output_dim))
        self.mock_tenslora._validate_forward_tenslora(x, qkv=1, layer=None)

    def test_validate_forward_tenslora_valid_depth_set(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=512,
            output_dim=512,
            tensor_fac="tucker",
            n_components=[2, 2, 2],
            tensor_method="depth",
            number_attention_layers=20,
        )

        x = torch.zeros((self.mock_tenslora.input_dim, self.mock_tenslora.output_dim))
        self.mock_tenslora._validate_forward_tenslora(x, qkv=None, layer=1)

    def test_validate_forward_tenslora_valid_att_qkv_depth(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=512,
            output_dim=512,
            tensor_fac="tucker",
            n_components=[2, 2, 2, 2, 2],
            tensor_method="att_qkv_depth",
            number_heads=8,
            number_attention_layers=20,
        )

        x = torch.zeros((self.mock_tenslora.input_dim, self.mock_tenslora.output_dim))
        self.mock_tenslora._validate_forward_tenslora(x, qkv=1, layer=1)

    def test_validate_forward_tenslora_invalid_qkv(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=512,
            output_dim=512,
            tensor_fac="tucker",
            n_components=[2, 2, 2, 2, 2],
            tensor_method="att_qkv_depth",
            number_heads=8,
            number_attention_layers=20,
        )
        x = torch.zeros((self.mock_tenslora.input_dim, self.mock_tenslora.output_dim))
        with self.assertRaises(AssertionError):
            self.mock_tenslora._validate_forward_tenslora(x, qkv=None, layer=1)
        with self.assertRaises(AssertionError):
            self.mock_tenslora._validate_forward_tenslora(x, qkv=3, layer=1)
        with self.assertRaises(AssertionError):
            self.mock_tenslora._validate_forward_tenslora(x, qkv=-1, layer=1)
        with self.assertRaises(AssertionError):
            self.mock_tenslora._validate_forward_tenslora(x, qkv=1.2, layer=1)

    def test_validate_forward_tenslora_invalid_depth(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=512,
            output_dim=512,
            tensor_fac="tucker",
            n_components=[2, 2, 2, 2, 2],
            tensor_method="att_qkv_depth",
            number_heads=8,
            number_attention_layers=20,
        )
        x = torch.zeros((self.mock_tenslora.input_dim, self.mock_tenslora.output_dim))
        with self.assertRaises(AssertionError):
            self.mock_tenslora._validate_forward_tenslora(x, qkv=1, layer=None)
        with self.assertRaises(AssertionError):
            self.mock_tenslora._validate_forward_tenslora(x, qkv=1, layer=self.mock_tenslora.number_attention_layers)
        with self.assertRaises(AssertionError):
            self.mock_tenslora._validate_forward_tenslora(x, qkv=1, layer=-1)
        with self.assertRaises(AssertionError):
            self.mock_tenslora._validate_forward_tenslora(x, qkv=1, layer=1.2)


class TestTensorMethodMatrixUpdateForward(unittest.TestCase):
    def setUp(self):
        self.input_dim = 512
        self.output_dim = 512
        self.ones = torch.ones((self.input_dim, self.output_dim))

    def test_tensor_method_att(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tensor_fac="tucker",
            n_components=[2, 2, 2],
            tensor_method="att",
            number_heads=8,
        )

        self.mock_tenslora.forward(self.ones)

    def test_tensor_method_qkv(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tensor_fac="tucker",
            n_components=[2, 2, 2],
            tensor_method="qkv",
        )

        self.mock_tenslora.forward(self.ones, qkv=0)

    def test_tensor_method_depth(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tensor_fac="tucker",
            n_components=[2, 2, 2],
            tensor_method="depth",
            number_attention_layers=20,
        )

        self.mock_tenslora.forward(self.ones, layer=0)

    def test_tensor_method_att_qkv(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tensor_fac="tucker",
            n_components=[2, 2, 2, 2],
            tensor_method="att_qkv",
            number_heads=8,
        )

        self.mock_tenslora.forward(self.ones, qkv=0)

    def test_tensor_method_att_depth(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tensor_fac="tucker",
            n_components=[2, 2, 2, 2],
            tensor_method="att_depth",
            number_heads=8,
            number_attention_layers=20,
        )

        self.mock_tenslora.forward(self.ones, layer=0)

    def test_tensor_method_qkv_depth(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tensor_fac="tucker",
            n_components=[2, 2, 2, 2],
            tensor_method="qkv_depth",
            number_heads=8,
            number_attention_layers=20,
        )

        self.mock_tenslora.forward(self.ones, qkv=0, layer=0)

    def test_tensor_method_att_qkv_depth(self):
        self.mock_tenslora = tenslora_adapter.TensLoRA_adapter(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            tensor_fac="tucker",
            n_components=[2, 2, 2, 2, 2],
            tensor_method="att_qkv_depth",
            number_heads=8,
            number_attention_layers=20,
        )

        self.mock_tenslora.forward(self.ones, qkv=0, layer=0)


if __name__ == "__main__":
    unittest.main()
