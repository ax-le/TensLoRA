"""
Standalone script to train with tensorlora on ViT models.
Does not support training on multiple GPUs.
"""

import copy
import random
from datetime import datetime
import typer

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from transformers import ViTForImageClassification
import wandb

import tenslora.adapters.add_lora_to_model as add_lora_to_model
from  tenslora.datasets_handler import AIRCRAFTDataset, CUBDataset, CIFAR10Dataset, DTDDataset, EUROSATDataset
from tenslora.utils.parameter_count import count_trainable_parameters, predict_tenslora_parameters

DEFAULT_MODEL_PATH = "/Brain/public/models/google/vit-base-patch16-224"

def get_classifier(
    device,
    num_classes,
    lora_type,
    n_components,
    tensor_method=None,
    dropout_prob=0.0,
    init_from_saved_tensors=False,
    tensor_path=None,
    tensor_persisted_name=None,
    scaling=1,
    seed=0,
    model_path=DEFAULT_MODEL_PATH,
    *args,
    **kwargs,
):
    if model_path is None:
        raise ValueError("model_path must be provided to load the base ViT model.")
    
    # Load the model
    model = ViTForImageClassification.from_pretrained(model_path)

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters.")

    if lora_type == "tenslora" and tensor_method is None:
        raise ValueError("Tensor method must be specified when using tenslora.")

    # Apply a LoRA to the model
    lora_model = add_lora_to_model.lora_router(
        model=model,
        lora_type=lora_type,
        n_components=n_components,
        model_type="vit",
        input_dim=768,  # ViT base patch16 input dimension
        output_dim=768,  # ViT base patch16 output dimension
        scaling=scaling,
        dropout_prob=dropout_prob,
        init_from_saved_tensors=init_from_saved_tensors,
        tensor_path=tensor_path,
        tensor_persisted_name=tensor_persisted_name,
        seed=seed,
        *args,
        **kwargs,
    )

    # Replace final classifier
    if init_from_saved_tensors == False:
        if lora_type == "lora_hf":
            lora_model.base_model.model.classifier = nn.Linear(768, num_classes)
        else:
            lora_model.classifier = nn.Linear(768, num_classes)
    else:
        # Load the classifier state dict from the saved tensors
        classifier_weights = torch.load(f"{tensor_path}/{tensor_persisted_name}_classifier.pth", weights_only=False)
        lora_model.classifier = classifier_weights

    # Move the model to the device
    return lora_model.to(device)


def save_adapters_and_classifier(lora_type, tensor_method, model, tensor_path, tensor_persisted_name):
    add_lora_to_model.save_tensors_from_model(
        lora_type,
        tensor_method,
        model,
        model_type="vit",
        tensor_path=tensor_path,
        tensor_persisted_name=tensor_persisted_name,
    )
    torch.save(model.classifier, f"{tensor_path}/{tensor_persisted_name}_classifier.pth")


def train_one_epoch(model, trainloader, optimizer, scheduler, criterion, device):
    """One training epoch"""
    # Set the model to training mode
    model.train()
    running_loss = 0.0

    # Iterate over the data
    for inputs, labels in tqdm(trainloader, desc="Training", leave=False):
        # Move data to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update the running loss
        running_loss += loss.item() * inputs.size(0)

    # Return the loss
    normalized_loss = running_loss / len(trainloader.dataset)
    return normalized_loss


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    """Model evaluation"""
    # Set the model to evaluation mode
    model.eval()
    total_loss = 0.0
    correct = 0

    # Iterate over the data
    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        # Move data to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)

        # Compute the loss and accuracy
        total_loss += loss.item() * inputs.size(0)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    # Return the average loss and accuracy
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main(
    # LoRA parameters
    lora_type: str = typer.Argument(..., help="Type of LoRA to use"),
    scaling: float = typer.Option(1.0, help="LoRA alpha value for scaling the LoRA weights"),
    # TensLoRA parameters
    tensor_method: str = typer.Option(None, help="Method for tensor decomposition (e.g., 'att', 'qkv', 'depth')"),
    tensor_fac: str = typer.Option(None, help="Tensor factorization method for tenslora (cp or tucker)"),
    tensor_init: str = typer.Option("orthogonal", help="Tensor initialization method for tenslora (orthogonal, normal, or kaming_uniform)"),

    n_components: str = typer.Option("4",help="Number of components for TensLoRA. Expected to be a string, to pass either int or list of int. Use underscores to separate multiple components (e.g., '4_8_16')"),
    # Dataset
    dataset: str = typer.Option("cola", help="Dataset to use for training."),
    # Training parameters
    lr: float = typer.Option(5e-4, help="Learning rate for the optimizer"),
    n_epochs: int = typer.Option(10, help="Number of epochs to train"),
    batch_size: int = typer.Option(64, help="Batch size"),
    weight_decay: float = typer.Option(0.01, help="Weight decay"),
    dropout_prob: float = typer.Option(0.0, help="Dropout probability for LoRA layers"),
    # Seed and scaling
    seed = typer.Option(None, help="Random seed for reproducibility"),

    # Other parameters
    use_wandb: bool = typer.Option(True, help="Enable Wandb"),
    save_directory: str = typer.Option(None, help="Directory to save tensors and model checkpoints"),
):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init wandb if needed
    if use_wandb:
        wandb.login()

    # Seed
    if seed is None:  # Fix the seed anyway
        print("No seed provided, using default seed 0.")
        seed = 0
    elif type(seed) is not int:
        raise ValueError(f"Seed must be an integer, got {type(seed)} instead.")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    # Set the number of components in LoRA/TensLoRA
    if type(n_components) is str:
        n_components_str = copy.deepcopy(n_components)
        # Parse n_components as a list of ints if underscores are present, else as a single int
        n_components = [int(x) for x in n_components.split("_")] if "_" in n_components else int(n_components)
    else:
        raise ValueError(f"Unsupported type for n_components: {type(n_components)}. Expected str or int.")

    ## Dataset setup
    # Dataset parameters
    datasets = {"cifar" : CIFAR10Dataset, "cub": CUBDataset, "aircraft" : AIRCRAFTDataset, "dtd": DTDDataset, 'eurosat': EUROSATDataset}
    assert dataset in datasets, f"Dataset {dataset} not found. Available datasets: {list(datasets.keys())}"
    dataset_class = datasets[dataset]

    # Instantiate the dataset
    dataset = dataset_class(batch_size=batch_size, seed=seed)
    trainloader = dataset.trainloader
    testloader = dataset.testloader
    num_classes = dataset.num_classes
    dataset_name = dataset.name

    print("Dataset loaded successfully!")

    # Get the number of batches, for the lr scheduler
    n_batches = len(trainloader)

    # Print experiment information
    print(
        f"{dataset_name} : Training params : lr: {lr} | n_epochs: {n_epochs} | weight_decay: {weight_decay} | dropout_prob: {dropout_prob}",
    )

    ## Create the model, with LoRA
    model = get_classifier(
        device,
        num_classes=num_classes,
        lora_type=lora_type,
        dropout_prob=dropout_prob,
        n_components=n_components,
        tensor_method=tensor_method,
        tensor_fac=tensor_fac,
        tensor_init=tensor_init,
        init_from_saved_tensors=False,
        scaling=scaling,
        seed=seed,
    )

    # Print the LoRA type and parameters
    print_tensor = f" - {tensor_method}_{tensor_fac}_{tensor_init}" if tensor_method else ""
    print(f"{lora_type}{print_tensor} | n_components: {n_components_str} | seed: {seed} | scaling: {scaling}")

    # Get the number of trainable parameters,
    adapters_trainable_params, trainable_params, all_params = count_trainable_parameters(model)

    if lora_type == "tenslora":  # Check that the parameters are actually appropriate for TensLoRA
        count_tenslora_params = predict_tenslora_parameters(
            method=tensor_method,
            tenslora_set_ranks=n_components,
            hidden_dim=768,
            layer=12,
            num_heads=12,
        )

        assert count_tenslora_params == adapters_trainable_params, (
            f"Counted TensLoRA parameters ({count_tenslora_params}) do not match adapter parameters ({adapters_trainable_params})."
        )

    print(
        f"Trainable params: {trainable_params} | All params: {all_params} | % of trainable: {100 * trainable_params / all_params:.3f}",
    )
    print(
        f"Adapter only params: {adapters_trainable_params} | % of trainable: {100 * adapters_trainable_params / all_params:.3f}",
    )

    ## Optimization setup
    # Optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=n_batches * n_epochs,
        eta_min=1e-6,  # Minimum learning rate
        last_epoch=-1,  # Start from the beginning
    )

    print("Optimizer and scheduler initialized.")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    if use_wandb:
        # wandb init
        wandb_job_name = f"{lora_type}"
        wandb_job_name += f"_{tensor_method}" if tensor_method else ""
        wandb_job_name += f"_{tensor_fac}" if tensor_fac else ""
        wandb_job_name += f"_n_components{n_components_str}_lr{lr}_wd{weight_decay}"

        date = datetime.now().strftime("%Y%m%d_%H%M%S")

        wandb_run = wandb.init(
            project=f"tenslora_{dataset_name}",
            name=f"{wandb_job_name}_{date}",
            config={
                "learning_rate": lr,
                "n_epochs": n_epochs,
                "weight_decay": weight_decay,
                "dropout_prob": dropout_prob,
                "lora_type": lora_type,
                "tensor_method": tensor_method,
                "tensor_fac": tensor_fac,
                "n_components_str": n_components_str,
                # Additional metadata
                "dataset": dataset_name,
                "epochs": n_epochs,
                "batch_size": batch_size,
                "model": "ViT-base-patch16-224",
                "trainable_params": trainable_params,
                "pourcentage_trainable": 100 * trainable_params / all_params,
            },
        )

    # Training state initialization
    start_epoch = 1
    best_acc = 0.0

    # Training loop
    for epoch in range(start_epoch, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")

        train_loss = train_one_epoch(model, trainloader, optimizer, scheduler, criterion, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        # Update best performance
        best_acc = max(best_acc, test_acc)

        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc * 100:.2f}% (Best: {best_acc * 100:.2f}%)")

        if use_wandb:
            # Log metrics to wandb
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "best_accuracy": best_acc,
                    "learning_rate": scheduler.get_last_lr()[0],
                },
            )

        # Save the model and tensors at the end of each epoch
        if save_directory:
            save_adapters_and_classifier(lora_type, tensor_method, model, save_directory, f"tensors_{epoch}_{tensor_method}")

    if use_wandb:
        # Finish the run and upload any remaining data.
        wandb_run.finish()


if __name__ == "__main__":
    typer.run(main)
