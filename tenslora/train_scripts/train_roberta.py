"""
Standalone script to train with tensorlora on ROBERTa models.
Does not support training on multiple GPUs.
"""

import copy
import os
import random
import typer
from tqdm import tqdm
import wandb

import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification, get_scheduler

from tenslora.adapters.add_lora_to_model import lora_router
from tenslora.datasets_handler import get_glue_dataset
from tenslora.utils.parameter_count import count_trainable_parameters, predict_tenslora_parameters

CACHE_DIR = ".cache"
HELP_PANEL_NAME_1 = "Training Parameters"
HELP_PANEL_NAME_2 = "LORA Parameters"

DEFAULT_MODEL_PATH = "/Brain/public/models/FacebookAI/roberta-base"

# Wrapper logic

def get_classifier(
        num_classes: int, 
        lora_type: str, 
        n_components, 
        tensor_method: str = None, 
        dropout_prob=0.0,
        init_from_saved_tensors=False,
        tensor_path=None,
        tensor_persisted_name=None,
        scaling=1,
        seed=0,
        model_path=DEFAULT_MODEL_PATH,
    ):
    model: RobertaForSequenceClassification = RobertaForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_classes,
        problem_type="single_label_classification",
        cache_dir=CACHE_DIR,
    )

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters.")

    if lora_type == "tenslora" and tensor_method is None:
        raise ValueError("Tensor method must be specified when using tenslora.")

    kwargs_ = (
        {"tensor_fac": "tucker", "tensor_init": "orthogonal", "tensor_method": tensor_method} if lora_type == "tenslora" else {}
    )

    lora_model = lora_router(
        model=model,
        lora_type=lora_type,
        n_components=n_components,
        model_type="roberta",
        input_dim=768,
        output_dim=768,
        scaling=scaling,
        ## If you need to change these parameters, uncomment them:
        # dropout_prob=dropout_prob,
        # init_from_saved_tensors=init_from_saved_tensors,
        # tensor_path=tensor_path,
        # tensor_persisted_name=tensor_persisted_name,
        seed=seed,
        **kwargs_,
    )

    return lora_model


def main(  # noqa: C901, PLR0912, PLR0915
    # LoRA parameters
    lora_type: str = typer.Argument(..., help="Type of LoRA to use"),
    scaling: float = typer.Option(1.0, help="LoRA alpha value for scaling the LoRA weights"),
    # TensLoRA parameters
    tensor_method: str = typer.Option(None, help="Method for tensor decomposition (e.g., 'att', 'qkv', 'depth')"),
    # tensor_fac: str = typer.Option(None, help="Tensor factorization method for tenslora (cp or tucker)"),
    # tensor_init: str = typer.Option("orthogonal", help="Tensor initialization method for tenslora (orthogonal, normal, or kaming_uniform)"),

    n_components: str = typer.Option("4",help="Number of components for TensLoRA. Expected to be a string, to pass either int or list of int. Use underscores to separate multiple components (e.g., '4_8_16')"),
    # Dataset
    dataset: str = typer.Option("cola", help="Dataset to use for training. Options: 'tldr', 'xsum'"),
    # Training parameters
    lr: float = typer.Option(5e-4, help="Learning rate for the optimizer"),
    n_epochs: int = typer.Option(10, help="Number of epochs to train"),
    batch_size: int = typer.Option(64, help="Batch size per GPU"),
    # weight_decay: float = typer.Option(0.01, help="Weight decay"),
    # dropout_prob: float = typer.Option(0.0, help="Dropout probability for LoRA layers"),
    seed = typer.Option(None, help="Random seed for reproducibility"),
    # Other parameters
    test: bool = typer.Option(False, help="Run in test mode"),
    run_name: str = typer.Option("tensorlora-llm", help="Run name for logging"),
    use_wandb: bool = typer.Option(True, help="Enable Wandb"),
):
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
        raise ValueError(f"Unsupported type for n_components: {type(n_components)}. Expected str.")

    ## Dataset setup
    # Dataset parameters
    train_dataloader, validation_dataloader, num_classes = get_glue_dataset(tokenizer, batch_size, dataset, test=test)
    print("Dataset loaded successfully!")

    # Print experiment information
    print(
        f"{dataset} : Training params : lr: {lr} | n_epochs: {n_epochs}",
    )

    ## Model setup
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_PATH,
        cache_dir=CACHE_DIR,
        use_fast=True,
    )

    model = get_classifier(
        num_classes,
        lora_type, 
        n_components=n_components, 
        tensor_method=tensor_method, 
        seed=seed, 
        scaling=scaling,
        # tensor_fac=tensor_fac,
        # tensor_init=tensor_init,
    )
    model.to("cuda")
    print("Model loaded successfully!")

    # Print the LoRA type and parameters
    print_tensor = f" - {tensor_method}_tucker_orthogonal" if tensor_method else ""
    print(f"{lora_type}{print_tensor} | n_components: {n_components_str} | seed: {seed} | scaling: {scaling}")

    # Quick test to see if the model works
    txt = "hello"
    inputs = tokenizer(txt, return_tensors="pt").input_ids
    outputs = model(inputs, output_hidden_states=True, return_dict=True)
    print("Model output:", outputs.hidden_states[-1])
    # Should return:
    # Model output: tensor([[[-0.0712,  0.0839,  0.0174,  ..., -0.0752, -0.0725, -0.0115],
    #     [-0.0242, -0.2140,  0.1199,  ..., -0.3125, -0.2366,  0.0626],
    #     [-0.0778,  0.0813, -0.0014,  ..., -0.1284, -0.0861, -0.0517]]],
    #   grad_fn=<NativeLayerNormBackward0>)

    ## Training setup
    # Training parameters
    n_steps = len(train_dataloader) * n_epochs
    warmup_ratio = 0.1
    num_decay_steps = min(500, int(n_steps * 0.15))

    # eval_ratio = 1/n_epochs
    log_ratio = 0.01

    log_every = max(1, int(n_steps * log_ratio))
    eval_every = max(1, len(train_dataloader))

    # Collect parameters for training
    parameters = []
    for name, param in model.named_parameters():
        if "lora" in name or "tenslora" in name or "classifier" in name:
            param.requires_grad = True  # Ensure LoRA parameters are trainable
            parameters.append(param)
            # print(f"Parameter {name} is trainable with shape {param.shape}")

        else:
            param.requires_grad = False

    # Count parameters
    adapters_trainable_params, trainable_params, all_params = count_trainable_parameters(model)
    non_trainable_params = all_params - trainable_params

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
    optimizer = torch.optim.AdamW(
        parameters,
        lr=lr,
        fused=True,
    )

    # Learning rate scheduler
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(n_steps * warmup_ratio),
        num_training_steps=n_steps,
    )

    print("Optimizer and scheduler initialized.")

    if use_wandb:
        wandb.init(
            project="tensorlora-llm",
            name=run_name,
            config={
                "learning_rate": lr,
                "n_components": n_components,
                "scaling": scaling,
                "seed": seed,
                "batch_size": batch_size,
                "lora_type": lora_type,
                "dataset": dataset,
                "n_steps": n_steps,
                "warmup_ratio": warmup_ratio,
                "num_decay_steps": num_decay_steps,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
                "fraction_trainable": trainable_params / all_params,
                "model_name": DEFAULT_MODEL_PATH,
            },
        )

    # Training loop
    step = 0
    pbar = tqdm(total=n_steps, desc="Training")

    best_acc = 0
    best_mcc = 0

    while step < n_steps:
        model.train()
        for batch in train_dataloader:
            inputs = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            outputs = model(inputs, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % log_every == 0:
                lr = scheduler.get_last_lr()[0]
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/learning_rate": lr,
                        },
                        step=step,
                    )
                pbar.set_postfix({"loss": loss.item(), "step": step, "lr": lr})

            if step % eval_every == 0:
                model.eval()
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for val_batch in validation_dataloader:
                        val_inputs = val_batch["input_ids"].to("cuda")
                        val_labels = val_batch["labels"].to("cuda")
                        val_attention_mask = val_batch["attention_mask"].to("cuda")

                        val_outputs = model(val_inputs, attention_mask=val_attention_mask)
                        logits = val_outputs.logits
                        preds = torch.argmax(logits, dim=-1)

                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(val_labels.cpu().numpy())

                accuracy = accuracy_score(all_labels, all_preds)
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
                mcc = matthews_corrcoef(all_labels, all_preds)

                if accuracy > best_acc:
                    best_acc = accuracy
                if mcc > best_mcc:
                    best_mcc = mcc

                print(
                    f"Step {step}: Eval Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}",
                    f" | Best Acc: {best_acc:.4f}, Best MCC: {best_mcc:.4f}",
                )

                if use_wandb:
                    wandb.log(
                        {
                            "eval/accuracy": accuracy,
                            "eval/precision": precision,
                            "eval/recall": recall,
                            "eval/f1": f1,
                            "eval/mcc": mcc,
                            "eval/best_accuracy": best_acc,
                            "eval/best_mcc": best_mcc,
                        },
                        step=step,
                    )

            pbar.update(1)
            step += 1
            if step >= n_steps:
                break

    pbar.close()
    print("Training completed successfully!")

if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism to avoid warnings
    typer.run(main)
