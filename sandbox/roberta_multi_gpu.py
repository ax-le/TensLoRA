"""
Standalone script to train with tensorlora on ROBERTa models.
"""

import builtins
import copy
import os
import random
from functools import partial

import idr_torch
import numpy as np
import torch
import typer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support
from transformers import AutoTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, get_scheduler

from tenslora.utils.parameter_count import count_trainable_parameters, predict_tenslora_parameters
import wandb
from tenslora.adapters.add_lora_to_model import lora_router


def print(*args, **kwargs):
    """Only print on rank 0."""
    if idr_torch.rank == 0:
        builtins.print(*args, **kwargs)


CACHE_DIR = ".cache"
HELP_PANEL_NAME_1 = "Training Parameters"
HELP_PANEL_NAME_2 = "LORA Parameters"

MODEL_NAME = "FacebookAI/roberta-base"


# Datasets


def _final_post_process(example, tokenizer):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": example["label"],
    }


def get_glue_dataset(tokenizer, subset: str, test: bool):
    train_dataset = load_dataset("glue", subset, split="train", cache_dir=CACHE_DIR).shuffle(seed=42)
    validation_dataset = load_dataset("glue", subset, split="validation", cache_dir=CACHE_DIR)
    if test:
        train_dataset = train_dataset.select(range(500))
        validation_dataset = validation_dataset.select(range(128))

    if subset == "cola":
        train_dataset = train_dataset.rename_column("sentence", "text")
        validation_dataset = validation_dataset.rename_column("sentence", "text")
        num_classes = 2
    else:
        raise NotImplementedError(f"Dataset {subset} not implemented for this script.")

    final_post_process = partial(_final_post_process, tokenizer=tokenizer)
    train_dataset = train_dataset.map(final_post_process, remove_columns=["text"], batched=True).remove_columns(["idx"])
    validation_dataset = validation_dataset.map(final_post_process, remove_columns=["text"], batched=True).remove_columns(["idx"])

    return train_dataset, validation_dataset, num_classes


# Wrapper logic


def get_classifier(
    num_classes: int,
    lora_type: str,
    n_components,
    tensor_method: str = None,
    seed: int = 0,
    scaling: float = 1.0,
):
    model: RobertaForSequenceClassification = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        problem_type="single_label_classification",
        cache_dir=CACHE_DIR,
    )

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters.")

    print("Using LoRA type:", lora_type)

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
        seed=seed,
        scaling=scaling,
        ## If you need to change these parameters, uncomment them:
        # dropout_prob=0.0,
        # init_from_saved_tensors=False,
        # tensor_path=None,
        # tensor_persisted_name=None,
        **kwargs_,
    )

    return lora_model


# Metrics


def get_compute_metrics(num_classes: int):
    def _compute_metrics(eval_preds):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids[1]

        preds = preds.argmax(axis=1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="macro" if num_classes > 2 else "binary",
        )
        mcc = matthews_corrcoef(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "mcc": mcc,
        }

    return _compute_metrics


def main(
    # LoRA parameters
    lora_type: str = typer.Argument(..., help="Type of LoRA to use"),
    # Dataset
    dataset: str = typer.Option("cola", help="Dataset to use for training. Options: 'tldr', 'xsum'"),
    # Training parameters
    lr: float = typer.Option(5e-4, help="Learning rate for the optimizer"),
    n_epochs: int = typer.Option(1, help="Number of epochs to train"),
    n_components: str = typer.Option(
        "4",
        help="Number of components for TensLoRA. Expected to be a string, to pass either int or list of int. "
        "Use underscores to separate multiple components (e.g., '4_8_16')",
    ),
    batch_size: int = typer.Option(64, help="Batch size per GPU"),
    test: bool = typer.Option(False, help="Run in test mode"),
    run_name: str = typer.Option("tensorlora-llm", help="Run name for logging"),
    seed: int = typer.Option(0, help="Random seed for reproducibility"),
    scaling: float = typer.Option(1.0, help="LoRA alpha value for scaling the LoRA weights"),
    # Tensor parameters
    tensor_method: str = typer.Option(None, help="Method for tensor decomposition (e.g., 'att', 'qkv', 'depth')"),
):
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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        use_fast=True,
    )

    train_dataset, validation_dataset, num_classes = get_glue_dataset(tokenizer, dataset, test=test)
    print("Dataset loaded successfully!")

    if type(n_components) is str:
        n_components_str = copy.deepcopy(n_components)
        # Parse n_components as a list of ints if underscores are present, else as a single int
        n_components = [int(x) for x in n_components.split("_")] if "_" in n_components else int(n_components)
    else:
        raise ValueError(f"Unsupported type for n_components: {type(n_components)}. Expected str.")

    model = get_classifier(
        num_classes,
        lora_type,
        n_components=n_components,
        tensor_method=tensor_method,
        seed=seed,
        scaling=scaling,
    )
    print("Model loaded successfully!")

    # Print the LoRA type and parameters
    print_tensor = f"{tensor_method}_Tucker_orthogonal" if tensor_method else ""
    print(
        f"LoRA / TensLoRA parameters: {lora_type} - {print_tensor} | "
        f"n_components: {n_components_str} | seed: {seed} | scaling: {scaling}",
    )

    txt = "hello"
    inputs = tokenizer(txt, return_tensors="pt").input_ids
    outputs = model(inputs, output_hidden_states=True, return_dict=True)
    print("Model output:", outputs.hidden_states[-1])
    # Should return:
    # Model output: tensor([[[-0.0712,  0.0839,  0.0174,  ..., -0.0752, -0.0725, -0.0115],
    #     [-0.0242, -0.2140,  0.1199,  ..., -0.3125, -0.2366,  0.0626],
    #     [-0.0778,  0.0813, -0.0014,  ..., -0.1284, -0.0861, -0.0517]]],
    #   grad_fn=<NativeLayerNormBackward0>)

    n_samples = len(train_dataset) * n_epochs
    n_gpus = idr_torch.world_size
    n_steps = n_samples // n_gpus // batch_size
    warmup_ratio = 0.1
    num_decay_steps = min(500, int(n_steps * 0.15))

    # Collect parameters for training
    parameters = []
    for name, param in model.named_parameters():
        if "lora" in name or "tenslora" in name or "classifier" in name:
            param.requires_grad = True  # Ensure LoRA parameters are trainable
            parameters.append(param)
            print(f"Parameter {name} is trainable with shape {param.shape}")

        else:
            param.requires_grad = False

    # Count parameters
    adapter_params, trainable_params, all_params = count_trainable_parameters(model)
    non_trainable_params = all_params - trainable_params

    if lora_type == "tenslora":  # Check that the parameters are actually appropriate for TensLoRA
        count_tenslora_params = predict_tenslora_parameters(
            method=tensor_method,
            tenslora_set_ranks=n_components,
            hidden_dim=768,
            layer=12,
            num_heads=12,
        )

        assert count_tenslora_params == adapter_params, (
            f"Counted TensLoRA parameters ({count_tenslora_params}) do not match adapter parameters ({adapter_params})."
        )

    print(
        f"Trainable parameters: {trainable_params:,}, Non-trainable parameters: {non_trainable_params:,},"
        f"(fraction train / total: {trainable_params / all_params:.2%})",
        f"\nOnly TensLoRA adapters trainable parameters: {adapter_params:,},",
    )

    if idr_torch.rank == 0:
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
                "n_samples": n_samples,
                "n_steps": n_steps,
                "warmup_ratio": warmup_ratio,
                "num_decay_steps": num_decay_steps,
                "trainable_parameters": trainable_params,
                "non_trainable_parameters": non_trainable_params,
                "fraction_trainable": trainable_params / all_params,
                "model_name": MODEL_NAME,
            },
        )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=lr,
        fused=True,
    )

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(n_steps * warmup_ratio),
        num_training_steps=n_steps,
    )

    training_args = TrainingArguments(
        # eval
        eval_strategy="steps",
        eval_steps=1/n_epochs if not test else 0.5,
        label_names=["input_ids", "labels"],
        # memory
        per_device_train_batch_size=batch_size,
        bf16=True,
        tf32=True,
        # training
        local_rank=idr_torch.rank,
        ddp_backend="nccl",
        seed=seed,
        torch_compile=True,
        # optimizer
        learning_rate=lr,
        max_steps=n_steps,
        # scheduler
        warmup_ratio=warmup_ratio,
        # i/o
        logging_steps=0.01,  # log every 1% of steps (100 logged values)
        dataloader_num_workers=8,
        save_strategy="no",  # disable saving
        output_dir=os.path.join(CACHE_DIR, "tmp"),
        run_name=run_name,
    )

    trainer = Trainer(
        optimizers=(optimizer, scheduler),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        args=training_args,
        compute_metrics=get_compute_metrics(num_classes),
    )

    trainer.train()

    print("Training completed successfully!")

    trainer.evaluate()
    print("Last evaluation completed successfully!")


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism to avoid warnings
    typer.run(main)
