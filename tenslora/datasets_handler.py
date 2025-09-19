from functools import partial
import os
import numpy as np
import cv2
from PIL import Image

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import CIFAR10, FGVCAircraft, DTD, EuroSAT
from transformers import ViTImageProcessorFast
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split

CACHE_DIR = ".cache"


class CIFAR10Dataset:
    def __init__(
        self,
        batch_size=32,
        seed = 0, 
        data_path="/Brain/public/datasets/cifar_10",
        model_path="/Brain/public/models/google/vit-base-patch16-224",
    ):
        # Load CIFAR-10 dataset and apply ViT image processor
        processor = ViTImageProcessorFast.from_pretrained(model_path)

        def transform(img):
            return processor(img, return_tensors="pt")["pixel_values"].squeeze(0)

        c10train = CIFAR10(data_path, train=True, download=False, transform=transform)
        c10test = CIFAR10(data_path, train=False, download=False, transform=transform)

        trainloader = DataLoader(c10train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        testloader = DataLoader(c10test, batch_size=batch_size * 2, num_workers=4, pin_memory=True)

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10
        self.name = "CIFAR10"


# Datasets


def _final_post_process(example, tokenizer):
    tokens = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": example["label"],
    }


def glue_collate(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
    }


def get_glue_dataset(tokenizer, batch_size: int, subset: str, test: bool):
    train_dataset = load_dataset("glue", subset, split="train", cache_dir=CACHE_DIR).shuffle(seed=42)
    validation_dataset = load_dataset("glue", subset, split="validation", cache_dir=CACHE_DIR)
    if test:
        train_dataset = train_dataset.select(range(500))
        validation_dataset = validation_dataset.select(range(128))

    if subset == "cola":
        train_dataset = train_dataset.rename_column("sentence", "text")
        validation_dataset = validation_dataset.rename_column("sentence", "text")
        num_classes = 2
    elif subset == "sst2":
        train_dataset = train_dataset.rename_column("sentence", "text")
        validation_dataset = validation_dataset.rename_column("sentence", "text")
        num_classes = 2
    elif subset == "mrpc":
        #Create a new column with the two sentences concatenated
        def concat_sentences(example):
            return {"text": example["sentence1"] + "</s></s>" + example["sentence2"]}
        train_dataset = train_dataset.map(concat_sentences)
        validation_dataset = validation_dataset.map(concat_sentences)
        num_classes = 2
    elif subset == "qnli":
        def concat_sentences(example):
            return {"text": example["question"] + "</s></s>" + example["sentence"]}
        train_dataset = train_dataset.map(concat_sentences)
        validation_dataset = validation_dataset.map(concat_sentences)
        num_classes = 2
    else:
        raise NotImplementedError(f"Dataset {subset} not implemented for this script.")

    final_post_process = partial(_final_post_process, tokenizer=tokenizer)
    train_dataset = train_dataset.map(final_post_process, remove_columns=["text"], batched=True).remove_columns(["idx"])
    validation_dataset = validation_dataset.map(final_post_process, remove_columns=["text"], batched=True).remove_columns(["idx"])

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size * 2,
        num_workers=8,
        pin_memory=True,
    )

    return train_dataloader, validation_dataloader, num_classes


class CUBDataset(Dataset):
    def __init__(
        self,
        batch_size=32,
        seed = 0,
        data_path="/Brain/private/r17bensa/CUB_200_2011",
    ):

        resize_size = 300
        input_size = 224
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        train_transforms = transforms.Compose([
                            transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                            transforms.CenterCrop((input_size, input_size)),
                            transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
                            ], p=0.5),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize
                    ])
        test_transforms = transforms.Compose([
                            transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                            transforms.CenterCrop((input_size, input_size)),
                            transforms.ToTensor(),
                            normalize
                    ])

        train_set = CUB(train_transforms, istrain=True, datapath=data_path)
        trainloader = DataLoader(train_set, num_workers=4, shuffle=True, batch_size=batch_size, pin_memory=True)
        
        test_set = CUB(test_transforms, istrain=False, datapath=data_path)
        testloader = DataLoader(test_set, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 200
        self.name = "CUB"

class CUB(Dataset):

    def __init__(self, 
                transforms,
                istrain,
                datapath):

        self.datapath = datapath
        self.transforms = transforms

        train_test_split = open(os.path.join(datapath, "train_test_split.txt"), "r")
        train_test_split = train_test_split.readlines()
        corresp = open(os.path.join(datapath,"images.txt"), "r")
        corresp = corresp.readlines()
        self.train_images = []
        self.test_images = []
        for i in range(len(train_test_split)) : 
            x = train_test_split[i].replace("\n", "").split(" ")
            y = corresp[i].replace("\n", "").split(" ")
            if x[1] == '1' : 
                self.train_images.append(y[1].split("/")[1])
            else : 
                self.test_images.append(y[1].split("/")[1])
        self.data_infos = self.getDataInfo(os.path.join(datapath, "images/"), istrain)


    def getDataInfo(self, root, istrain):
        data_infos = []
        folders = os.listdir(root)
        folders.sort() # sort by alphabet
        print("[dataset] class number:", len(folders))
        for class_id, folder in enumerate(folders):
            files = os.listdir(root+folder)
            for file in files:
                data_path = root+folder+"/"+file
                if istrain & (file in self.train_images) : 
                    data_infos.append({"path":data_path, "label":class_id})
                if (not istrain) & (file in self.test_images) : 
                    data_infos.append({"path":data_path, "label":class_id})
        print(istrain, len(data_infos))
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.
        
        img = Image.fromarray(img)
        if self.transforms is not None : 
            img = self.transforms(img)

        return img, label


class AIRCRAFTDataset(Dataset):
    def __init__(
        self,
        batch_size=32,
        seed = 0,
        data_path="/Brain/public/datasets/",
    ):

        resize_size = 300
        input_size = 224
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        train_transforms = transforms.Compose([
                            transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                            transforms.CenterCrop((input_size, input_size)),
                            transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
                            ], p=0.5),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                    ])
        test_transforms = transforms.Compose([
                            transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                            transforms.CenterCrop((input_size, input_size)),
                            transforms.ToTensor(),
                    ])

        train_set = FGVCAircraft(root = data_path, 
             split = 'trainval', transform = train_transforms, annotation_level = 'variant')
        test_set = FGVCAircraft(root = data_path, 
             split = 'test', transform = test_transforms,  annotation_level = 'variant')

        trainloader = torch.utils.data.DataLoader(train_set, num_workers=4, shuffle=True, batch_size=batch_size, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_set, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 100
        self.name = "Aircraft"


class DTDDataset(Dataset):
    def __init__(
        self,
        batch_size=32,
        seed = 0,
        data_path="/Brain/public/datasets",
    ):

        resize_size = 224
        input_size = 224
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        train_transforms = transforms.Compose([
                        transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                        #transforms.CenterCrop((input_size, input_size)),
                        #transforms.RandomApply([
                        #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
                        #], p=0.5),
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                ])
        test_transforms = transforms.Compose([
                        transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                        #transforms.CenterCrop((input_size, input_size)),
                        transforms.ToTensor(),
                ])

        train_set = DTD(root = data_path, split = "train", transform = train_transforms)

        train_set, _ = random_split(train_set, [1000, len(train_set) - 1000])

        val_set =  DTD(root = data_path, split = "val", transform = train_transforms)
        test_set =  DTD(root = data_path, split = "test", transform = test_transforms)
        #train_set_merged = ConcatDataset([train_set, val_set])

        trainloader = torch.utils.data.DataLoader(train_set, num_workers=4, shuffle=True, batch_size=batch_size, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_set, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 47
        self.name = "DTD"



class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)


class EUROSATDataset(Dataset):
    def __init__(
        self,
        batch_size=32,
        seed = 0,
        data_path="/Brain/public/datasets/",
    ):

        resize_size = 300
        input_size = 224
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        train_transforms = transforms.Compose([
                            transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                            transforms.CenterCrop((input_size, input_size)),
                            transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)
                            ], p=0.5),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                    ])
        test_transforms = transforms.Compose([
                            transforms.Resize((resize_size, resize_size), Image.BILINEAR),
                            transforms.CenterCrop((input_size, input_size)),
                            transforms.ToTensor(),
                    ])

        dataset = EuroSAT(root = data_path)
        print(len(dataset))
        train_set, test_set = random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
        train_set = TransformedSubset(train_set, transform=train_transforms)
        test_set = TransformedSubset(test_set, transform=test_transforms)

        trainloader = torch.utils.data.DataLoader(train_set, num_workers=4, shuffle=True, batch_size=batch_size, pin_memory=True)
        testloader = torch.utils.data.DataLoader(test_set, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10
        self.name = "Eurosat"
