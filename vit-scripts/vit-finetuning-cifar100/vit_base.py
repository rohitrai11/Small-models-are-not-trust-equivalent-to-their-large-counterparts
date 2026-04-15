import os
import json
import torch
import torch.nn as nn
import evaluate
import numpy as np
import logging

from datasets import load_dataset
from torchvision.transforms import (
    Compose, Resize, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, RandomErasing
)
from transformers import (
    ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer, DefaultDataCollator
)

# =====================================================
# Logging Setup
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# =====================================================
# 1. Setup & Data (NO RESIZE)
# =====================================================

MODEL_NAME = "google/vit-base-patch16-224-in21k"

logger.info(f"Loading processor: {MODEL_NAME}")
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

mean, std = processor.image_mean, processor.image_std

logger.info("Creating image transforms (NO RESIZE)")

train_transforms = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(),
    ColorJitter(0.2, 0.2, 0.2),
    ToTensor(),
    Normalize(mean=mean, std=std),
    RandomErasing(p=0.25, value=0)
])

val_transforms = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=mean, std=std)
])

def apply_train_transforms(examples):
    images = [train_transforms(img.convert("RGB")) for img in examples["img"]]
    return {"pixel_values": images, "labels": examples["fine_label"]}

def apply_val_transforms(examples):
    images = [val_transforms(img.convert("RGB")) for img in examples["img"]]
    return {"pixel_values": images, "labels": examples["fine_label"]}


logger.info("Loading CIFAR-100 dataset")
dataset = load_dataset("cifar100")

logger.info("Applying dataset transforms")
train_ds = dataset["train"]
test_ds = dataset["test"]

train_ds.set_transform(apply_train_transforms)
test_ds.set_transform(apply_val_transforms)

# =====================================================
# 2. Model (NO INTERPOLATION)
# =====================================================

logger.info("Loading ViT model")

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=100,
    ignore_mismatched_sizes=True,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16
)

# Zero-init classifier
logger.info("Initializing zero-head classifier")

in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, 100)

with torch.no_grad():
    model.classifier.weight.zero_()
    model.classifier.bias.zero_()

# =====================================================
# 3. Training Arguments
# =====================================================

logger.info("Setting up training arguments")

training_args = TrainingArguments(
    output_dir="./vit-BASE-cifar100-nointerp",
    remove_unused_columns=False,
    num_train_epochs=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    weight_decay=0.0,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    bf16=True,
    tf32=True,
    max_grad_norm=1.0,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    dataloader_num_workers=2,
)

# =====================================================
# 4. Metrics
# =====================================================

logger.info("Loading accuracy metric")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    acc = metric.compute(predictions=predictions, references=labels)

    logger.info(f"Evaluation accuracy: {acc['accuracy']:.4f}")

    return acc

# =====================================================
# 5. Trainer (STANDARD, NO CUSTOM FORWARD)
# =====================================================

logger.info("Initializing Trainer")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics
)

# Resume logic
last_checkpoint = None

if os.path.isdir(training_args.output_dir):
    from transformers.trainer_utils import get_last_checkpoint

    last_checkpoint = get_last_checkpoint(training_args.output_dir)

    if last_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        logger.info("No checkpoint found, starting fresh training")

logger.info("Starting training")

trainer.train(resume_from_checkpoint=last_checkpoint)

logger.info("Training completed")

# =====================================================
# 6. Save Results
# =====================================================

best_path = "./vit-base-cifar100-nointerp-best"

logger.info(f"Saving best model to: {best_path}")

trainer.save_model(best_path)
processor.save_pretrained(best_path)

logger.info("Running final evaluation")

results = trainer.evaluate()

with open(os.path.join(best_path, "accuracy.json"), "w") as f:
    json.dump({"best_accuracy": results["eval_accuracy"]}, f)

logger.info(f"Best Accuracy: {results['eval_accuracy']:.4f}")

print(f"Fine-tuning complete. Best Accuracy: {results['eval_accuracy']:.4f}")