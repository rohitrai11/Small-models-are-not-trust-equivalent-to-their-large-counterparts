from huggingface_hub import login
login("add you hf token") #yatika token

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm




# =====================================================
# CONFIG
# =====================================================
MODEL_NAME = "google/vit-base-patch16-224-in21k"
IMAGE_SIZE = 224
BATCH_SIZE = 32              # physical batch
GRAD_ACCUM_STEPS = 4         # effective batch = 128
EPOCHS = 5
BASE_LR = 5e-4               # safe for head-only finetuning
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
STEPS_PER_EPOCH = 10000      # since streaming dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the config first and update the image size
# config = ViTConfig.from_pretrained(MODEL_NAME)
# config.image_size = IMAGE_SIZE  #update resolution in config
# config.num_labels = 1000        # ImageNet-1K

# =====================================================
# LOAD DATA (Streaming + Shuffle)
# =====================================================
print("Loading ImageNet-1K (Streaming)...")

dataset = load_dataset(
    "imagenet-1k",
    split="train",
    streaming=True
)

# Shuffle is CRITICAL for streaming datasets
dataset = dataset.shuffle(buffer_size=10_000)

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

def collate_fn(batch):
    images = [x["image"].convert("RGB") for x in batch]
    labels = torch.tensor([x["label"] for x in batch])

    inputs = processor(
        images,
        size=IMAGE_SIZE,
        return_tensors="pt"
    )

    return {
        "pixel_values": inputs["pixel_values"],
        "labels": labels
    }

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn
)

# =====================================================
# MODEL (Head Only Finetuning)
# =====================================================
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1000,                 # ImageNet-1K
    ignore_mismatched_sizes=True     # replace classification head
)

# Freeze backbone
for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

model.to(DEVICE)

# =====================================================
# OPTIMIZER & SCHEDULER
# =====================================================
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=BASE_LR,
    weight_decay=WEIGHT_DECAY,
)

total_steps = STEPS_PER_EPOCH * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# Mixed Precision
scaler = torch.cuda.amp.GradScaler()

# =====================================================
# TRAINING LOOP
# =====================================================
model.train()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    progress_bar = tqdm(enumerate(train_loader), total=STEPS_PER_EPOCH)

    optimizer.zero_grad()

    for step, batch in progress_bar:
        if step >= STEPS_PER_EPOCH:
            break

        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.cuda.amp.autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        progress_bar.set_postfix(
            loss=loss.item() * GRAD_ACCUM_STEPS,
            lr=scheduler.get_last_lr()[0]
        )

print("Training complete.")
model.save_pretrained("./vit_head_only_224")
print("Model saved.")
