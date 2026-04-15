from huggingface_hub import login
login("add your hf token")

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms
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
MODEL_NAME = "google/vit-huge-patch14-224-in21k"
IMAGE_SIZE = 224
BATCH_SIZE = 32              # Physical batch per GPU step
GRAD_ACCUM_STEPS = 16        # Effective batch = 32 * 16 = 512 (Crucial for Huge models)
EPOCHS = 10
BACKBONE_LR = 2e-5           # Very low for pre-trained weights
HEAD_LR = 2e-4               # Slightly higher for the new classifier
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
STEPS_PER_EPOCH = 10000      
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD DATA & AUGMENTATION
# =====================================================
print("Loading ImageNet-1K (Streaming)...")
dataset = load_dataset("imagenet-1k", split="train", streaming=True)
dataset = dataset.shuffle(buffer_size=10_000)

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# Robust Augmentation Pipeline
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

def collate_fn(batch):
    pixel_values = torch.stack([
        train_transforms(x["image"].convert("RGB")) for x in batch
    ])
    labels = torch.tensor([x["label"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# =====================================================
# MODEL (Partial Finetuning)
# =====================================================
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1000,
    ignore_mismatched_sizes=True
)

# Strategy: Freeze most of the model, but unfreeze the last 4 blocks 
# ViT-Huge has 32 layers (0 to 31).
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last 4 encoder blocks + the final LayerNorm + the Head
for i in range(28, 32):
    for param in model.vit.encoder.layer[i].parameters():
        param.requires_grad = True
for param in model.vit.layernorm.parameters():
    param.requires_grad = True
for param in model.classifier.parameters():
    param.requires_grad = True

model.to(DEVICE)

# =====================================================
# OPTIMIZER (Differential LR)
# =====================================================
# We group parameters so the backbone learns slower than the head
optimizer = AdamW([
    {'params': [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad], 'lr': BACKBONE_LR},
    {'params': model.classifier.parameters(), 'lr': HEAD_LR}
], weight_decay=WEIGHT_DECAY)

total_steps = STEPS_PER_EPOCH * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * total_steps),
    num_training_steps=total_steps,
)

scaler = torch.cuda.amp.GradScaler()

# =====================================================
# TRAINING LOOP
# =====================================================
model.train()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    progress_bar = tqdm(enumerate(train_loader), total=STEPS_PER_EPOCH)
    
    optimizer.zero_grad()
    
    # Trackers for the accumulation period
    accum_correct = 0
    accum_total = 0
    running_loss = 0

    for step, batch in progress_bar:
        if step >= STEPS_PER_EPOCH:
            break

        pixel_values = batch["pixel_values"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.cuda.amp.autocast():
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            logits = outputs.logits

        # Metrics
        preds = torch.argmax(logits, dim=1)
        accum_correct += (preds == labels).sum().item()
        accum_total += labels.size(0)
        running_loss += loss.item()

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            effective_acc = 100 * accum_correct / accum_total
            progress_bar.set_postfix(
                loss=f"{running_loss:.4f}",
                acc=f"{effective_acc:.2f}%"
            )

            # Reset trackers for next accumulation cycle
            accum_correct = 0
            accum_total = 0
            running_loss = 0

print("Training complete.")
model.save_pretrained("./vit_huge_finetuned_top4")

# Save the processor so you have the normalization/resize logic
processor.save_pretrained("./vit_huge_finetuned_top4")

print("Model and Processor saved successfully in ./vit_huge_finetuned_top4")

