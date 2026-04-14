#this script does uniform binning. 



import torch
import torch.nn.functional as F
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import log_loss
from datasets import load_dataset
import numpy as np
import json
import time
from tqdm import tqdm

# -------------------------------
# PATHS
# -------------------------------
model_path = "/workspace/amit/yatika/vit-scripts/finetune_vit_imagenet/fine-tune-huge/vit_head_only_huge_224"
# -------------------------------
# DEVICE SETUP
# -------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading local weights...")

model = ViTForImageClassification.from_pretrained(model_path)

image_processor = ViTImageProcessor.from_pretrained(
  "google/vit-huge-patch14-224-in21k"
)

model.to(device)
model.eval()

print("Model loaded")

# -------------------------------
# MODEL SANITY CHECKS
# -------------------------------
print("\n===== MODEL SANITY CHECKS =====")

print("Model num_labels:", model.config.num_labels)

if hasattr(model, "classifier"):
    print("Classifier weight shape:", model.classifier.weight.shape)

print("===============================\n")

# -------------------------------
# LOAD HUGGINGFACE IMAGENET
# -------------------------------
print("Loading ImageNet validation dataset...")

hf_dataset = load_dataset("imagenet-1k", split="validation")

print("HF dataset loaded")

# -------------------------------
# DATASET WRAPPER
# -------------------------------
class HFImageNetDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        item = self.dataset[idx]

        image = item["image"].convert("RGB")
        label = item["label"]

        inputs = self.processor(image, return_tensors="pt")

        img_tensor = inputs["pixel_values"].squeeze(0)

        return img_tensor, label


test_dataset = HFImageNetDataset(hf_dataset, image_processor)

num_labels = model.config.num_labels

print("Dataset wrapper ready")

# -------------------------------
# DATASET SANITY CHECKS
# -------------------------------
print("\n===== DATASET SANITY CHECKS =====")

print("Number of images:", len(test_dataset))
print("Number of classes:", num_labels)

sample_img, sample_label = test_dataset[0]

print("Sample tensor shape:", sample_img.shape)
print("Sample label id:", sample_label)

print("===============================\n")

# -------------------------------
# DATALOADER (CPU SAFE)
# -------------------------------
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("DataLoader ready")

# -------------------------------
# INFERENCE
# -------------------------------
results = []
forward_pass_times_sec = []

print(f"\nRunning inference on {len(test_dataset)} images...\n")

with torch.no_grad():
    for imgs, labels in tqdm(test_loader):

        start_time = time.time()

        imgs = imgs.to(device)

        outputs = model(imgs)

        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        forward_pass_times_sec.append(time.time() - start_time)

        labels = labels.numpy()

        for i in range(len(labels)):
            results.append({
                "true_id": int(labels[i]),
                "predicted_prob": probs[i]
            })

print("Inference complete")

# -------------------------------
# CALIBRATION LOGIC
# -------------------------------
y_true_ids = np.array([r["true_id"] for r in results])
probs = np.array([r["predicted_prob"] for r in results])

y_pred_ids = probs.argmax(axis=1)
confidences = probs.max(axis=1)

correct = (y_pred_ids == y_true_ids)

# -------------------------------
# TOP-5 ACCURACY
# -------------------------------
top5_preds = np.argsort(probs, axis=1)[:, -5:]
top5_correct = np.any(top5_preds == y_true_ids[:, None], axis=1)
top5_accuracy = top5_correct.mean()

# -------------------------------
# CALIBRATION METRICS
# -------------------------------
def compute_calibration(confidences, correct, n_bins=10):

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0

    bin_stats = {}
    tpr_bins = {}
    conf_bins = {}

    N = len(confidences)

    for i in range(n_bins):

        lo = bins[i]
        hi = bins[i + 1]

        mask = (confidences >= lo) & (confidences < hi)

        count = mask.sum()

        bin_key = f"{lo:.2f}-{hi:.2f}"

        if count > 0:

            acc_bin = correct[mask].mean()
            conf_bin = confidences[mask].mean()

            gap = abs(acc_bin - conf_bin)

            ece += (count / N) * gap
            mce = max(mce, gap)

            bin_stats[bin_key] = int(count)
            tpr_bins[bin_key] = float(acc_bin)
            conf_bins[bin_key] = float(conf_bin)

        else:

            bin_stats[bin_key] = 0
            tpr_bins[bin_key] = None
            conf_bins[bin_key] = None

    worst_bin_key = max(
        [(k, abs((tpr_bins[k] or 0) - (conf_bins[k] or 0))) for k in bin_stats],
        key=lambda x: x[1]
    )[0]

    mce_string = f"{mce:.4f} (bin [{worst_bin_key}), acc={tpr_bins[worst_bin_key]}, conf={conf_bins[worst_bin_key]})"

    return float(ece), float(mce), mce_string, bin_stats, tpr_bins, conf_bins


ece, mce, mce_string, softmax_bins, tpr_bins, avg_conf_bins = compute_calibration(
    confidences,
    correct
)

# -------------------------------
# FINAL METRICS
# -------------------------------
final_metrics = {

    "accuracy": float(correct.mean()),
    "top5_accuracy": float(top5_accuracy),

    "ece": ece,
    "mce": mce,
    "mce_detail": mce_string,

    "brier_score": float(
        np.mean(np.sum((probs - np.eye(num_labels)[y_true_ids]) ** 2, axis=1))
    ),

    "log_loss": float(
        log_loss(y_true_ids, probs, labels=np.arange(num_labels))
    ),

    "bin_counts": softmax_bins,
    "true_positive_rates": tpr_bins,
    "avg_confidences": avg_conf_bins,

    "total_forward_pass_time_sec": float(sum(forward_pass_times_sec))
}

# -------------------------------
# SAVE RESULTS
# -------------------------------
output_file = "imagenet_calibration_metrics_vit_huge.json"

with open(output_file, "w") as f:
    json.dump(final_metrics, f, indent=2)

# -------------------------------
# PRINT RESULTS
# -------------------------------
print("\n===== IMAGENET ViT CALIBRATION METRICS =====")

for k, v in final_metrics.items():
    if not isinstance(v, dict):
        print(f"{k}: {v}")

print("\nResults saved to:", output_file)