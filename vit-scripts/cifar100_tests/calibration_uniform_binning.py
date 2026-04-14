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
import os

# -------------------------------
# MODEL + PROCESSOR CONFIG
# -------------------------------
models_config = [
   {    
        "name": "vit_large",
        "model_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-large-cifar100-nointerp/checkpoint-7038",
        "processor_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best"
    },
    {
        "name": "vit_base",
        "model_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best",
        "processor_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best"
    
    },
    {
        "name": "vit_huge",

          "model_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-huge-cifar100-nointerp-best",
        "processor_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-huge-cifar100-nointerp-best"
    }
]

# -------------------------------
# OUTPUT DIRECTORY
# -------------------------------
output_dir = "cifar100_multi_model_results"
os.makedirs(output_dir, exist_ok=True)

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
# LOAD CIFAR-100 ONCE
# -------------------------------
print("Loading CIFAR-100 test dataset...")
hf_dataset = load_dataset("cifar100", split="test")
print("Dataset loaded")

# -------------------------------
# DATASET WRAPPER
# -------------------------------
class HFCIFAR100Dataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["img"].convert("RGB")
        label = item["fine_label"]

        inputs = self.processor(image, return_tensors="pt")
        img_tensor = inputs["pixel_values"].squeeze(0)

        return img_tensor, label

# -------------------------------
# CALIBRATION FUNCTION (UNCHANGED)
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


# ===============================
# LOOP OVER MODELS
# ===============================
for config in models_config:

    name = config["name"]
    model_path = config["model_path"]
    processor_path = config["processor_path"]

    print("\n======================================")
    print(f"Running model: {name}")
    print("======================================")

    # -------------------------------
    # LOAD MODEL + PROCESSOR
    # -------------------------------
    model = ViTForImageClassification.from_pretrained(model_path)
    image_processor = ViTImageProcessor.from_pretrained(processor_path)

    model.to(device)
    model.eval()

    num_labels = model.config.num_labels

    # -------------------------------
    # DATASET + LOADER
    # -------------------------------
    test_dataset = HFCIFAR100Dataset(hf_dataset, image_processor)

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0
    )

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

            #  bf16 fix
            probs = F.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

            forward_pass_times_sec.append(time.time() - start_time)

            labels = labels.numpy()

            for i in range(len(labels)):
                results.append({
                    "true_id": int(labels[i]),
                    "predicted_prob": probs[i]
                })

    print("Inference complete")

    # -------------------------------
    # METRICS
    # -------------------------------
    y_true_ids = np.array([r["true_id"] for r in results])
    probs = np.array([r["predicted_prob"] for r in results])

    y_pred_ids = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    correct = (y_pred_ids == y_true_ids)

    # Top-5
    top5_preds = np.argsort(probs, axis=1)[:, -5:]
    top5_correct = np.any(top5_preds == y_true_ids[:, None], axis=1)
    top5_accuracy = top5_correct.mean()

    # Calibration
    ece, mce, mce_string, softmax_bins, tpr_bins, avg_conf_bins = compute_calibration(
        confidences,
        correct
    )

    # -------------------------------
    # FINAL METRICS
    # -------------------------------
    final_metrics = {
        "model_name": name,
        "model_path": model_path,
        "processor_path": processor_path,

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
    # SAVE FILE (INSIDE NEW DIR)
    # -------------------------------
    output_file = os.path.join(output_dir, f"{name}_metrics.json")

    with open(output_file, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"\nSaved results to: {output_file}")

print("\nALL MODELS COMPLETED ")