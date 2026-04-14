# this script does quantile binning for the CIFAR-100 test set, computes calibration metrics, and generates a reliability diagram with confidence intervals for each bin. It is designed to run on multiple ViT models and saves the results in a structured format for easy analysis.


import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import beta

from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader

# =========================
# CONFIG
# =========================
BATCH_SIZE = 32
NUM_BINS = 10

MODEL_CONFIGS = {
     "vit_base": {
        "model_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best",
        "processor_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best"
    },
    "vit_large": {
        "model_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-large-cifar100-nointerp/checkpoint-7038",
        "processor_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best"
    },
    "vit_huge": {
        "model_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-huge-cifar100-nointerp-best",
        "processor_path": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-huge-cifar100-nointerp-best"
    }
}

BASE_OUTPUT_DIR = "./Predictions/ViT-CIFAR100/"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# =========================
# DATASET (CIFAR-100)
# =========================
def get_dataloader(processor):
    dataset = load_dataset("cifar100", split="test")
  

    def preprocess(example):
        image = example["img"].convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        return {
            "pixel_values": inputs["pixel_values"][0],
            "label": example["fine_label"]   # CIFAR-100 uses fine_label
        }

    dataset = dataset.map(preprocess)
    dataset.set_format(type="torch", columns=["pixel_values", "label"])

    def collate_fn(batch):
        pixel_values = torch.stack([
            torch.tensor(x["pixel_values"]) if not isinstance(x["pixel_values"], torch.Tensor) else x["pixel_values"]
            for x in batch
        ])
        labels = torch.tensor([x["label"] for x in batch])
        return {"pixel_values": pixel_values, "labels": labels}
    
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# =========================
# METRICS (UNCHANGED)
# =========================
def compute_metrics(probs, labels, output_dir, num_bins=10, taskname="CIFAR100"):
    probs = torch.stack(probs).cpu().numpy().astype(np.float32)
    labels = torch.stack(labels).cpu().numpy()

    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    bin_edges = np.percentile(max_probs, np.linspace(0, 100, num_bins + 1))
    bin_edges = np.unique(bin_edges)
    actual_num_bins = len(bin_edges) - 1

    bin_indices = np.digitize(max_probs, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, actual_num_bins - 1)

    bin_correct = np.zeros(actual_num_bins)
    bin_conf = np.zeros(actual_num_bins)
    bin_counts = np.zeros(actual_num_bins)

    for i in range(len(labels)):
        b = bin_indices[i]
        bin_counts[b] += 1
        bin_conf[b] += max_probs[i]
        bin_correct[b] += (preds[i] == labels[i])

    ece = 0.0
    mce = 0.0
    mce_str = ""

    true_positive_rates = []
    avg_confidences = []
    bin_centers = []
    errors = []

    for b in range(actual_num_bins):
        if bin_counts[b] > 0:
            acc = bin_correct[b] / bin_counts[b]
            conf = bin_conf[b] / bin_counts[b]
            diff = abs(acc - conf)

            ece += (bin_counts[b] / len(labels)) * diff

            if diff >= mce:
                mce = diff
                left, right = bin_edges[b], bin_edges[b + 1]
                mce_str = f"{mce:.4f} (from bin [{left:.2f}, {right:.2f}), center={(left+right)/2:.2f}, count={int(bin_counts[b])}, acc={acc:.4f}, conf={conf:.4f})"

            true_positive_rates.append(acc)
            avg_confidences.append(conf)
            bin_centers.append(conf)

            alpha = 0.05
            n = bin_counts[b]
            k = bin_correct[b]

            lower = beta.ppf(alpha/2, k, n - k + 1) if k > 0 else 0
            upper = beta.ppf(1 - alpha/2, k + 1, n - k) if k < n else 1

            errors.append([acc - lower, upper - acc])
        else:
            true_positive_rates.append(np.nan)
            avg_confidences.append(np.nan)
            bin_centers.append((bin_edges[b] + bin_edges[b+1]) / 2)
            errors.append([0, 0])

    # Reliability diagram
    plt.figure(figsize=(8, 6))
    err_array = np.array(errors).T
    plt.errorbar(bin_centers, true_positive_rates, yerr=err_array, fmt='o', capsize=5)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram ({taskname})")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reliability.png"))
    plt.close()

    def bin_label(i):
        return f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"

    softmax_bins_dict = {bin_label(i): int(bin_counts[i]) for i in range(actual_num_bins)}
    true_positive_dict = {
        bin_label(i): (float(true_positive_rates[i]) if not np.isnan(true_positive_rates[i]) else None)
        for i in range(actual_num_bins)
    }
    avg_conf_dict = {
        bin_label(i): (float(avg_confidences[i]) if not np.isnan(avg_confidences[i]) else None)
        for i in range(actual_num_bins)
    }

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    brier_score = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    true_class_probs = probs[np.arange(len(labels)), labels]
    log_loss = -np.mean(np.log(true_class_probs + 1e-12))

    return (
        ece, mce, brier_score, log_loss,
        softmax_bins_dict, true_positive_dict, avg_conf_dict, mce_str
    )

# =========================
# INFERENCE (UNCHANGED)
# =========================
def inference(model, dataloader, device):
    model.eval()

    prob_lst = []
    gold_lst = []

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, ncols=100):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits

            probs = F.softmax(logits, dim=1).float()
            preds = torch.argmax(probs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            prob_lst.extend(probs.cpu())
            gold_lst.extend(labels.cpu())

    accuracy = correct / total
    return accuracy, prob_lst, gold_lst

# =========================
# MAIN
# =========================
def main():
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name, paths in MODEL_CONFIGS.items():
        print(f"\n===== Running {model_name} =====")

        model_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        start = time.time()

        processor = ViTImageProcessor.from_pretrained(paths["processor_path"])
        model = ViTForImageClassification.from_pretrained(paths["model_path"])
        model.to(device)

        dataloader = get_dataloader(processor)

        accuracy, prob_lst, gold_lst = inference(model, dataloader, device)

        (
            ece, mce, brier, logloss,
            softmax_bins_dict, true_positive_dict,
            avg_conf_dict, mce_str
        ) = compute_metrics(
            prob_lst, gold_lst, model_output_dir, num_bins=NUM_BINS
        )

        total_time = time.time() - start

        metrics = {
            "accuracy": float(accuracy),
            "ece": float(ece),
            "mce": float(mce),
            "Maximum Calibration Error (MCE)": mce_str,
            "brier_score": float(brier),
            "log_loss": float(logloss),
            "total_inference_time": float(total_time),
            "softmax_bins": softmax_bins_dict,
            "true_positive_rates": true_positive_dict,
            "avg_confidences": avg_conf_dict
        }

        with open(os.path.join(model_output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        print(metrics)

        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()