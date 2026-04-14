import torch
import numpy as np
import os
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw
from transformers import ViTImageProcessor, ViTForImageClassification
from datasets import load_dataset

class ViTSHAPExplainer:
    def __init__(self, model_path, processor_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = ViTForImageClassification.from_pretrained(
            model_path, attn_implementation="eager", local_files_only=True
        ).to(self.device)
        self.model.eval()

        self.processor = ViTImageProcessor.from_pretrained(processor_path)

        self.patch_size = self.model.config.patch_size
        self.image_size = self.model.config.image_size
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size ** 2

        print(f"[{datetime.now()}] Loaded model: {model_path}")
        print(f"[{datetime.now()}] Processor: {processor_path}")

    def preprocess(self, image):
        image = image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size))
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        return pixel_values

    def compute_patch_importance(self, image, steps=100, target_class=None):
        pixel_values = self.preprocess(image)
        pixel_values.requires_grad = True

        outputs = self.model(pixel_values)
        logits = outputs.logits
        pred_class = torch.argmax(logits, dim=1).item()

        if target_class is None:
            target_class = pred_class

        baseline = torch.zeros_like(pixel_values)
        integrated_grads = torch.zeros_like(pixel_values)

        for alpha in np.linspace(0, 1, steps):
            interpolated = (baseline + alpha * (pixel_values - baseline)).detach().requires_grad_(True)
            out = self.model(interpolated)
            score = out.logits[0, target_class] #wrt to target class logit

            self.model.zero_grad()
            score.backward()
            integrated_grads += interpolated.grad.detach()

        integrated_grads /= steps
        attributions = (pixel_values.detach() - baseline) * integrated_grads

        attr = attributions.cpu().numpy()[0]
        attr2d = np.abs(attr).mean(axis=0)

        patch_scores = np.zeros(self.num_patches)
        for patch_idx in range(self.num_patches):
            r = patch_idx // self.grid_size
            c = patch_idx % self.grid_size
            ys = r * self.patch_size
            ye = (r + 1) * self.patch_size
            xs = c * self.patch_size
            xe = (c + 1) * self.patch_size
            patch_scores[patch_idx] = attr2d[ys:ye, xs:xe].mean()

        patch_scores = (patch_scores - patch_scores.min()) / (patch_scores.max() - patch_scores.min() + 1e-9)
        return patch_scores, pred_class

    def get_topk_patches(self, patch_scores, k):
        idx = np.argsort(patch_scores)[::-1][:k]
        return idx.tolist()

    def patches_to_mask(self, topk_patches):
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        for p in topk_patches:
            r = p // self.grid_size
            c = p % self.grid_size
            ys = r * self.patch_size
            ye = (r + 1) * self.patch_size
            xs = c * self.patch_size
            xe = (c + 1) * self.patch_size
            mask[ys:ye, xs:xe] = 1
        return mask

    def draw_topk(self, image, topk, save_path):
        img = image.resize((self.image_size, self.image_size))
        draw = ImageDraw.Draw(img)
        for p in topk:
            r = p // self.grid_size
            c = p % self.grid_size
            xs = c * self.patch_size
            ys = r * self.patch_size
            xe = xs + self.patch_size
            ye = ys + self.patch_size
            draw.rectangle([xs, ys, xe, ye], outline="red", width=3)
        img.save(save_path)

def pixel_jaccard(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return 0.0 if union == 0 else intersection / union

def run_analysis(model_configs, topk_values=[2,5,10,15,20,25,30], num_samples=2000, vis_samples=10, steps=100):
    print(f"\n[{datetime.now()}] Loading CIFAR-100 test dataset...")
    dataset = load_dataset("cifar100", split="test")

    models = {name: ViTSHAPExplainer(cfg["model"], cfg["processor"]) for name, cfg in model_configs.items()}
    model_names = list(models.keys())

    pair_scores = {f"{m1}_vs_{m2}": {k: [] for k in topk_values} 
                   for i, m1 in enumerate(model_names) 
                   for j, m2 in enumerate(model_names) if j > i}

    total_correct = 0
    total_images = 0

    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    start_time = time.time()

    for idx, sample in enumerate(dataset):
        if idx >= num_samples:
            break

        image_start = time.time()
        image_key = "image" if "image" in sample else "img"
        label_key = "label" if "label" in sample else "fine_label"
        image = sample[image_key].convert("RGB")
        target_class = int(sample[label_key])
        total_images += 1

        model_masks_top30 = {}
        correct_flags = {}

        # Compute patch scores and predicted classes
        for name, explainer in models.items():
            cache_file = f"cache/{name}_{idx}.npy"
            if os.path.exists(cache_file):
                loaded = np.load(cache_file, allow_pickle=True)
                if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
                    cached = loaded.item()
                    patch_scores = cached["patch_scores"]
                    pred_class = int(cached["pred_class"])
                else:
                    # Legacy cache format: patch scores only; recompute pred_class.
                    patch_scores = loaded
                    _, pred_class = explainer.compute_patch_importance(image, steps=steps, target_class=target_class)
            else:
                patch_scores, pred_class = explainer.compute_patch_importance(image, steps=steps, target_class=target_class)
                np.save(cache_file, {
                    "patch_scores": patch_scores,
                    "pred_class": int(pred_class)
                })

            correct_flags[name] = int(pred_class == target_class)
            if pred_class == target_class:
                total_correct += 1  # Accumulate correct predictions
            model_masks_top30[name] = explainer.get_topk_patches(patch_scores, 30)

        # Compute masks and Jaccard similarity for all images
        model_masks_dict = {k: {} for k in topk_values}
        for k in topk_values:
            for name in models.keys():
                topk_patches = model_masks_top30[name][:k]
                mask = models[name].patches_to_mask(topk_patches)
                model_masks_dict[k][name] = mask

                # Visualize top-k patches for first vis_samples images
                if idx < vis_samples:
                    save_dir = f"visualizations/top{k}"
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = f"{save_dir}/{name}_{idx}.png"
                    models[name].draw_topk(image, topk_patches, save_path)

            for key in pair_scores.keys():
                m1, m2 = key.split("_vs_")
                s = pixel_jaccard(model_masks_dict[k][m1], model_masks_dict[k][m2])
                pair_scores[key][k].append(float(s))

        image_time = time.time() - image_start
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (num_samples - idx - 1)
        percent = (idx + 1) / num_samples * 100
        print(f"[{datetime.now()}] Image {idx+1}/{num_samples} "
              f"({percent:.2f}%) | Time/img: {image_time:.2f}s | ETA: {remaining/3600:.2f} hrs")

    summary_scores = {pair: {k: float(np.mean(vals[k])) if vals[k] else None for k in topk_values} 
                      for pair, vals in pair_scores.items()}

    results = {
        "summary_scores": summary_scores,
        "total_images": total_images,
        "total_correct": total_correct,
        "accuracy": total_correct / total_images
    }

    with open("pixel_jaccard_results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("pixel_jaccard_results_detailed.json", "w") as f:
        json.dump(pair_scores, f, indent=2)

    print("\n==============================")
    print(f"Total images processed: {total_images}")
    print(f"Total correct predictions: {total_correct}")
    print(f"Accuracy: {total_correct / total_images:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    model_configs = {
        "vit_large": {
            "model": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-large-cifar100-nointerp/checkpoint-7038",
            "processor": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best"
        },
        "vit_base": {
            "model":"/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best",
            "processor": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best"
        },
        "vit_huge": {
            "model": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-huge-cifar100-nointerp-best",
            "processor": "/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-huge-cifar100-nointerp-best"
        }
    }

    run_analysis(model_configs, topk_values=[2,5,10,15,20,25,30], num_samples=10000, vis_samples=10, steps=100)
