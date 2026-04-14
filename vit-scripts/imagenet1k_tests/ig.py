import torch
import numpy as np
import os
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw
from transformers import ViTImageProcessor, ViTForImageClassification
from datasets import load_dataset
from huggingface_hub import login

login("add you hf-token here if needed")

# =====================================================
# Utility: Create experiment directory
# =====================================================
def create_experiment_dir(base_dir="shap_results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "visualizations"), exist_ok=True)
    
    return exp_dir


class ViTSHAPExplainer:
    def __init__(self, model_path, processor_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = ViTForImageClassification.from_pretrained(
            model_path, attn_implementation="eager", local_files_only=True
        ).to(self.device)

        self.model = self.model.half()
        self.model.eval()

        self.processor = ViTImageProcessor.from_pretrained(processor_path)

        self.patch_size = self.model.config.patch_size
        self.image_size = self.model.config.image_size
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size ** 2

    def preprocess(self, image):
        image = image.convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size))

        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device).half()

    # FIX: always compute prediction separately
    def predict(self, image):
        pixel_values = self.preprocess(image)
        with torch.no_grad():
            logits = self.model(pixel_values).logits
        return torch.argmax(logits, dim=1).item()

    def compute_patch_importance(self, image, steps=20, target_class=None):
        pixel_values = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(pixel_values).logits
            pred_class = torch.argmax(logits, dim=1).item()

        if target_class is None:
            target_class = pred_class

        baseline = torch.zeros_like(pixel_values)
        integrated_grads = torch.zeros_like(pixel_values)

        alphas = torch.linspace(0, 1, steps, device=self.device)

        for alpha in alphas:
            interpolated = (baseline + alpha * (pixel_values - baseline)).detach()
            interpolated.requires_grad_(True)

            out = self.model(interpolated)
            score = out.logits[0, target_class]

            self.model.zero_grad()
            score.backward()

            integrated_grads += interpolated.grad.detach()

        integrated_grads /= steps
        attributions = (pixel_values - baseline) * integrated_grads

        attr = attributions.float().cpu().numpy()[0]
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

        patch_scores = (patch_scores - patch_scores.min()) / (
            patch_scores.max() - patch_scores.min() + 1e-9
        )

        return patch_scores

    def get_topk_patches(self, patch_scores, k):
        return np.argsort(patch_scores)[::-1][:k].tolist()

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


def run_analysis(model_configs, topk_values=[2,5,10,15,20,25,30],
                 num_samples=50000, vis_samples=10, steps=20):

    exp_dir = create_experiment_dir()
    cache_dir = os.path.join(exp_dir, "cache")
    vis_base_dir = os.path.join(exp_dir, "visualizations")

    print(f"[{datetime.now()}] Loading dataset (streaming)...")

    #  STREAMING ENABLED
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)

    models = {
        name: ViTSHAPExplainer(cfg["model"], cfg["processor"])
        for name, cfg in model_configs.items()
    }

    model_names = list(models.keys())

    pair_scores = {
        f"{m1}_vs_{m2}": {k: [] for k in topk_values}
        for i, m1 in enumerate(model_names)
        for j, m2 in enumerate(model_names) if j > i
    }

    total_correct = 0
    total_images = 0

    start_time = time.time()

    for idx, sample in enumerate(dataset):
        if idx >= num_samples:
            break

        image = sample["image"]
        target_class = int(sample["label"])

        total_images += 1
        model_masks_top30 = {}

        for name, explainer in models.items():

            # FIXED accuracy
            pred_class = explainer.predict(image)
            if pred_class == target_class:
                total_correct += 1

            cache_file = os.path.join(cache_dir, f"{name}_{idx}.npy")

            if os.path.exists(cache_file):
                patch_scores = np.load(cache_file)
            else:
                patch_scores = explainer.compute_patch_importance(
                    image, steps=steps, target_class=target_class
                )
                np.save(cache_file, patch_scores)

            model_masks_top30[name] = explainer.get_topk_patches(patch_scores, 30)

        for k in topk_values:
            model_masks_dict = {}

            for name in models.keys():
                topk_patches = model_masks_top30[name][:k]
                mask = models[name].patches_to_mask(topk_patches)
                model_masks_dict[name] = mask

                if idx < vis_samples:
                    save_dir = os.path.join(vis_base_dir, f"top{k}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{name}_{idx}.png")
                    models[name].draw_topk(image, topk_patches, save_path)

            for key in pair_scores.keys():
                m1, m2 = key.split("_vs_")
                s = pixel_jaccard(model_masks_dict[m1], model_masks_dict[m2])
                pair_scores[key][k].append(float(s))

        if idx % 10 == 0:
            elapsed = time.time() - start_time
            avg = elapsed / (idx + 1)
            eta = avg * (num_samples - idx - 1)
            print(f"[{datetime.now()}] {idx+1}/{num_samples} | ETA: {eta/3600:.2f} hrs")

    # ================= SAVE RESULTS =================

    summary_scores = {
        pair: {k: float(np.mean(vals[k])) for k in topk_values}
        for pair, vals in pair_scores.items()
    }

    results = {
        "summary_scores": summary_scores,
        "total_images": total_images,
        "total_correct": total_correct,
        "accuracy": total_correct / total_images
    }

    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(exp_dir, "detailed.json"), "w") as f:
        json.dump(pair_scores, f, indent=2)

    print("\n==============================")
    print(f"Saved results in: {exp_dir}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    model_configs = {
        "vit_large": {
            "model": "/home/gpuuser2/gpuuser2_a/yatika/from-95-GB/yatika/vit-scripts/finetune_vit_imagenet/fine_tune_large/vit_head_large_224",
            "processor": "google/vit-large-patch16-224-in21k"
        },
        "vit_base": {
            "model": "/home/gpuuser2/gpuuser2_a/yatika/from-95-GB/yatika/vit-scripts/finetune_vit_imagenet/vit_head_base_224",
            "processor": "google/vit-base-patch16-224-in21k"
        },
        "vit_huge": {
            "model": "/home/gpuuser2/gpuuser2_a/yatika/from-95-GB/yatika/vit-scripts/finetune_vit_imagenet/fine-tune-huge/vit_head_only_huge_224",
            "processor": "google/vit-huge-patch14-224-in21k"
        } 
    }

    run_analysis(model_configs, num_samples=50000, steps=20)