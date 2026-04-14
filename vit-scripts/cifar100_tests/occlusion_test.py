import torch
import torch.nn.functional as F
import numpy as np
import os
import json
from PIL import Image
from tqdm import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from huggingface_hub import login

login("hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")  # replace with your Hugging Face token


class MultiModelOcclusionAnalyzer:
    def __init__(self, model_configs, device=None, batch_size=32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.k_values = [2, 5, 10, 15, 20, 25, 30]

        self.models = []
        for cfg in model_configs:
            model = cfg["model"].to(self.device)
            model.eval()

            self.models.append({
                "name": cfg["name"],
                "model": model,
                "processor": cfg["processor"],
                "patch_size": cfg["patch_size"],
                "image_size": cfg["image_size"],
                "grid_size": cfg["image_size"] // cfg["patch_size"],
                "id2label": model.config.id2label
            })

    def _blur_patch(self, patch):
        return F.avg_pool2d(patch, kernel_size=3, stride=1, padding=1)

    def generate_occluded_batch(self, pixel_values, grid_size, patch_size):
        B, C, H, W = pixel_values.shape
        occluded_images = []

        for patch_id in range(grid_size ** 2):
            occluded = pixel_values.clone()

            row = patch_id // grid_size
            col = patch_id % grid_size

            y_start, y_end = row * patch_size, (row + 1) * patch_size
            x_start, x_end = col * patch_size, (col + 1) * patch_size

            patch = occluded[:, :, y_start:y_end, x_start:x_end]
            occluded[:, :, y_start:y_end, x_start:x_end] = self._blur_patch(patch)

            occluded_images.append(occluded)

        return torch.cat(occluded_images, dim=0)

    def batch_forward(self, model, inputs):
        outputs = []
        with torch.no_grad():
            for i in range(0, inputs.shape[0], self.batch_size):
                batch = inputs[i:i+self.batch_size]

                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                    out = model(batch).logits

                outputs.append(out)

        return torch.cat(outputs, dim=0)

    def compute_patch_importance(self, model_dict, image, target_class):
        processor = model_dict["processor"]
        model = model_dict["model"]

        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = model(pixel_values).logits
        pred_class = torch.argmax(logits, dim=1).item()

        if pred_class != target_class:
            return None, pred_class

        original_logit = logits[0, target_class].item()

        occluded_batch = self.generate_occluded_batch(
            pixel_values,
            model_dict["grid_size"],
            model_dict["patch_size"]
        )

        occluded_logits = self.batch_forward(model, occluded_batch)

        importance = original_logit - occluded_logits[:, target_class].detach().cpu().numpy()

        return importance, pred_class

    def patches_to_mask(self, top_patches, grid_size, patch_size, image_size):
        mask = np.zeros((image_size, image_size), dtype=np.uint8)

        for patch_id in top_patches:
            row = patch_id // grid_size
            col = patch_id % grid_size

            y_start, y_end = row * patch_size, (row + 1) * patch_size
            x_start, x_end = col * patch_size, (col + 1) * patch_size

            mask[y_start:y_end, x_start:x_end] = 1

        return mask

    def jaccard(self, m1, m2):
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return inter / union if union != 0 else 0.0

    def run(self, dataset, n_images=50000, save_vis=5, output_dir="./occlusion_results_multiK"):
        os.makedirs(output_dir, exist_ok=True)

        # create per-k visualization folders
        vis_dirs = {}
        for k in self.k_values:
            vis_dir = os.path.join(output_dir, f"vis_top{k}")
            os.makedirs(vis_dir, exist_ok=True)
            vis_dirs[k] = vis_dir

        cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        pairs = list(combinations([m["name"] for m in self.models], 2))

        scores = {
            k: {p: [] for p in pairs}
            for k in self.k_values
        }

        vis_count = {k: 0 for k in self.k_values}
        valid_count = 0

        for idx, sample in enumerate(tqdm(dataset)):
            if idx >= n_images:
                break

            image_key = "image" if "image" in sample else "img"
            label_key = "label" if "label" in sample else "fine_label"

            cache_path = os.path.join(cache_dir, f"{idx}.jpg")

            if os.path.exists(cache_path):
                image = Image.open(cache_path).convert("RGB")
                image = image.resize((224, 224), Image.BILINEAR)
                target_class = int(sample[label_key])
            else:
                image = sample[image_key].convert("RGB")
                target_class = int(sample[label_key])
                # Resize CIFAR-100 images to match model input size (224x224)
                image = image.resize((224, 224), Image.BILINEAR)
                image.save(cache_path)

            model_outputs = {}
            valid = True

            for m in self.models:
                importance, pred_class = self.compute_patch_importance(m, image, target_class)
                if importance is None:
                    valid = False
                    break
                model_outputs[m["name"]] = importance

            if not valid:
                continue

            valid_count += 1

            # compute top-30 ONCE
            top30 = {
                m["name"]: np.argsort(-model_outputs[m["name"]])[:30]
                for m in self.models
            }
            # inside the for k in self.k_values loop, after computing masks
            for k in self.k_values:
                masks = {}

                for m in self.models:
                    top_k = top30[m["name"]][:k]

                    mask = self.patches_to_mask(
                        top_k,
                        m["grid_size"],
                        m["patch_size"],
                        m["image_size"]
                    )
                    masks[m["name"]] = mask

                    # only save for first few samples (save_vis)
                    if vis_count[k] < save_vis:
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.imshow(image)

                        # create red transparent overlay for top-k patches
                        red_overlay = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
                        red_overlay[..., 0] = 255                # Red channel
                        red_overlay[..., 3] = mask * 120         # Alpha channel (0-255)

                        ax.imshow(red_overlay)
                        ax.set_title(f"{m['name']} | top{k}")
                        ax.axis("off")

                        plt.savefig(
                            os.path.join(vis_dirs[k], f"sample_top{k}_{m['name']}.png"),
                            bbox_inches="tight"
                        )
                        plt.close()

                vis_count[k] += 1
            # for k in self.k_values:
            #     masks = {}

            #     for m in self.models:
            #         top_k = top30[m["name"]][:k]

            #         mask = self.patches_to_mask(
            #             top_k,
            #             m["grid_size"],
            #             m["patch_size"],
            #             m["image_size"]
            #         )

            #         masks[m["name"]] = mask

            #         # save visualizations
            #         if vis_count[k] < save_vis:
            #             plt.figure(figsize=(5, 5))
            #             plt.imshow(image)
            #             plt.title(f"{m['name']} | top{k}")
            #             plt.axis("off")

            #             plt.savefig(
            #                 os.path.join(vis_dirs[k], f"{idx}_{m['name']}.png"),
            #                 bbox_inches="tight"
            #             )
            #             plt.close()

            #     vis_count[k] += 1

                for (m1, m2) in pairs:
                    j = self.jaccard(masks[m1], masks[m2])
                    scores[k][(m1, m2)].append(j)

        # aggregate results
        final_results = {}
        for k in self.k_values:
            final_results[f"top_{k}"] = {
                "valid_samples": valid_count,
                "pairwise_jaccard": {
                    f"{a}__vs__{b}": float(np.mean(scores[k][(a, b)]))
                    for (a, b) in scores[k]
                }
            }

        with open(os.path.join(output_dir, "jaccard_results_all_k.json"), "w") as f:
            json.dump(final_results, f, indent=2)

        print("DONE. Results saved.")


def main():
    dataset = load_dataset("cifar100", split="test")

    model_configs = [
       {
            "name": "vit_large",
            "model": ViTForImageClassification.from_pretrained("/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-large-cifar100-nointerp/checkpoint-7038"),
            "processor": ViTImageProcessor.from_pretrained("/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best"),
            "patch_size": 16,
            "image_size": 224
        },
        {
            "name": "vit_huge",
            "model": ViTForImageClassification.from_pretrained("/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-huge-cifar100-nointerp-best"),
            "processor": ViTImageProcessor.from_pretrained("/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-huge-cifar100-nointerp-best"),
            "patch_size": 14,
            "image_size": 224
        },
        {
            "name": "vit_base",
            "model": ViTForImageClassification.from_pretrained("/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best"),
            "processor": ViTImageProcessor.from_pretrained("/home/gpuuser0/gpuuser0_a/yatika-btp/from-95-GB/yatika/vit-scripts/CIFAR-100-finetuning/vit_no_interpolation_finetuning/vit-base-cifar100-nointerp-best"),
            "patch_size": 16,
            "image_size": 224
        },
    ]

    analyzer = MultiModelOcclusionAnalyzer(model_configs, batch_size=32)

    analyzer.run(
        dataset,
        n_images=50000,
        save_vis=5,
        output_dir="./occlusion_results_multiK_v2"
    )


if __name__ == "__main__":
    main()