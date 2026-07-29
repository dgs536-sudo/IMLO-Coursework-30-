# Pet Breed Image Classifier

A convolutional neural network built **from scratch** in PyTorch to classify 37 dog and cat breeds from the Oxford-IIIT Pet dataset — trained without any pretrained weights or transfer learning.

## Overview

This project started as an exploration of CNN fundamentals and turned into a debugging exercise once training accuracy plateaued far below where it should have. The dataset provides pixel-level trimap segmentation masks alongside breed labels, and using them to isolate the animal from its background — rather than editing image files on disk — was the fix that took test accuracy from ~23% to 62%, over 20x the random-guess baseline for a 37-class problem.

## Key results

| Metric | Value |
|---|---|
| Number of classes | 37 breeds |
| Dataset size | 7,349 images (Oxford-IIIT Pet) |
| Final test accuracy | ~62% |
| Random-guess baseline | ~2.7% (1/37) |
| Improvement over baseline | 20x+ |

## The problem: overfitting plateau

Early training runs stalled well below a usable accuracy. Two root causes were identified and addressed:

1. **Background noise dominating the signal** — with 37 visually similar breeds, cluttered and inconsistent backgrounds were making it harder for the network to learn breed-distinguishing features from the animal itself.
2. **A hard constraint against modifying source images on disk** — ruling out the obvious fix of pre-processing and saving masked images as a new dataset.

**Solution:** an in-memory augmentation pipeline that applies **trimap-based foreground masking** on the fly, inside a custom PyTorch `Dataset` class (`PetWithTrimap`). Each image is masked against its trimap at load time — zeroing out background pixels (`trimap == 2`) before the image ever reaches the model — with the original files on disk left untouched.

```python
fg_mask = (trimap_np != 2).astype(np.float32)
fg_mask_3ch = fg_mask[:, :, np.newaxis]
image_np = image_np * fg_mask_3ch
```

## Architecture

A 4-block custom CNN (no pretrained backbone):

- **4 convolutional blocks**, each with two `Conv2d` layers, `BatchNorm2d`, and `ReLU`, followed by `MaxPool2d` (channels: 32 → 64 → 128 → 256; spatial size: 128 → 64 → 32 → 16 → 8)
- **Classifier head**: flatten → `Linear(256×8×8, 512)` → `BatchNorm1d` → `ReLU` → `Dropout(0.5)` → `Linear(512, 37)`
- **Kaiming initialisation** on all conv/linear layers

## Training setup

- **Optimizer**: Adam, `lr=1e-3`, `weight_decay=3e-4`
- **LR schedule**: Cosine annealing over 30 epochs (`eta_min=1e-5`)
- **Loss**: Cross-entropy
- **Augmentation** (train only): random horizontal flip, rotation (±15°), colour jitter, random affine translation, random grayscale, random erasing
- **Input size**: 128×128, normalised with ImageNet mean/std

## Tech stack

Python · PyTorch · torchvision · NumPy · PIL

## Running it

```bash
pip install torch torchvision pillow numpy
```

The notebook downloads the Oxford-IIIT Pet dataset automatically via `torchvision.datasets.OxfordIIITPet` (including category labels and segmentation trimaps) on first run. Trained for 30 epochs on GPU (Colab); CPU training will work but is significantly slower given the 4-block architecture.

## Notes

- Trained and developed in Google Colab; model checkpoint saved as `model.pth`.
- This project intentionally avoids transfer learning to focus on CNN fundamentals — a pretrained backbone (ResNet, EfficientNet, etc.) would substantially outperform this accuracy, but that wasn't the point of the exercise.
