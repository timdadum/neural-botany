import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms


class CenterCropToSquare:
    """Crop to min(w, h) centered on the image midpoint."""

    def __call__(self, img):
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return img.crop((left, top, left + s, top + s))


def get_dataloader(batch_size=128, image_size=32, color=False,
                    data_dir="./data", num_workers=0):
    steps = [CenterCropToSquare(), transforms.Resize(image_size)]
    if color:
        steps += [transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)]
    else:
        steps += [transforms.Grayscale(), transforms.ToTensor(),
                  transforms.Normalize([0.5], [0.5])]

    transform = transforms.Compose(steps)
    splits = [
        datasets.Flowers102(data_dir, split=s, download=True, transform=transform)
        for s in ("train", "val", "test")
    ]
    full = ConcatDataset(splits)
    mode = "RGB" if color else "grayscale"
    print(f"Dataset: {len(full)} images  |  {image_size}×{image_size} {mode}")
    return DataLoader(full, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True)
