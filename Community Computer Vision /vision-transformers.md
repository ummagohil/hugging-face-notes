# Swin Transformer

- Shifted windows approach (reduces number of operations required)
- Considered a hierarchical backbone for computer vision
  - A backbone, in terms of deep learning, is a part of a neural network that does feature extraction. Additional layers can be added to the backbone to do a variety of vision tasks. Hierarchical backbones have tiered structures, sometimes with varying resolutions.

## Swin Transformer class

1. Initialise parameters (`window_size`, `ape` and `fused_window_process`)
2. Apply patch embedding
3. Apply positional embeddings
4. Apply depth decay
5. Layer construction
6. Classification head

```python
from datasets import load_dataset
from transformers import AutoImageProcessor, SwinForImageClassification
import torch

model = SwinForImageClassification.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224"
)
image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/swin-tiny-patch4-window7-224"
)

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label_id = logits.argmax(-1).item()
predicted_label_text = model.config.id2label[predicted_label_id]

print(predicted_label_text)
```

# Convolutional Vision Transformer (CvT)

- Hierarchy of Transformers containing a new Convolutional token embedding.
- Convolutional Transformer block leveraging a Convolutional Projection.
- Removal of Positional Encoding due to built-in local context structure introduced by convolutions.
- Fewer Parameters and lower FLOPs (Floating Point Operations per second) compared to other vision transformer architectures.

```python
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
    if method == "dw_bn":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            dim_in,
                            dim_in,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            bias=False,
                            groups=dim_in,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(dim_in)),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "avg":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "avg",
                        nn.AvgPool2d(
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            ceil_mode=True,
                        ),
                    ),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "linear":
        proj = None
    else:
        raise ValueError("Unknown method ({})".format(method))

    return proj

class ConvEmbed(nn.Module):
    def __init__(
        self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x
```

# Dilated Neighborhood Attention Transformer (DINAT)

```python
from transformers import AutoImageProcessor, DinatForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = AutoImageProcessor.from_pretrained("shi-labs/dinat-mini-in1k-224")
model = DinatForImageClassification.from_pretrained("shi-labs/dinat-mini-in1k-224")

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```

# MobileViT v2

```js
import fetch from "node-fetch";
import fs from "fs";
async function query(filename) {
  const data = fs.readFileSync(filename);
  const response = await fetch(
    "https://api-inference.huggingface.co/models/apple/mobilevitv2-1.0-imagenet1k-256",
    {
      headers: { Authorization: `Bearer ${API_TOKEN}` },
      method: "POST",
      body: data,
    }
  );
  const result = await response.json();
  return result;
}
query("cats.jpg").then((response) => {
  console.log(JSON.stringify(response));
});
```

# FineTuning Vision Transformer for Object Detection

- Using Hugging Face Transformers and PyTorch: `!pip install -U -q datasets transformers[torch] evaluate timm albumentations accelerate`

1. Get the image and it's height and width
2. Make a draw object that can draw text and lines on images
3. Get annotations dict from the sample
4. Iterate over it
5. For each iteration, get the bounding box co-ords (x - horizontal, y - vertical, w - width, h - height)
6. If bounding box measures are normalised scale it otherwise leave it
7. Draw the rectangle and the class category text

```python
import numpy as np
from PIL import Image, ImageDraw


def draw_image_from_idx(dataset, idx):
    sample = dataset[idx]
    image = sample["image"]
    annotations = sample["objects"]
    draw = ImageDraw.Draw(image)
    width, height = sample["width"], sample["height"]

    for i in range(len(annotations["id"])):
        box = annotations["bbox"][i]
        class_idx = annotations["id"][i]
        x, y, w, h = tuple(box)
        if max(box) > 1.0:
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
        else:
            x1 = int(x * width)
            y1 = int(y * height)
            x2 = int((x + w) * width)
            y2 = int((y + h) * height)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=1)
        draw.text((x1, y1), annotations["category"][i], fill="white")
    return image


draw_image_from_idx(dataset=train_dataset, idx=10)
```

- There is a function to plot multiple images

```python
import matplotlib.pyplot as plt


def plot_images(dataset, indices):
    """
    Plot images and their annotations.
    """
    num_rows = len(indices) // 3
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    for i, idx in enumerate(indices):
        row = i // num_cols
        col = i % num_cols

        # Draw image
        image = draw_image_from_idx(dataset, idx)

        # Display image on the corresponding subplot
        axes[row, col].imshow(image)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


# Now use the function to plot images

plot_images(train_dataset, range(9))
```

- Ideally preprocess the data set for the same attributes such as height, width, rotation, resizing etc.

```python
import albumentations
import numpy as np
import torch

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)
```

- Combining the image and annotation transformations over the whole batch of dataset:

```python
# transforming a batch


def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        out = transform(image=image, bboxes=objects["bbox"], category=objects["id"])

        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")
```

# DEtection TRansformer (DETR)

- DETR is mainly used for object detection tasks, which is the process of detecting objects in an image
- DEtection TRansformer, DETR for short, simplifies the detector by using an encoder-decoder transformer after the feature extraction backbone to directly predict bounding boxes in parallel, requiring minimal post-processing
- The feature map of size `[dimension, height, width]` is downsized and then flattened to `[dimension, less than height x width]`

```python
import torch
from torch import nn
from torchvision.models import resnet50


class DETR(nn.Module):
    def __init__(
        self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers
    ):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )
        h = self.transformer(
            pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1)
        )
        return self.linear_class(h), self.linear_bbox(h).sigmoid()
```

# Vision Transformer for Image Segmentation

```python
from transformers import pipeline
from PIL import Image
import requests

segmentation = pipeline("image-segmentation", "facebook/maskformer-swin-base-coco")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

results = segmentation(images=image, subtask="panoptic")
results
```

# OneFormer

- Approach to image segmentation, a computer vision task that involves dividing an image into meaningful segments

## Key Features

- Task-Dynamic Mask: decide whether to pay attention to the overall scene, identify specific objects with clear boundaries or combination of both
- Task-Conditioned Joined Training: instead of training separately for semantic, instance and panoptic segmentation, it uniformly samples the task during training
- Query-Text Contrastive Loss: versatile-like a multitasking assistant and learn specifics of each task by comparing visual features with textual descriptions

`!pip install -q natten`

```python
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt


def run_segmentation(image, task_type):
    """Performs image segmentation based on the given task type.

    Args:
        image (PIL.Image): The input image.
        task_type (str): The type of segmentation to perform ('semantic', 'instance', or 'panoptic').

    Returns:
        PIL.Image: The segmented image.

    Raises:
        ValueError: If the task type is invalid.
    """

    processor = OneFormerProcessor.from_pretrained(
        "shi-labs/oneformer_ade20k_dinat_large"
    )  # Load once here
    model = OneFormerForUniversalSegmentation.from_pretrained(
        "shi-labs/oneformer_ade20k_dinat_large"
    )

    if task_type == "semantic":
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
        outputs = model(**inputs)
        predicted_map = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

    elif task_type == "instance":
        inputs = processor(images=image, task_inputs=["instance"], return_tensors="pt")
        outputs = model(**inputs)
        predicted_map = processor.post_process_instance_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]["segmentation"]

    elif task_type == "panoptic":
        inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
        outputs = model(**inputs)
        predicted_map = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]["segmentation"]

    else:
        raise ValueError(
            "Invalid task type. Choose from 'semantic', 'instance', or 'panoptic'"
        )

    return predicted_map


def show_image_comparison(image, predicted_map, segmentation_title):
    """Displays the original image and the segmented image side-by-side.

    Args:
        image (PIL.Image): The original image.
        predicted_map (PIL.Image): The segmented image.
        segmentation_title (str): The title for the segmented image.
    """

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_map)
    plt.title(segmentation_title + " Segmentation")
    plt.axis("off")
    plt.show()


url = "https://huggingface.co/datasets/shi-labs/oneformer_demo/resolve/main/ade20k.jpeg"
response = requests.get(url, stream=True)
response.raise_for_status()  # Check for HTTP errors
image = Image.open(response.raw)

task_to_run = "semantic"
predicted_map = run_segmentation(image, task_to_run)
show_image_comparison(image, predicted_map, task_to_run)
```

# Knowledge Distillation with Vision Transformer
