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

# DEtection TRansformer (DETR)

# Vision Transformer for Image Segmentation

# OneFormer

# Knowledge Distillation with Vision Transformer
