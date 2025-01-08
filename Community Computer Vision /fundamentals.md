# Fundamentals

## What is Computer Vision

- Low level processes: primitive operations on images (eg. image sharpening, changing the contract). The input and output are both images.
- Mid level processes: object classification and image description - can be done with a combination of image preprocessing and ML algorithms.
- High level processes: making sense of the whole image (eg. image-to-text). These tasks are usually associated with human cognition.

## Feature Description

- Numerical feats

  - Array/lists: elements in the array corresponds to a feature
  - Tensors: multidimensional arrays often used in ML frameworks to hand large data sets

- Categorical feats

  - Dictionaries/lists: assigning categories ot numerical label/directly storing categorical values
  - One hot encoding: transforming categorical variables into binary vectors where each bit represents a category

- Image feats

  - Pixel values: strong pixel values matrices of multi-dimensional arrays
  - Convolutional Neural Network Features (CNN): extracting features using pre-trained CNN models

- SIFT (technique used in feature descriptors)
  - Scale space extrema detection: starts by detecting potential interest points in an image across multiple scales - looks for locations in the image where the difference of Gaussian function reaches a max/min over space and scale (which leads onto next part - these keypoint locations are considered stable under various scale changes)
  - Keypoint localisation: SIFT refines the positions of keypoints to sub-pixel accuracy and discards low contrast key points and keypoints on edge, to ensure accurate localisation
  - Orientation assignment: SIFT computes a dominant orientation of each keypoint based on local image gradient directions - this stage makes the descriptor invariant to image rotation
  - Descriptor generation: a descriptor is computed for each keypoint region, capturing information about the local image gradients near the keypoint
  - Descriptor matching: descriptors are used for matching keypoints between different images and the descriptors from one image are compared to those in another image to find correspondences.

![alt text](https://huggingface.co/datasets/hf-vision/course-assets/resolve/main/feature-extraction-feature-matching/Flow-Chart-for-SURF-Feature-Detection.png)

## Feature Matching

### Brute-Force Search

- You check each piece one by one - it's simple but not efficient (time consuming)

```python
import cv2
import numpy as np

# Initialise SIFT detector
sift = cv2.SIFT_create()

# key points and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# find matches using k nearest neighbors
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# apply ratio test to threshold the best matches
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# draw the matches
img3 = cv2.drawMatchesKnn(
    img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
```

#### Fast Library for Approximate Nearest Neighbors (FLANN)

- FLANN streamlines the process by finding pieces that are approximately similar. This means it can make educated guesses about which pieces might fit well together, even if they’re not an exact match.
- Under the hood, FLANN is uses something called k-D trees. Think of it as organizing the puzzle pieces in a special way.
- Instead of checking every piece against every other piece, FLANN arranges them in a tree-like structure that makes finding matches faster.

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2
)

search_params = dict(checks=50)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

draw_params = dict(
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    matchesMask=matchesMask,
    flags=cv2.DrawMatchesFlags_DEFAULT,
)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
```
