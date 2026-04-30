# Image Captioning using Pretrained CLIP + GPT-2

## Project Overview

This project implements an image captioning system using a pretrained vision-language model and a pretrained language model. The main goal is to generate natural language captions for input images.

The system uses:

- **CLIP** as the image encoder
- **GPT-2** as the language model
- A custom trainable **CLIP-to-GPT prefix mapper**
- The **Flickr30k** dataset for image-caption training and evaluation

Instead of training a full multimodal model from scratch, this project uses pretrained models and trains only the connection between the image encoder and the language model. This makes the project realistic for Google Colab while still showing meaningful multimodal learning.

---

## Project Title

**Image Captioning using Pretrained CLIP and GPT-2 with a Trainable Prefix Mapping Network**

---

## Objective

The objective of this project is to generate captions for images by connecting visual features from CLIP with text generation capabilities from GPT-2.

The project focuses on:

1. Loading image-caption data.
2. Extracting image embeddings using pretrained CLIP.
3. Mapping CLIP image embeddings into GPT-2 soft prompt embeddings.
4. Generating captions using GPT-2.
5. Evaluating captions using validation loss and BLEU score.
6. Improving results using a Transformer-based mapper and optional GPT-2 fine-tuning.

---

## Why Pretrained Models Are Used

Training CLIP or GPT-2 from scratch requires very large datasets and high computational resources. For a student or semester-level project, using pretrained models is practical and expected.

In this project:

- CLIP is already trained to understand image-text relationships.
- GPT-2 is already trained to generate natural language.
- The custom part is the trainable mapper between CLIP and GPT-2.

This keeps the project manageable while still requiring real model training.

---

## System Architecture

The model architecture is:

```text
Input Image
    ↓
Pretrained CLIP Vision Encoder
    ↓
Image Embedding
    ↓
Trainable Prefix Mapper
    ↓
GPT-2 Prefix Embeddings
    ↓
Pretrained GPT-2 Language Model
    ↓
Generated Caption
```

---

## Main Components

### 1. CLIP Image Encoder

CLIP is used to convert an image into a dense vector representation. In this project, the CLIP vision model is frozen, meaning its weights are not updated during training.

Model used:

```text
openai/clip-vit-base-patch32
```

The notebook uses:

```python
CLIPVisionModelWithProjection
```

This gives image embeddings through:

```python
outputs.image_embeds
```

---

### 2. GPT-2 Language Model

GPT-2 is used to generate captions from prefix embeddings.

Model used:

```text
gpt2
```

GPT-2 is also frozen in the basic version of the project. It receives image information through learned prefix embeddings.

---

### 3. Trainable Prefix Mapper

The prefix mapper is the main trainable part of the project.

Its job is to convert:

```text
CLIP image embedding → GPT-2 prefix embeddings
```

Two mapper versions can be used:

#### Version 1: MLP Mapper

Simple baseline mapper:

```text
Linear → Tanh → Linear
```

This version is easy to train but can produce generic captions.

#### Version 2: Transformer Mapper

Improved mapper:

```text
CLIP Image Embedding
    ↓
Linear Projection
    ↓
Learnable Prefix Tokens
    ↓
Transformer Encoder
    ↓
GPT-2 Prefix Embeddings
```

The Transformer mapper usually gives better captions because it has more capacity than a simple MLP.

---

## Dataset

The project uses the Flickr30k dataset.

Recommended Hugging Face dataset:

```text
lmms-lab/flickr30k
```

This dataset contains images and multiple human-written captions for each image.

A typical image may have captions like:

```text
- A group of people are shopping at an outdoor market.
- People are browsing clothes at a flea market.
- Two women are looking at shirts at an outside vendor.
```

For the first training run, use a subset of the dataset to reduce training time.

Recommended starting values:

```python
TRAIN_SIZE = 3000
VAL_SIZE = 300
EPOCHS = 3
```

For better results:

```python
TRAIN_SIZE = 15000
VAL_SIZE = 1000
EPOCHS = 6
```

---

## Folder Structure

Recommended folder structure:

```text
clip-caption-project/
│
├── data/
│   └── Dataset is downloaded automatically through Hugging Face
│
├── outputs/
│   ├── clip_gpt2_prefix.pt
│   └── sample_captions.txt
│
├── samples/
│   └── test.jpg
│
├── notebook/
│   └── clip_gpt2_captioning_colab.ipynb
│
└── README.md
```

In Google Colab, the main project folder is:

```text
/content/drive/MyDrive/clip-caption-project
```

---

## Tools and Frameworks

The project uses:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Google Colab
- PIL
- Matplotlib
- NLTK

Install required libraries in Colab:

```bash
pip install -q transformers datasets accelerate pillow tqdm nltk matplotlib
```

---

## Google Colab Setup

### Step 1: Enable GPU

In Google Colab:

```text
Runtime → Change runtime type → GPU
```

### Step 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Create Project Folders

```python
import os

PROJECT_DIR = "/content/drive/MyDrive/clip-caption-project"

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/outputs", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/samples", exist_ok=True)
```

---

## Model Configuration

Recommended baseline configuration:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_NAME = "lmms-lab/flickr30k"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GPT2_MODEL_NAME = "gpt2"

PREFIX_LENGTH = 10
MAX_CAPTION_LENGTH = 40
BATCH_SIZE = 8
EPOCHS = 3

TRAIN_SIZE = 3000
VAL_SIZE = 300

LEARNING_RATE = 2e-4
```

Recommended improved configuration:

```python
PREFIX_LENGTH = 10
MAX_CAPTION_LENGTH = 40
BATCH_SIZE = 8
EPOCHS = 6

TRAIN_SIZE = 15000
VAL_SIZE = 1000

LEARNING_RATE = 2e-4
```

---

## Training Process

The training process is:

1. Load Flickr30k dataset.
2. Load pretrained CLIP and GPT-2.
3. Freeze CLIP and GPT-2.
4. Train only the prefix mapper.
5. Save the trained mapper.
6. Generate captions for validation images.
7. Evaluate using BLEU score and human comparison.

During training, each image is processed by CLIP to produce image features. The mapper converts those features into GPT-2 prefix embeddings. GPT-2 then learns to generate the correct caption conditioned on those prefix embeddings.

---

## Saving the Model

The trained prefix mapper can be saved using:

```python
torch.save({
    "prefix_mapper_state_dict": model.prefix_mapper.state_dict(),
    "prefix_length": PREFIX_LENGTH,
    "clip_model_name": CLIP_MODEL_NAME,
    "gpt2_model_name": GPT2_MODEL_NAME
}, SAVE_PATH)
```

Recommended save path:

```text
/content/drive/MyDrive/clip-caption-project/outputs/clip_gpt2_prefix.pt
```

Only the mapper needs to be saved because CLIP and GPT-2 can be loaded again from Hugging Face.

---

## Caption Generation

After training, the model can generate captions for validation images or custom images.

Example output from an early model:

```text
Generated caption:
A group of people are shopping at a local grocery store.

Real captions:
- Two women are looking at shirts at an outside flea market.
- People are shopping and browsing at outdoor vendors.
```

This shows that the model understands the general scene but may still produce generic captions.

---

## Evaluation

The project can be evaluated using:

### 1. Validation Loss

Validation loss shows how well the model predicts captions on unseen images.

### 2. BLEU Score

BLEU compares generated captions with human-written reference captions.

Useful BLEU metrics:

```text
BLEU-1: Measures unigram word overlap.
BLEU-4: Measures longer phrase overlap.
```

### 3. Human Evaluation

Human evaluation is also useful because automatic metrics are not perfect for image captioning.

Human evaluation can check:

- Is the caption relevant to the image?
- Is the caption grammatically correct?
- Does the caption mention the main objects?
- Does the caption avoid hallucinations?
- Is the caption too generic?

---

## Common Problems and Fixes

### Problem 1: Dataset script error

Error:

```text
RuntimeError: Dataset scripts are no longer supported
```

Fix:

Use:

```python
DATASET_NAME = "lmms-lab/flickr30k"
```

instead of:

```python
DATASET_NAME = "nlphuji/flickr30k"
```

---

### Problem 2: CLIP output is not a tensor

Error:

```text
TypeError: linear(): argument 'input' must be Tensor, not BaseModelOutputWithPooling
```

Fix:

Use:

```python
CLIPVisionModelWithProjection
```

and extract features using:

```python
outputs.image_embeds
```

---

### Problem 3: CLIPVisionModelWithProjection has no get_image_features

Error:

```text
AttributeError: 'CLIPVisionModelWithProjection' object has no attribute 'get_image_features'
```

Fix:

Use the model helper function:

```python
image_features = model.get_clip_image_features(pixel_values)
```

not:

```python
model.clip.get_image_features(...)
```

---

### Problem 4: Captions are repetitive

Example bad caption:

```text
A man is wearing a red shirt and a blue shirt and a red shirt.
```

Fix generation settings:

```python
no_repeat_ngram_size=3
repetition_penalty=1.25
max_new_tokens=25
length_penalty=0.9
```

---

## Results Discussion

The first version of the model may generate captions that are correct at a high level but not very detailed.

Example:

```text
Generated: A group of people are shopping at a local grocery store.
Actual: People are shopping and browsing at outdoor vendors.
```

This is acceptable for a first version because the model identifies the shopping scene. However, it may miss important details such as:

- outdoor market
- roller derby
- skates
- specific object names
- exact actions

The improved Transformer mapper helps reduce generic output and improves the connection between image features and language generation.

---

## Limitations

This project has several limitations:

1. CLIP produces global image embeddings, so it may miss fine details.
2. GPT-2 was not originally trained for image captioning.
3. Frozen GPT-2 cannot fully adapt to visual caption style.
4. Flickr30k is smaller than COCO.
5. BLEU does not always match human judgment.
6. Generated captions may hallucinate objects.
7. Captions can become repetitive if generation settings are weak.

---

# Future Improvements

## Improvement 1: Train on More Data

The easiest improvement is to increase the training size.

Current small setup:

```python
TRAIN_SIZE = 3000
EPOCHS = 3
```

Better setup:

```python
TRAIN_SIZE = 15000
EPOCHS = 6
```

Best setup if Colab allows:

```python
TRAIN_SIZE = 30000
EPOCHS = 8
```

More training data helps the mapper learn stronger image-caption relationships.

---

## Improvement 2: Use Transformer Mapper

The Transformer mapper is better than the MLP mapper because it can model interactions between the projected image embedding and learnable prefix tokens.

Report comparison:

```text
Version 1: MLP mapper
Version 2: Transformer mapper
```

Compare both versions using:

- validation loss
- BLEU-1
- BLEU-4
- sample generated captions
- human evaluation

---

## Improvement 3: Fine-Tune Last GPT-2 Layers

In the basic version, GPT-2 is frozen. This keeps training cheap but limits caption quality.

A later improvement is to unfreeze only the last 2 GPT-2 transformer blocks:

```python
for param in model.gpt2.parameters():
    param.requires_grad = False

for block in model.gpt2.transformer.h[-2:]:
    for param in block.parameters():
        param.requires_grad = True

for param in model.gpt2.transformer.ln_f.parameters():
    param.requires_grad = True
```

Use a smaller learning rate for GPT-2:

```python
optimizer = torch.optim.AdamW(
    [
        {"params": model.prefix_mapper.parameters(), "lr": 2e-4},
        {"params": gpt2_trainable_params, "lr": 1e-5}
    ],
    weight_decay=0.01
)
```

This allows GPT-2 to adapt slightly to the image captioning task without fully fine-tuning the entire model.

---

## Improvement 4: Use a Larger Language Model

GPT-2 is small and old compared to newer language models. A future version can replace GPT-2 with:

- GPT-2 Medium
- DistilGPT-2 for faster experiments
- T5-small
- OPT-125M
- TinyLlama, if resources allow

However, larger models require more GPU memory.

---

## Improvement 5: Use MS-COCO Dataset

Flickr30k is good for starting, but MS-COCO is a stronger dataset for image captioning.

Future work:

```text
Train on Flickr30k first.
Then train or evaluate on MS-COCO captions.
```

COCO has more images and more diverse captions, which can improve generalization.

---

## Improvement 6: Add CIDEr Evaluation

The project currently uses BLEU because it is easier to implement.

For a stronger image captioning project, add:

- CIDEr
- METEOR
- ROUGE-L
- SPICE

CIDEr is especially common for image captioning.

---

## Improvement 7: Compare with BLIP

BLIP is a pretrained image captioning model. It can be used as a baseline.

Comparison:

```text
Our model: CLIP + GPT-2 + trainable mapper
Baseline: pretrained BLIP image captioning
```

This comparison makes the project stronger because it shows how your custom model performs against a model designed specifically for captioning.

---

## Improvement 8: Add Attention Visualization

A possible research-style improvement is to visualize where the model focuses in the image.

Options:

- Grad-CAM for CLIP vision encoder
- similarity heatmaps
- image patch relevance

This improves explainability.

---

## Improvement 9: Improve Caption Decoding

Current generation uses beam search. Future decoding methods:

```text
Beam search
Top-k sampling
Top-p nucleus sampling
Temperature sampling
Contrastive search
```

Different decoding strategies can produce captions that are more fluent or more diverse.

---

## Improvement 10: Train Multiple Experiments

For a strong report, run multiple experiments:

| Experiment | CLIP | GPT-2 | Mapper | Trainable Parameters |
|---|---|---|---|---|
| Baseline | Frozen | Frozen | MLP | Mapper only |
| Improved | Frozen | Frozen | Transformer | Mapper only |
| Advanced | Frozen | Last 2 layers trainable | Transformer | Mapper + GPT-2 last layers |

Then compare:

- training loss
- validation loss
- BLEU-1
- BLEU-4
- example captions
- human evaluation

---

## Suggested Report Structure

```text
1. Introduction
2. Problem Statement
3. Objectives
4. Related Work
5. Dataset Description
6. Model Architecture
7. Methodology
8. Training Setup
9. Evaluation Metrics
10. Results
11. Discussion
12. Limitations
13. Future Improvements
14. Conclusion
15. References
```

---

## Conclusion

This project demonstrates a practical multimodal image captioning system using pretrained CLIP and GPT-2. The main contribution is the trainable prefix mapper that connects image embeddings to language generation.

The basic model can generate reasonable captions, while the improved Transformer mapper and optional GPT-2 fine-tuning can significantly improve results.

This project is suitable for a multimodal AI course because it covers:

- image-text representation
- pretrained vision-language models
- language generation
- cross-modal mapping
- captioning evaluation
- practical deep learning training in Google Colab

---

## Final Recommended Version

For final submission, the best balanced version is:

```text
Dataset: Flickr30k
Image Encoder: Frozen CLIPVisionModelWithProjection
Language Model: Frozen GPT-2
Mapper: Transformer-based prefix mapper
Training: 15000 images, 6 epochs
Evaluation: BLEU-1, BLEU-4, validation loss, human evaluation
```

For an advanced version:

```text
Dataset: Flickr30k or COCO subset
Image Encoder: Frozen CLIP
Language Model: GPT-2 with last 2 layers fine-tuned
Mapper: Transformer-based prefix mapper
Evaluation: BLEU, CIDEr, human evaluation
```
