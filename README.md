# Tar
LLaVa Based Multi-Model Translator

This project implements a multimodal model inspired by the LLaVA (Large Language and Vision Assistant) architecture. It connects a pre-trained vision encoder (CLIP) with a pre-trained large language model (Gemma) to understand and process both images and text.

The core idea is to train a simple projection layer that maps the output of the vision model into the language model's embedding space. This allows the language model to "see" the image and perform tasks like image captioning or visual question answering.

## Architecture

The model consists of three main components:

1.  **Vision Encoder**: A frozen `openai/clip-vit-base-patch32` model is used to extract feature embeddings from input images.
2.  **Language Model**: A frozen `google/gemma-3-1b-it` model is used for text understanding and generation.
3.  **Projection Layer**: A small, trainable neural network that projects the vision embeddings into the same dimensional space as the language model's word embeddings.

During training, only the projection layer's weights are updated. The vision and language models remain frozen.

## How it Works

1.  An image is passed through the CLIP vision encoder to get image features (`Hv`).
2.  A text caption is passed through the Gemma tokenizer and embedding layer to get text embeddings (`Hq`).
3.  The image features are passed through the trainable projection layer to align their dimension with the text embeddings.
4.  The projected image embeddings and the text embeddings are concatenated.
5.  This combined sequence of embeddings is fed into the Gemma language model to predict the original text caption.
6.  The loss is calculated only on the text part of the sequence, and backpropagation updates the weights of the projection layer.

## Project Structure

```
├── main.py                 # Main script to run training
├── dataset.py              # Handles loading and splitting the dataset
├── models/
│   ├── clip_model.py       # Wrapper for the CLIP vision model
│   ├── gemma_model.py      # Wrapper for the Gemma language model
│   └── llava_model.py      # Contains the Projection Layer and custom Dataset
└── README.md               # This file
```

## Prerequisites

You need to have Python and PyTorch installed. You can install the required libraries using pip:

```bash
pip install torch transformers datasets pillow requests
```

Make sure you have sufficient memory (CPU/GPU RAM) to load the models. The Gemma 1B model will be loaded onto the available device.

## How to Run

To start the training process, simply run the `main.py` script:

```bash
python main.py
```

The script will:
1.  Download the `conceptual_captions` dataset (a 10k subset).
2.  Load the pre-trained CLIP and Gemma models.
3.  Train the projection layer for one epoch.
4.  Save the trained projection layer weights to `checkpoint.pth`.

During training, a progress bar will show the status for the current epoch. The average loss will be printed upon completion.

PS. this project needs cleaned datasets to train.