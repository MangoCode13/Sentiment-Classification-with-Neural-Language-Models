# Sentiment Classification (DistilBERT, 2-stage fine-tuning)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MangoCode13/Sentiment-Classification-with-Neural-Language-Models/blob/main/stage1_notebook.ipynb)

This repo contains the Stage 1 langugage model training notebook for Assignment 2, model checkpoint, dependencies, and prediction outputs for binary sentiment classification:

- `stage1_notebook.ipynb`
- `model_checkpoint/` with reloadable model artifacts
- `requirements.txt`
- `public_test_predictions.csv` in required submission format

## Environment setup

1. Create a virtual environment:

```bash
python3 -m venv .venv
```

2. Activate (using bash):

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add training and test data from [Assignment 2](https://github.com/judyzhu-gmu/CYSE499_650_Summer2026/blob/main/assignment2/assignment2_instructions.md) to the folder containing the notebook:
* `train.csv`
* `public_test.csv`

## How to run stage1_notebook.ipynb

1. Open stage1_notebook.ipynb
2. Select the workspace Python environment (`.venv`) as the notebook kernel.
3. Run all cells from top to bottom (or use “Run All”).
4. Wait for training/tuning cells to finish.
5. Confirm outputs in the final section:
	 - `public_test_predictions.csv`
	 - `model_checkpoint/`

## What the notebook does

- Loads `train.csv` and `public_test.csv` and the pretrained model: [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- Cleans text and tokenizes with a head+tail strategy to fit 512 tokens.
- Builds a stratified train/validation split from training data only.
- Defines class weights for training (`class_weights = [3.0, 1.0]`) and evaluation metrics.
- Two-stage model training:
	- Stage 1: classifier head only.
	- Runs hyperparameter tuning for the first stage of training and saves best settings.
	- Stage 2: unfreezes top transformer layers with a lower learning rate.
- Evaluates on validation split and prints metrics.
- Generates submission predictions and saves a reloadable checkpoint.
