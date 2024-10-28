from datasets import load_dataset
import torch
from transformers import XLMRobertaForSequenceClassification, Trainer, TrainingArguments
# Load the IMDb dataset
dataset = load_dataset("imdb")
from transformers import AutoTokenizer, AutoModelForMaskedLM

# load token and model
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
# Load model
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Convert datasets to PyTorch tensors
tokenized_dataset = tokenized_dataset.map(lambda x: {'labels': x['label']}, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])