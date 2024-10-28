from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
# Load model
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
# Tokenize function
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize, batched=True)


# Load model
model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)