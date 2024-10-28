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

#training_args
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)