from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from typing import Dict

def compute_metrics(pred):
    """
    Tính toán các metrics đánh giá model
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def load_and_prepare_data(max_samples=5000):
    """
    Load IMDB dataset và chuẩn bị dữ liệu với số lượng mẫu giới hạn
    """
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    
    # Giảm kích thước dataset để test nhanh
    if max_samples:
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(max_samples))
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(max_samples//10))
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    
    def preprocess_function(examples):
        return {
            **tokenizer(
                examples["text"],
                truncation=True,
                max_length=256,  # Giảm max_length để tăng tốc
                padding=False
            ),
            "labels": examples["label"]
        }
    
    # Áp dụng preprocessing với số lượng process nhiều hơn
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,  # Tăng số process để xử lý parallel
        remove_columns=dataset["train"].column_names
    )
    
    tokenized_datasets = tokenized_datasets.with_format("torch")
    
    return tokenized_datasets, tokenizer

def train_model(tokenized_datasets, tokenizer):
    """
    Fine-tune model với các tối ưu cho tốc độ
    """
    model = XLMRobertaForSequenceClassification.from_pretrained(
        "FacebookAI/xlm-roberta-base",
        num_labels=2
    )
    
    # Thiết lập training arguments với các tối ưu
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=3e-5,  # Tăng learning rate
        per_device_train_batch_size=32,  # Tăng batch size
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=100,  # Đánh giá thường xuyên hơn
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=True,  # Bật mixed precision training
        gradient_accumulation_steps=4,  # Thêm gradient accumulation
        warmup_ratio=0.1,  # Thêm learning rate warmup
        dataloader_num_workers=4,  # Tăng số worker cho DataLoader
        gradient_checkpointing=True,  # Bật gradient checkpointing để tiết kiệm bộ nhớ
        report_to="none",  # Tắt reporting để tăng tốc
    )
    
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8  # Tối ưu cho mixed precision training
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print("Bắt đầu training...")
    trainer.train()
    
    return trainer, model

def evaluate_model(trainer):
    """
    Đánh giá model sau khi train
    """
    eval_results = trainer.evaluate()
    print("\nKết quả đánh giá:")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"Precision: {eval_results['eval_precision']:.4f}")
    print(f"Recall: {eval_results['eval_recall']:.4f}")
    
    return eval_results

def predict_sentiment(text: str, model, tokenizer) -> Dict:
    """
    Dự đoán sentiment cho một văn bản mới
    """
    # Chuẩn bị input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,  # Giảm max_length để phù hợp với training
        padding=True
    )
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    # Chuyển đổi kết quả
    predicted_class = predictions.argmax().item()
    confidence = predictions[0][predicted_class].item()
    
    return {
        'sentiment': 'Positive' if predicted_class == 1 else 'Negative',
        'confidence': confidence,
        'text': text
    }

def main():
    print("Loading and preparing data...")
    # Chỉ dùng 5000 mẫu để test nhanh
    tokenized_datasets, tokenizer = load_and_prepare_data(max_samples=5000)
    
    print("\nKhởi tạo và training model...")
    trainer, model = train_model(tokenized_datasets, tokenizer)
    
    print("\nĐánh giá model...")
    eval_results = evaluate_model(trainer)
    
    # Lưu model
    print("\nLưu model...")
    output_dir = "./imdb-xlm-roberta"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Test với một số ví dụ
    test_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "What a terrible waste of time. I wouldn't recommend this to anyone."
    ]
    
    print("\nTest dự đoán với một số ví dụ:")
    for text in test_texts:
        result = predict_sentiment(text, model, tokenizer)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()