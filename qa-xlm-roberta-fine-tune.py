from transformers import (
    XLMRobertaTokenizerFast,  # Thay đổi sang Fast Tokenizer
    XLMRobertaForQuestionAnswering,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset
import numpy as np
import torch
import evaluate
from typing import Dict
import collections
from tqdm.auto import tqdm

def prepare_train_features(examples, tokenizer, max_length=384, stride=128):
    """
    Chuẩn bị features cho QA task với Fast Tokenizer
    """
    # Một số câu hỏi có thể có nhiều answers hoặc không có answer
    first_answers = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in examples["answers"]]
    
    # Tokenize questions và contexts
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Cho phép nhiều features cho một example
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Khởi tạo các features mới
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Lấy sequence ID để phân biệt question với context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Lấy example_id từ mapping
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]

        # Nếu không có answer, set CLS index là answer
        if len(answer["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index của answer
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            # Token index của context
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Kiểm tra answer có nằm trong span không
            if not (offsets[token_start_index][0] <= start_char and 
                   offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Token start và end positions
                while token_start_index < len(offsets) and \
                        offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def load_and_prepare_data(max_samples=None):
    """
    Load SQuAD dataset và chuẩn bị cho training
    """
    # Load dataset
    datasets = load_dataset("squad")
    
    # Giảm kích thước dataset nếu cần
    if max_samples:
        datasets["train"] = datasets["train"].shuffle(seed=42).select(range(max_samples))
        datasets["validation"] = datasets["validation"].shuffle(seed=42).select(range(max_samples//10))
    
    # Load Fast Tokenizer
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("FacebookAI/xlm-roberta-base")
    
    # Tokenize datasets
    tokenized_datasets = datasets.map(
        lambda x: prepare_train_features(x, tokenizer),
        batched=True,
        remove_columns=datasets["train"].column_names,
        num_proc=4
    )
    
    return tokenized_datasets, tokenizer

def train_model(tokenized_datasets, tokenizer):
    """
    Fine-tune model cho QA task
    """
    model = XLMRobertaForQuestionAnswering.from_pretrained("FacebookAI/xlm-roberta-base")
    
    training_args = TrainingArguments(
        output_dir="./results_qa",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        push_to_hub=False,
        fp16=True,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Bắt đầu training...")
    trainer.train()
    
    return trainer, model

def predict_answer(question: str, context: str, model, tokenizer) -> Dict:
    """
    Dự đoán câu trả lời cho câu hỏi mới
    """
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=384,
        truncation="only_second",
        stride=128,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = inputs.pop("offset_mapping")[0]
    sequence_ids = inputs.sequence_ids(0)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Lấy vị trí có xác suất cao nhất cho start và end
    start_logits = outputs.start_logits[0].numpy()
    end_logits = outputs.end_logits[0].numpy()
    
    # Chỉ lấy tokens thuộc context (sequence_id = 1)
    context_mask = [i for i, s in enumerate(sequence_ids) if s == 1]
    start_logits = [float('-inf') if i not in context_mask else l for i, l in enumerate(start_logits)]
    end_logits = [float('-inf') if i not in context_mask else l for i, l in enumerate(end_logits)]
    
    start_idx = int(np.argmax(start_logits))
    end_idx = int(np.argmax(end_logits[start_idx:]) + start_idx)
    
    # Chuyển token positions thành text spans
    answer_start = offset_mapping[start_idx][0]
    answer_end = offset_mapping[end_idx][1]
    answer = context[answer_start:answer_end]
    
    # Tính confidence score
    confidence = float(np.exp(start_logits[start_idx]) * np.exp(end_logits[end_idx]))
    
    return {
        'question': question,
        'context': context,
        'answer': answer,
        'confidence': confidence,
        'start_position': answer_start,
        'end_position': answer_end
    }

def main():
    print("Loading and preparing data...")
    tokenized_datasets, tokenizer = load_and_prepare_data(max_samples=1000)
    
    print("\nKhởi tạo và training model...")
    trainer, model = train_model(tokenized_datasets, tokenizer)
    
    print("\nLưu model...")
    output_dir = "./qa-xlm-roberta"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Test với một số ví dụ
    test_examples = [
        {
            'question': "What is the capital of France?",
            'context': "Paris is the capital and largest city of France. It is situated on the river Seine."
        },
        {
            'question': "When was Python created?",
            'context': "Python was created by Guido van Rossum and was released in 1991. It emphasizes code readability."
        }
    ]
    
    print("\nTest dự đoán:")
    for example in test_examples:
        result = predict_answer(example['question'], example['context'], model, tokenizer)
        print(f"\nQuestion: {result['question']}")
        print(f"Context: {result['context']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()