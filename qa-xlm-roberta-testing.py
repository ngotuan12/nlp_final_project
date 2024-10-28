from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering
import torch
import numpy as np
from typing import Dict

class QAModel:
    def __init__(self, model_path="./qa-xlm-roberta"):
        """
        Khởi tạo model và tokenizer từ đường dẫn đã lưu
        """
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_path)
        self.model = XLMRobertaForQuestionAnswering.from_pretrained(model_path)
        self.model.eval()  # Chuyển sang chế độ evaluation
        
        # Nếu có GPU thì sử dụng
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        print(f"Đã load model từ {model_path}")
        print(f"Đang sử dụng device: {self.device}")

    def predict(self, question: str, context: str, max_length: int = 384) -> Dict:
        """
        Dự đoán câu trả lời cho một câu hỏi
        """
        # Chuẩn bị input
        inputs = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            max_length=max_length,
            truncation="only_second",
            stride=128,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Chuyển input lên GPU nếu có
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Lấy offset mapping và sequence ids
        offset_mapping = inputs.pop("offset_mapping")[0]
        sequence_ids = inputs.sequence_ids(0)
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Lấy logits và chuyển về numpy
        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()
        
        # Chỉ lấy tokens thuộc context (sequence_id = 1)
        context_mask = [i for i, s in enumerate(sequence_ids) if s == 1]
        start_logits = [float('-inf') if i not in context_mask else l for i, l in enumerate(start_logits)]
        end_logits = [float('-inf') if i not in context_mask else l for i, l in enumerate(end_logits)]
        
        # Tìm vị trí start và end tốt nhất
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
    
    def predict_batch(self, questions: list, contexts: list, max_length: int = 384) -> list:
        """
        Dự đoán câu trả lời cho nhiều câu hỏi cùng lúc
        """
        if len(questions) != len(contexts):
            raise ValueError("Số lượng câu hỏi và contexts phải bằng nhau")
            
        results = []
        batch_size = 8  # Có thể điều chỉnh batch size tùy theo GPU memory
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_contexts = contexts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_questions,
                batch_contexts,
                return_tensors="pt",
                max_length=max_length,
                truncation="only_second",
                stride=128,
                padding=True,
                return_offsets_mapping=True,
            )
            
            # Chuyển inputs lên device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            offset_mapping = inputs.pop("offset_mapping")
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Xử lý từng sample trong batch
            for j in range(len(batch_questions)):
                sequence_ids = inputs.sequence_ids(j)
                offsets = offset_mapping[j]
                
                # Lấy logits cho sample hiện tại
                start_logits = outputs.start_logits[j].cpu().numpy()
                end_logits = outputs.end_logits[j].cpu().numpy()
                
                # Mask cho context tokens
                context_mask = [i for i, s in enumerate(sequence_ids) if s == 1]
                masked_start_logits = [float('-inf') if i not in context_mask else l for i, l in enumerate(start_logits)]
                masked_end_logits = [float('-inf') if i not in context_mask else l for i, l in enumerate(end_logits)]
                
                # Tìm vị trí tốt nhất
                start_idx = int(np.argmax(masked_start_logits))
                end_idx = int(np.argmax(masked_end_logits[start_idx:]) + start_idx)
                
                # Lấy answer text
                answer_start = offsets[start_idx][0]
                answer_end = offsets[end_idx][1]
                answer = batch_contexts[j][answer_start:answer_end]
                
                # Tính confidence
                confidence = float(np.exp(masked_start_logits[start_idx]) * np.exp(masked_end_logits[end_idx]))
                
                results.append({
                    'question': batch_questions[j],
                    'context': batch_contexts[j],
                    'answer': answer,
                    'confidence': confidence,
                    'start_position': answer_start,
                    'end_position': answer_end
                })
        
        return results

def main():
    # Khởi tạo model
    qa_model = QAModel("./qa-xlm-roberta")  # Thay đổi đường dẫn nếu cần
    
    # Ví dụ đơn lẻ
    question = "What is the capital of France?"
    context = "Paris is the capital and largest city of France. It is situated on the river Seine."
    
    result = qa_model.predict(question, context)
    print("\nDự đoán đơn lẻ:")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Ví dụ batch
    questions = [
        "What is the capital of France?",
        "When was Python created?",
        "Who is the CEO of Apple?"
    ]
    
    contexts = [
        "Paris is the capital and largest city of France. It is situated on the river Seine.",
        "Python was created by Guido van Rossum and was released in 1991. It emphasizes code readability.",
        "Tim Cook is the CEO of Apple Inc., succeeding Steve Jobs in 2011. Under his leadership, Apple has become one of the most valuable companies."
    ]
    
    results = qa_model.predict_batch(questions, contexts)
    
    print("\nDự đoán batch:")
    for result in results:
        print(f"\nQuestion: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()