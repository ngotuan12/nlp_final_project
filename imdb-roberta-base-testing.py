from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch

def load_model(model_path="./imdb-xlm-roberta"):
    """
    Load model và tokenizer đã lưu
    """
    try:
        # Load tokenizer và model
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
        model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
        print(f"Đã load model thành công từ {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Lỗi khi load model: {str(e)}")
        return None, None

def predict_sentiment(text: str, model, tokenizer):
    """
    Dự đoán sentiment cho văn bản mới
    """
    # Chuyển model sang eval mode
    model.eval()
    
    # Chuẩn bị input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Lấy kết quả
    predicted_class = predictions.argmax().item()
    confidence = predictions[0][predicted_class].item()
    
    return {
        'text': text,
        'sentiment': 'Positive' if predicted_class == 1 else 'Negative',
        'confidence': confidence,
        'probabilities': {
            'positive': predictions[0][1].item(),
            'negative': predictions[0][0].item()
        }
    }

def batch_predict(texts: list, model, tokenizer, batch_size=32):
    """
    Dự đoán cho nhiều văn bản cùng lúc
    """
    model.eval()
    results = []
    
    # Xử lý theo batch
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        
        # Dự đoán
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Xử lý kết quả cho từng text trong batch
        for j, text in enumerate(batch_texts):
            predicted_class = predictions[j].argmax().item()
            confidence = predictions[j][predicted_class].item()
            
            results.append({
                'text': text,
                'sentiment': 'Positive' if predicted_class == 1 else 'Negative',
                'confidence': confidence,
                'probabilities': {
                    'positive': predictions[j][1].item(),
                    'negative': predictions[j][0].item()
                }
            })
    
    return results

def main():
    # Load model
    model, tokenizer = load_model(model_path='FacebookAI/xlm-roberta-base')
    
    if model is None or tokenizer is None:
        print("Không thể load model. Vui lòng kiểm tra lại đường dẫn.")
        return
    
    # Ví dụ dự đoán đơn lẻ
    text = "This movie was absolutely fantastic! I loved every minute of it."
    result = predict_sentiment(text, model, tokenizer)
    print("\nDự đoán đơn lẻ:")
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probability Positive: {result['probabilities']['positive']:.4f}")
    print(f"Probability Negative: {result['probabilities']['negative']:.4f}")
    
    # Ví dụ dự đoán batch
    texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "What a terrible waste of time. I wouldn't recommend this to anyone.",
        "An okay movie, but nothing special.",
        "One of the best films I've seen this year!",
        "I fell asleep during the movie, it was so boring."
    ]
    
    print("\nDự đoán batch:")
    results = batch_predict(texts, model, tokenizer)
    for result in results:
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probability Positive: {result['probabilities']['positive']:.4f}")
        print(f"Probability Negative: {result['probabilities']['negative']:.4f}")

if __name__ == "__main__":
    main()