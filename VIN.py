from transformers import AutoTokenizer, AutoModelForCausalLM

# Tải mô hình và tokenizer từ Hugging Face
tokenizer = AutoTokenizer.from_pretrained("vinai/RecGPT-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("vinai/RecGPT-7B-Instruct")

# Lịch sử mua sắm của người dùng
user_history = """
1. Sản phẩm: Giày thể thao Nike Air Max
2. Sản phẩm: Áo thể thao Adidas Dri-Fit
3. Sản phẩm: Quần short thể thao Puma
"""

# Tạo prompt đầu vào cho mô hình
prompt = f"Lịch sử mua sắm của người dùng: {user_history}\nDựa trên lịch sử này, gợi ý sản phẩm tiếp theo mà người dùng có thể quan tâm là:"

# Tokenize đầu vào
inputs = tokenizer(prompt, return_tensors="pt")

# Mô hình dự đoán
outputs = model.generate(**inputs, max_length=100)

# Giải mã kết quả
recommendation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(recommendation)