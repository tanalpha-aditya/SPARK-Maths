import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
from tqdm import tqdm

# Load dataset
df = pd.read_csv("../../Dataset/GSM8k/modified/main_test.csv")

# Model name
torch.cuda.set_device(0)
model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 4-bit Quantization Configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with balanced device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="balanced"  # Ensures proper distribution across available GPUs
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# File to save results
output_file = "output.txt"

def extract_answer(response):
    """Extracts the answer from model response using regex."""
    match = re.search(r"\\boxed{(\d+)}", response)
    return int(match.group(1)) if match else None

def get_model_answer(question):
    """Generates an answer for the given question using the model."""
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return extract_answer(response), response

# Tracking accuracy
correct = 0
total = len(df)

# Open file in append mode
with open(output_file, "a") as f:
    for index, row in tqdm(df.iterrows(), total=total, desc="Processing Questions"):
        question = row["question"]
        true_value = row["value"]
        
        model_answer, response_text = get_model_answer(question)
        
        # Save to file
        f.write(f"Question {index + 1}: {question}\n")
        f.write(f"Model Response: {response_text}\n")
        f.write(f"Extracted Answer: {model_answer}, True Answer: {true_value}\n")
        f.write("=" * 80 + "\n")  # Separator
        f.flush()  # Ensure data is written immediately
        
        # Accuracy check
        if model_answer == true_value:
            correct += 1

accuracy = (correct / total) * 100
print(f"Model Accuracy: {accuracy:.2f}%")

# Save final accuracy to file
with open(output_file, "a") as f:
    f.write(f"\nFinal Accuracy: {accuracy:.2f}%\n")
