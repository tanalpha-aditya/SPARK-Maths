import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import time

# Ensure correct GPU is used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset_path = "../../Dataset/MATH/Pre_processed_test/filtered_test_serial.csv"
df = pd.read_csv(dataset_path)
# df = pd.read_csv("../../Dataset/GSM8k/modified/main_test.csv")
output_csv = "math_m1_answers.csv"
checkpoint_csv = "checkpoint_math_answers.csv"

# Check if checkpoint exists (Resume from last save)
if os.path.exists(checkpoint_csv):
    print(f"Resuming from checkpoint: {checkpoint_csv}")
    df_checkpoint = pd.read_csv(checkpoint_csv)
    df = df[~df["Serial Number"].isin(df_checkpoint["Serial Number"])]  # Remove already processed rows
else:
    df_checkpoint = pd.DataFrame(columns=["Serial Number", "question", "model_solution"])  # Initialize empty checkpoint

# Load tokenizer and model
model_name = "unsloth/Qwen2.5-Math-1.5B-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

def get_model_solution(question, serial_number, max_retries=5):
    """Generates an answer for the given question using the model with retries."""
    text = f"<|system|>\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n<|user|>\n{question}\n<|assistant|>"
    model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}  # Move inputs to GPU

    attempt = 0
    while attempt < max_retries:
        try:
            with torch.no_grad():
                generated_ids = model.generate(**model_inputs, max_new_tokens=512)
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA Out of Memory! Skipping question {serial_number}.")
            torch.cuda.empty_cache()
            return "OOM_Error"
        except Exception as e:
            print(f"Error processing question {serial_number} (attempt {attempt + 1}/{max_retries}): {e}")
            attempt += 1
            time.sleep(2)  # Small delay before retrying

    print(f"Skipping question {serial_number} after {max_retries} failed attempts.")
    return "Error_After_Max_Retries"

# Process each question
checkpoint_data = []
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Questions"):
    serial_number = row["Serial Number"]
    question = row["question"]
    model_solution = get_model_solution(question, serial_number)

    # Save checkpoint after each question
    checkpoint_data.append([serial_number, question, model_solution])
    checkpoint_df = pd.DataFrame(checkpoint_data, columns=["Serial Number", "question", "model_solution"])
    checkpoint_df.to_csv(checkpoint_csv, mode='a', index=False, header=not os.path.exists(checkpoint_csv))
    checkpoint_data = []  # Reset buffer after saving

# Save final results
df.to_csv(output_csv, index=False)
print(f"Model answers saved to: {output_csv}")

