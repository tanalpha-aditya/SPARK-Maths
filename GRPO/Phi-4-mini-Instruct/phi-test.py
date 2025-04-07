import torch
import os
import json
import re
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from tqdm import tqdm # Progress bar
from typing import List, Dict, Union
import torch._dynamo

# --- Configuration ---
# !! IMPORTANT: Make sure these match your training setup !!
BASE_MODEL_NAME = "unsloth/Phi-3.5-mini-instruct"
# Path where your trained adapters are saved (inside the training OUTPUT_DIR)
TRAINED_MODEL_PATH = "outputs/Phi-GRPO-Combined"
OUTPUT_RESULTS_DIR = "test_results" # Directory to save detailed results
MAX_SEQ_LENGTH = 768 # Must match training
MAX_NEW_TOKENS = MAX_SEQ_LENGTH - 256 # Max tokens to generate (adjust if needed)
# Use lower precision if you run into memory issues, but fp16 is generally good for inference
LOAD_IN_4BIT = False # Usually False for inference unless memory constrained
LOAD_IN_8BIT = False # Alternative if 4bit causes issues

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

# --- Helper Functions (Copied/adapted from your training script) ---

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
only integer answer without verbose
</answer>
"""

def extract_xml_answer(text: str) -> str:
    # Try to extract from LaTeX boxed format first (if your model might produce it)
    latex_match = re.search(r"\\boxed\{([\d\.,]+)\}", text) # Allow commas/decimals
    if latex_match:
        extracted = latex_match.group(1).replace(",", "") # Remove commas for comparison
    else:
        # If LaTeX extraction fails, extract from XML <answer> tags
        # Make the pattern less strict about newlines before/after content
        answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
             extracted = answer_match.group(1).strip().replace(",", "") # Remove commas
        else:
            extracted = None # Explicitly set to None if not found

    # Fallback: Extract last number if nothing else worked
    if extracted is None:
        # Find all sequences of digits, possibly with decimals or commas
        # Look from the end of the string backwards
        number_matches = re.findall(r"([\d\.,]+)", text)
        if number_matches:
             # Take the last number found, remove commas
             extracted = number_matches[-1].replace(",", "")
        else:
             extracted = None # Still not found

    # Final check and cleanup
    if extracted is not None:
        # Remove any potential trailing periods if it looks like a whole number
        if '.' in extracted and extracted.endswith('.'):
           extracted = extracted.rstrip('.')
        # Handle cases like ".5" -> "0.5"
        if extracted.startswith('.'):
            extracted = '0' + extracted
        # Remove trailing ".0" or ".00" etc. for integer comparison if needed,
        # but be careful not to remove necessary decimals
        if '.' in extracted and float(extracted) == int(float(extracted)):
             extracted = str(int(float(extracted)))
        return extracted
    else:
        return "N/A" # Return specific string if no answer found


def extract_hash_answer(text: str) -> str | None:
    """Extracts answer after ####, handling potential formatting issues."""
    if "####" not in text:
        # Fallback: Try to extract the last number if #### is missing
        number_matches = re.findall(r"([\d\.,]+)", text)
        if number_matches:
            answer_text = number_matches[-1].strip().replace(",", "")
        else:
            return "N/A" # No clear answer found
    else:
        answer_text = text.split("####")[-1].strip().replace(",", "") # Take last part

    # Clean up potential non-numeric chars sometimes left
    answer_text = re.sub(r"[^\d\.\-]", "", answer_text) # Keep digits, dot, minus

    # Handle potential empty strings after cleanup
    if not answer_text:
        return "N/A"

    # Standardize number format (optional but good practice)
    try:
        # Convert to float first to handle decimals, then int if it's a whole number
        num = float(answer_text)
        if num == int(num):
            return str(int(num))
        else:
            return str(num) # Keep as float string if decimal
    except ValueError:
        # If conversion fails, return N/A or the cleaned text? Let's return N/A
        return "N/A"


def get_gsm8k_questions(split="test") -> Dataset:
    data = load_dataset("openai/gsm8k", "main", split=split)
    data = data.map(
        lambda x: {
            "prompt": x["question"], # Keep original prompt for reference
            "chat_prompt": [ # Format for the model
                {"role": "system", "content" : SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]}
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    # Filter out examples where answer extraction failed
    original_count = len(data)
    data = data.filter(lambda x: x["answer"] is not None and x["answer"] != "N/A")
    filtered_count = len(data)
    if original_count != filtered_count:
        print(f"Warning (GSM8K {split}): Filtered out {original_count - filtered_count} examples due to missing/invalid ground truth answers.")
    return data


def get_math500_questions(split="test") -> Dataset:
    data = load_dataset("HuggingFaceH4/MATH-500", split=split)
    def process_math_example(x):
        # Extract answer using the XML/Latex extractor on the ground truth
        extracted_gt = extract_xml_answer(x['answer'])
        return {
            "prompt": x["problem"], # Keep original prompt
            "chat_prompt": [ # Format for the model
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["problem"]}
            ],
            "answer": extracted_gt, # Use the extracted ground truth
        }
    data = data.map(process_math_example)
    # Filter out examples where answer extraction failed
    original_count = len(data)
    data = data.filter(lambda x: x["answer"] is not None and x["answer"] != "N/A")
    filtered_count = len(data)
    if original_count != filtered_count:
        print(f"Warning (MATH-500 {split}): Filtered out {original_count - filtered_count} examples due to missing/invalid ground truth answers.")

    return data


# --- Model Loading ---
@torch.inference_mode() # Essential for efficient inference
def load_trained_model(base_model_name, adapter_path):
    print(f"Loading base model: {base_model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, # Auto-detects (usually bfloat16 or float16)
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        # token = "hf_...", # Add your Hugging Face token if needed
    )

    print(f"Applying LoRA adapters from: {adapter_path}")
    # No need to call get_peft_model again, just load the adapters
    model.load_adapter(adapter_path)
    print("Adapters loaded successfully.")

    # Important for generation: set padding correctly
    tokenizer.padding_side = "left" # Use left padding for batch generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Set pad token if missing

    return model, tokenizer

# --- Generation Function ---
@torch.inference_mode()
def generate_responses(model, tokenizer, dataset, batch_size=4):
    results = []
    model.eval() # Set model to evaluation mode

    # Apply chat template and tokenize prompts in batches
    prompts_chat_formatted = [tokenizer.apply_chat_template(item['chat_prompt'], tokenize=False, add_generation_prompt=True)
                              for item in dataset]

    print(f"Generating responses for {len(dataset)} prompts (batch size {batch_size})...")
    for i in tqdm(range(0, len(prompts_chat_formatted), batch_size)):
        batch_prompts = prompts_chat_formatted[i:i+batch_size]
        # Tokenize the batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)

        # Generate responses
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id, # Use EOS for padding during generation
            do_sample=False, # Greedy decoding for math problems
            # temperature=0.1, # Optional: for sampling
            # top_p=0.9,       # Optional: for sampling
        )

        # Decode generated sequences (excluding the prompt part)
        input_token_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_token_len:]
        decoded_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # Store results for this batch
        original_data_batch = dataset[i:i+batch_size]
        for idx, response in enumerate(decoded_responses):
            original_item = original_data_batch[idx]
            results.append({
                "prompt": original_item['prompt'],
                "chat_prompt_text": batch_prompts[idx], # Store the actual text fed to model
                "ground_truth": original_item['answer'],
                "generated_response": response,
            })
        # Optional: Clear cache periodically if memory is tight
        # if i % 10 == 0: torch.cuda.empty_cache()

    return results

# --- Evaluation Function ---
def evaluate_results(results):
    correct_count = 0
    evaluated_data = []
    print("Evaluating generated responses...")
    for item in tqdm(results):
        extracted = extract_xml_answer(item['generated_response'])
        ground_truth = item['ground_truth']

        # Normalize both for comparison (remove spaces, handle potential float vs int)
        # Basic normalization: remove whitespace
        extracted_norm = extracted.replace(" ", "") if extracted else "N/A"
        gt_norm = ground_truth.replace(" ", "") if ground_truth else "N/A"

        # Attempt numeric comparison if possible
        is_correct = False
        try:
            if extracted_norm != "N/A" and gt_norm != "N/A":
                # Compare as floats to handle decimals correctly
                if abs(float(extracted_norm) - float(gt_norm)) < 1e-4: # Tolerance for float comparison
                    is_correct = True
                else:
                     # Sometimes answers might be simple fractions, try evaluating them if needed
                     # Example: eval('1/2') == 0.5 -> requires careful implementation
                     pass # Simple comparison failed
            # If one is N/A or comparison failed/not numeric, fallback to string comparison
            elif extracted_norm == gt_norm:
                 is_correct = True # e.g., if both are "N/A" or some specific string format matches

        except ValueError:
            # If conversion to float fails, fallback to string comparison
             if extracted_norm == gt_norm:
                 is_correct = True

        if is_correct:
            correct_count += 1

        evaluated_data.append({
            **item, # Keep original prompt, gt, generation
            "extracted_answer": extracted,
            "is_correct": is_correct
        })

    accuracy = (correct_count / len(results)) * 100 if results else 0
    return accuracy, evaluated_data

# --- Main Execution ---
if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch._dynamo.config.suppress_errors = True

    # 1. Load Model
    model, tokenizer = load_trained_model(BASE_MODEL_NAME, TRAINED_MODEL_PATH)

    # --- 2. Evaluate GSM8K ---
    print("\n" + "="*30)
    print(" Evaluating GSM8K Test Set ")
    print("="*30)
    gsm8k_test_dataset = get_gsm8k_questions(split="test")
    if not gsm8k_test_dataset:
        print("No valid GSM8K test data loaded. Skipping.")
    else:
        gsm8k_results_raw = generate_responses(model, tokenizer, gsm8k_test_dataset)
        gsm8k_accuracy, gsm8k_results_evaluated = evaluate_results(gsm8k_results_raw)
        print(f"\nGSM8K Test Accuracy: {gsm8k_accuracy:.2f}%")

        # Save detailed GSM8K results
        gsm8k_output_file = os.path.join(OUTPUT_RESULTS_DIR, "gsm8k_test_results.json")
        with open(gsm8k_output_file, "w") as f:
            json.dump(gsm8k_results_evaluated, f, indent=2)
        print(f"Detailed GSM8K results saved to: {gsm8k_output_file}")

    torch.cuda.empty_cache() # Clear cache before next dataset

    # --- 3. Evaluate MATH-500 ---
    print("\n" + "="*30)
    print(" Evaluating MATH-500 Test Set ")
    print("="*30)
    math500_test_dataset = get_math500_questions(split="test")
    if not math500_test_dataset:
        print("No valid MATH-500 test data loaded. Skipping.")
    else:
        math500_results_raw = generate_responses(model, tokenizer, math500_test_dataset)
        math500_accuracy, math500_results_evaluated = evaluate_results(math500_results_raw)
        print(f"\nMATH-500 Test Accuracy: {math500_accuracy:.2f}%")

        # Save detailed MATH-500 results
        math500_output_file = os.path.join(OUTPUT_RESULTS_DIR, "math500_test_results.json")
        with open(math500_output_file, "w") as f:
            json.dump(math500_results_evaluated, f, indent=2)
        print(f"Detailed MATH-500 results saved to: {math500_output_file}")

    print("\nTesting complete.")