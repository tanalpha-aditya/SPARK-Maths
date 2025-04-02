import os
import json
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams
from tqdm import tqdm
import wandb  # Import Weights & Biases
import re

# ------------------------------ Configuration ------------------------------

# WANDB_PROJECT = "QwenMATH-GRPO"  # Replace with your W&B project name
MODEL_PATH = "outputs/checkpoint-2500"  # Adjust if saved elsewhere
TEST_RESULTS_PATH = "test_results2500.jsonl"

MAX_SEQ_LENGTH = 1024
MAX_PROMPT_LENGTH = 512  # Adjusted to fit your training setup

SYSTEM_PROMPT = """
Respond in the following format and complete full answer always:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# ------------------------------ Data Preparation ------------------------------


def get_gsm8k_questions(split="test"):
    """Load the GSM8K dataset for testing."""
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data


def extract_hash_answer(text: str) -> str | None:
    """Extract the answer from the GSM8K dataset format."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def extract_answer(text: str) -> str | None:
    """
    Extracts the final numerical answer from a response.
    - Extracts from LaTeX `\boxed{}` format if present.
    - Converts all decimal values to integers (e.g., `3.14` → `3`, `5.99` → `5`).
    - As a fallback, extracts the last number in the response.
    - Returns the integer answer as a string.
    """
    # Attempt to extract from \boxed{}
    latex_match = re.search(r"\\boxed\{([\d\.]+)\}", text)
    if latex_match:
        extracted = latex_match.group(1)
    else:
        # If LaTeX extraction fails, try regex for a number at the end
        regex_match = re.search(r"(\d+\.?\d*)$", text)
        extracted = regex_match.group(1) if regex_match else None

    if extracted is None:
        return None  # No valid number found

    # Convert to integer (removes decimal part)
    extracted_int = int(float(extracted))  # Handles cases like "3.99" → 3, "42.0" → 42

    return str(extracted_int)  # Return as string for consistency



# ------------------------------ Model Loading ------------------------------


def load_model():
    """Loads the trained model from the specified path."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
    )
    return model, tokenizer


# ------------------------------ Testing Function ------------------------------


def test(model, tokenizer, test_dataset, batch_size=4):
    """Evaluates the model's performance and saves results to a JSON Lines file."""
    model.eval()
    total_correct = 0
    total_samples = 0

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.95, max_tokens=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH
    )

    with torch.no_grad(), open(TEST_RESULTS_PATH, "w", encoding="utf-8") as outfile:
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Testing"):
            batch = test_dataset[i : i + batch_size]
            prompts = batch["prompt"]
            answers = batch["answer"]

            # Tokenize prompts and generate text in batches
            tokenized_prompts = [
                tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                for prompt in prompts
            ]

            # Generate text using the model
            outputs = model.fast_generate(
                tokenized_prompts, sampling_params=sampling_params, lora_request=None
            )

            predictions = [output.outputs[0].text for output in outputs]

            for prompt, pred, ans in zip(prompts, predictions, answers):
                extracted_prediction = extract_answer(pred)
                print(extracted_prediction)
                # Determine if the prediction is correct
                is_correct = extracted_prediction == ans

                # # Log the result to wandb
                # wandb.log({
                #     f"example_{total_samples}/prompt": prompt[-1]["content"],
                #     f"example_{total_samples}/prediction": pred,
                #     f"example_{total_samples}/extracted_prediction": extracted_prediction,
                #     f"example_{total_samples}/answer": ans,
                #     f"example_{total_samples}/is_correct": is_correct
                # })

                if is_correct:
                    total_correct += 1

                total_samples += 1

                # Save the results to the JSON Lines file
                result = {
                    "prompt": prompt[-1]["content"],
                    "answer": ans,
                    "response": pred,
                    "extracted_answer": extracted_prediction,
                    "correct": is_correct,
                }
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

    accuracy = total_correct / total_samples
    print(f"Accuracy on test dataset: {accuracy:.4f}")

    # wandb.log({"test/accuracy": accuracy})
    model.train()
    print(f"Test results saved to {TEST_RESULTS_PATH}")


# ------------------------------ Main Function ------------------------------


def main():
    """Main function to load the model and test it on the GSM8K test dataset."""
    torch.cuda.empty_cache()  # Clear CUDA cache

    # Initialize Weights & Biases for logging
    # wandb.init(project=WANDB_PROJECT, job_type="testing")

    # Load the trained model and tokenizer
    model, tokenizer = load_model()

    # Load test dataset
    test_dataset = get_gsm8k_questions(split="test")

    # Run the test
    test(model, tokenizer, test_dataset)

    # Finish W&B logging
    # wandb.finish()


# ------------------------------ Run the Main Function ------------------------------

if __name__ == "__main__":
    main()
