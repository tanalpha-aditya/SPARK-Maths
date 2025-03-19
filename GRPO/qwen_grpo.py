import re
import os
import json
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
import torch
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
import wandb  # Import Weights & Biases
from tqdm import tqdm  # Progress bar


# ------------------------------ Configuration ------------------------------
wandb.login(key="3109e45ecb4ed9dad85e22af19852af76198d140")
WANDB_PROJECT = "QwenMATH-GRPO"  # Replace with your W&B project name
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
OUTPUT_DIR = "outputs"

MAX_SEQ_LENGTH = 1024
LORA_RANK = 32
MAX_PROMPT_LENGTH = 256  # Adjusted to keep completion length within reasonable bounds
NUM_GENERATIONS = 6  # Adjusted to keep completion length within reasonable bounds
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 500  # Adjusted to a more realistic number
SAVE_STEPS = 500
LEARNING_RATE = 5e-6

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# ------------------------------ Data Preparation ------------------------------


XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


# ------------------------------ Reward Functions ------------------------------


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in matches]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


# ------------------------------ Model Initialization ------------------------------


def initialize_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer


# ------------------------------ Training Function ------------------------------


def train(model, tokenizer, train_dataset, reward_functions):
    training_args = GRPOConfig(
        learning_rate=LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH,
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        max_grad_norm=0.3,  # Increased max_grad_norm for better stabilization
        report_to="wandb",  # Enable Weights & Biases reporting
        output_dir=OUTPUT_DIR,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    return trainer


# ------------------------------ Testing Function ------------------------------


def extract_answer(text: str) -> str | None:
    """
    Extracts the final numerical answer from a response.
    It tries XML extraction first, then falls back to a regex-based approach.
    """
    # Try XML extraction
    xml_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if xml_match:
        return xml_match.group(1).strip()

    # If XML extraction fails, try regex for a number at the end
    regex_match = re.search(r"(\d+)$", text)
    if regex_match:
        return regex_match.group(1)  # Return the captured number

    return None  # No answer found


def test(model, tokenizer, test_dataset: Dataset, results_file="test_results.jsonl", batch_size=4):
    """
    Evaluates the model's performance, saves results to a JSON Lines file,
    and logs metrics to Weights & Biases.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.95, max_tokens=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH
    )

    with torch.no_grad(), open(results_file, "w", encoding="utf-8") as outfile:
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Testing"):
            batch = test_dataset[i : i + batch_size]
            prompts = batch["prompt"]
            answers = batch["answer"]

            # Tokenize prompts and generate text in batches
            tokenized_prompts = [
                tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                for prompt in prompts
            ]

            # Generate text using the model for the batch of prompts
            outputs = model.fast_generate(
                tokenized_prompts, sampling_params=sampling_params, lora_request=None
            )

            predictions = [output.outputs[0].text for output in outputs]

            for prompt, pred, ans in zip(prompts, predictions, answers):
                extracted_prediction = extract_answer(pred)

                # Determine if the prediction is correct
                is_correct = extracted_prediction == ans

                # Log the result to wandb
                wandb.log({
                    f"example_{total_samples}/prompt": prompt[-1]["content"],
                    f"example_{total_samples}/prediction": pred,
                    f"example_{total_samples}/extracted_prediction": extracted_prediction,
                    f"example_{total_samples}/answer": ans,
                    f"example_{total_samples}/is_correct": is_correct
                })

                if is_correct:
                    total_correct += 1

                total_samples += 1

                # Save the results to the JSON Lines file
                result = {
                    "prompt": prompt[-1]["content"],  # Extract user prompt
                    "answer": ans,
                    "response": pred,
                    "extracted_answer": extracted_prediction,
                    "correct": is_correct,
                }
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

    accuracy = total_correct / total_samples
    print(f"Accuracy on test dataset: {accuracy:.4f}")

    wandb.log({"test/accuracy": accuracy})
    model.train()
    print(f"Test results saved to {results_file}")

# ------------------------------ Main Function ------------------------------


def main(test_dataset_path=None):
    """
    Main function to orchestrate the loading of the model, preparing the training
    and testing datasets, setting up the training process, and evaluating the model.
    """
    # Initialize Weights & Biases (wandb) for experiment tracking
    wandb.init(project=WANDB_PROJECT, job_type="training")

    # Log the essential configurations at the start
    wandb.config.update(
        {
            "model_name": MODEL_NAME,
            "max_seq_length": MAX_SEQ_LENGTH,
            "lora_rank": LORA_RANK,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "system_prompt": SYSTEM_PROMPT,
        }
    )

    # Initialize the model and tokenizer
    model, tokenizer = initialize_model()

    # Load and prepare the training dataset
    train_dataset = get_gsm8k_questions()
    test_dataset = get_gsm8k_questions(split="test")

    # Load the testing dataset if provided
    if test_dataset_path:
        test_dataset = load_dataset("json", data_files=test_dataset_path)["train"]
    else:
        print("No test dataset path provided. Skipping testing.")
        test_dataset = None

    # Define the reward functions to guide the training
    reward_functions = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]
    # Train the model
    trainer = train(model, tokenizer, train_dataset, reward_functions)

    # Save only the main model after training
    model_output_path = os.path.join(OUTPUT_DIR, "QWEN-Math-GRPO")
    model.save_pretrained(model_output_path, push_to_hub=False)
    tokenizer.save_pretrained(model_output_path, push_to_hub=False)
    print(f"Main model saved to {model_output_path}")

    # Save only the main model and log its path to wandb for tracking
    artifact = wandb.Artifact("trained-model", type="model")
    artifact.add_dir(model_output_path)  # Add the saved model directory to the artifact
    wandb.log_artifact(artifact)  # Log the artifact
    print(f"Main model and tokenizer saved to {model_output_path} and uploaded to wandb")

    
    # Perform testing if a test dataset is available
    if test_dataset:
        test(model, tokenizer, test_dataset, results_file="test_results.jsonl")

    wandb.finish()

# ------------------------------ Run the Main Function ------------------------------


if __name__ == "__main__":
    # Example usage of the main function, you can pass the path to your test dataset
    # Ensure the test_dataset_path is correctly pointing to your test dataset
    # or leave it as None if you don't have a test dataset available.
    test_dataset_path = "../Dataset/GSM8k/modified/main_test.csv"  # Add your test dataset path here
    main(test_dataset_path=test_dataset_path)