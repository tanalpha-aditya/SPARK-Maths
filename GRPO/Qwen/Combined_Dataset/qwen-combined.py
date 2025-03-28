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
from typing import List, Dict, Union

# ------------------------------ Configuration ------------------------------
wandb.login(key="3109e45ecb4ed9dad85e22af19852af76198d140")  # Replace with *your* W&B API key
WANDB_PROJECT = "QwenMATH-GRPO-Combined"  # Changed project name
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
OUTPUT_DIR = "outputs"

MAX_SEQ_LENGTH = 786
LORA_RANK = 16
MAX_PROMPT_LENGTH = 256
NUM_GENERATIONS = 6  # Generations per prompt during training
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
MAX_STEPS = 6000 # Increased steps, as combined dataset is larger
SAVE_STEPS = 500
LEARNING_RATE = 5e-6

# No system prompt is needed, since Qwen2.5-Math already has its own instruction format.
# We'll rely on the dataset's existing structure.

# ------------------------------ Data Preparation ------------------------------


def extract_boxed_answer(text: str) -> str:
    """Extracts the answer from the Qwen2.5-Math \\boxed{} format."""
    match = re.search(r"\\boxed\{(.+?)\}", text)  # Allow any characters inside the box, non-greedy
    if match:
        extracted = match.group(1)
        return extracted #Removed standardize_number
    return "N/A"  # Return "N/A" if not found


def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            # We *only* include the question as the prompt.  No system prompt.
            "prompt": [{"role": "user", "content": x["question"]}],
            "answer": extract_hash_answer(x["answer"]),  # Use the helper function
        }
    )
    return data

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    answer_text = text.split("####")[1].strip()
    return answer_text #Removed standardize_number

def get_math500_questions(split="test") -> Dataset:
    data = load_dataset("HuggingFaceH4/MATH-500")[split]
    print(data[0])
    data = data.map(
        lambda x: {
            "prompt": [{"role": "user", "content": x['problem']}], #problem is used as user input here
            "answer": x['answer'], #answer used as is no processing
        }
    )
    return data

def combine_datasets(dataset1, dataset2) -> Dataset:
    """Combines two datasets, handling potential key mismatches."""
    # Get the dictionaries for both datasets
    dict1 = dataset1.to_dict()
    dict2 = dataset2.to_dict()

    # Find the common keys
    common_keys = list(set(dict1.keys()) & set(dict2.keys()))

    # Create a new dictionary to store the combined data, using only common keys
    combined_data = {key: dict1[key] + dict2[key] for key in common_keys}

    # Convert the combined dictionary to a Dataset object and shuffle
    combined_dataset = Dataset.from_dict(combined_data)
    return combined_dataset.shuffle(seed=42)


# ------------------------------ Reward Functions ------------------------------

def correctness_reward_func(prompts: List[List[Dict[str, str]]], completions: List[List[Dict[str, str]]], answer: List[str], **kwargs) -> List[float]:
    rewards = []
    for i in range(len(prompts)):
        responses = [completion["content"] for completion in completions[i]]
        extracted_responses = [extract_boxed_answer(r) for r in responses]
        correct_answers = [answer[i]] * len(responses)

        print("-" * 20)
        print(f"Question:\n{prompts[i][-1]['content']}")  # User's question
        print(f"Answer:\n{answer[i]}")

        for j, (resp, extracted, corr_ans) in enumerate(zip(responses, extracted_responses, correct_answers)):
             print(f"Response {j+1}:\n{resp}")
             print(f"Extracted {j+1}:\n{extracted}")

        for extracted, corr_ans in zip(extracted_responses, correct_answers):
            if extracted == "N/A" or corr_ans == "N/A": #Handle "N/A"
              rewards.append(0.0)
            elif extracted == corr_ans:
                rewards.append(2.0)
            else:
                rewards.append(0.0)
    return rewards

def count_tokens_hardcoded(text: str) -> int:
    """
    Estimates the number of tokens in a text using a simple word-splitting
    and punctuation-based approach.  This is an approximation and will
    NOT be perfectly accurate, especially for languages other than English
    or text with unusual formatting.
    """
    # Split by spaces (most common tokenization)
    words = text.split()
    count = len(words)

    # Add 1 for each punctuation mark (very rough approximation)
    for char in text:
        if char in ['.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}', '-', '/', '\\']:
            count += 1

    # Add 1 for latex tag opening.
    count += text.count(r"\\(")
    count += text.count(r"\\)")
    count += text.count(r"\\boxed")
    count += text.count(r"\{")
    count += text.count(r"\}")

    return count

def int_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    rewards = []
    for completion_group in completions:
        batch_rewards = []
        for completion in completion_group:
            response = completion["content"]
            extracted_response = extract_boxed_answer(response)  # Use boxed extraction
            batch_rewards.append(0.5 if extracted_response.isdigit() else 0.0) #Keep this function because some part of our dataset has this
        rewards.extend(batch_rewards)
    return rewards



def length_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """Rewards longer completions (token count), normalized."""
    rewards = []
    for completion_group in completions:
        batch_rewards = []
        for completion in completion_group:
            response = completion["content"]
            # Tokenize the completion *without* adding special tokens.
            length = count_tokens_hardcoded(response)
            # Normalize to a range [0, 0.5].  Avoids overwhelming other rewards.
            # The 0.5 scaling factor is a hyperparameter you might tune.
            normalized_length = min(length / MAX_SEQ_LENGTH, 1.0)  # Cap at 1.0
            reward = 0.5 * normalized_length
            batch_rewards.append(reward)
        rewards.extend(batch_rewards)
    return rewards



def reasoning_length_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
    """Rewards longer reasoning sections *before* the boxed answer."""
    rewards = []
    for completion_group in completions:
        batch_rewards = []
        for completion in completion_group:
            response = completion["content"]
            # Find the start of the boxed answer.
            boxed_match = re.search(r"\\boxed\{", response)
            if boxed_match:
                reasoning_part = response[:boxed_match.start()]
            else:
                # If no boxed answer, consider the whole response as reasoning
                # (or give a small/zero reward - design choice).  Here we reward the full length.
                reasoning_part = response

            length = count_tokens_hardcoded(reasoning_part)
            # Normalize, as with the overall length reward.
            normalized_length = min(length / (MAX_SEQ_LENGTH * 0.8), 1.0) # Cap at 1, use 80% of max_seq as reasonable max.
            reward = 0.5 * normalized_length #Hyperparameter
            batch_rewards.append(reward)
        rewards.extend(batch_rewards)
    return rewards


# ------------------------------ Model Initialization ------------------------------

def initialize_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=0.2,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[  # Double-check these are correct for Qwen2.5-Math
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
        max_grad_norm=0.3,
        report_to="wandb",
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,  # Keep the 'answer' column
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_functions,
        # reward_kwargs={"tokenizer" : tokenizer},
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    return trainer


# ------------------------------ Testing Function ------------------------------

def test(model, tokenizer, test_dataset: Dataset, results_file="test_results.jsonl", batch_size=4):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_valid = 0

    sampling_params = SamplingParams(
        temperature=0.7, top_p=0.95, max_tokens=MAX_SEQ_LENGTH - MAX_PROMPT_LENGTH
    )

    with torch.no_grad(), open(results_file, "w", encoding="utf-8") as outfile:
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Testing"):
            batch = test_dataset[i : i + batch_size]
            prompts = batch["prompt"]
            answers = batch["answer"]

            tokenized_prompts = [
                tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                for prompt in prompts
            ]
            outputs = model.fast_generate(
                tokenized_prompts, sampling_params=sampling_params, lora_request=None
            )
            predictions = [output.outputs[0].text for output in outputs]


            for prompt, pred, ans in zip(prompts, predictions, answers):
                extracted_prediction = extract_boxed_answer(pred)  # Use boxed extraction
                is_correct = extracted_prediction == ans

                if extracted_prediction != "N/A": #Check N/A as we dont want those stats
                    total_valid += 1
                    if is_correct:
                        total_correct += 1

                total_samples += 1
                wandb.log({
                    f"example_{total_samples}/prompt": prompt[-1]["content"],
                    f"example_{total_samples}/prediction": pred,
                    f"example_{total_samples}/extracted_prediction": extracted_prediction,
                    f"example_{total_samples}/answer": ans,
                    f"example_{total_samples}/is_correct": is_correct
                })

                result = {
                    "prompt": prompt[-1]["content"],
                    "answer": ans,
                    "response": pred,
                    "extracted_answer": extracted_prediction,
                    "correct": is_correct,
                }
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

    accuracy = (total_correct / total_valid) if total_valid > 0 else 0.0
    print(f"Accuracy on valid samples: {accuracy:.4f}")
    wandb.log({"test/accuracy": accuracy})
    model.train()
    print(f"Test results saved to {results_file}")


# ------------------------------ Main Function ------------------------------

def main(test_dataset_path=None):
    torch.cuda.empty_cache()
    wandb.init(project=WANDB_PROJECT, job_type="training")

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
            # No system prompt
        }
    )

    model, tokenizer = initialize_model()
    # --- Load and combine datasets ---
    train_gsm8k = get_gsm8k_questions()
    train_math500 = get_math500_questions()
    train_dataset = combine_datasets(train_gsm8k, train_math500)


    test_gsm8k = get_gsm8k_questions(split="test")
    test_math500 = get_math500_questions(split="test")
    test_dataset = combine_datasets(test_gsm8k, test_math500)
    # --- End dataset loading ---

    if test_dataset_path:
        test_dataset = load_dataset("json", data_files=test_dataset_path)["train"]

    # Only include reward functions relevant to Qwen2.5-Math's output format
    reward_functions = [
        int_reward_func,         # Encourage integer answers
        correctness_reward_func,  # Reward correct answers
        length_reward_func,       # Reward longer completions
        reasoning_length_reward_func,  # Reward longer reasoning
    ]

    trainer = train(model, tokenizer, train_dataset, reward_functions)

    model_output_path = os.path.join(OUTPUT_DIR, "QWEN-Math-GRPO-Combined")  # Changed output path
    model.save_pretrained(model_output_path, push_to_hub=False)
    tokenizer.save_pretrained(model_output_path, push_to_hub=False)

    artifact = wandb.Artifact("trained-model", type="model")
    artifact.add_dir(model_output_path)
    wandb.log_artifact(artifact)
    print(f"Model/tokenizer saved to {model_output_path}, uploaded to wandb.")

    if test_dataset:
        test(model, tokenizer, test_dataset)

    wandb.finish()


if __name__ == "__main__":
    main()