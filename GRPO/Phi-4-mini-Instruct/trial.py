from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from vllm import LLM, SamplingParams
import torch
import re

# Load the Phi-4-mini-instruct model and tokenizer
MODEL_NAME = "microsoft/Phi-4-mini-instruct"  # Replace with the actual model name or path
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Adapted extract_answer function
def extract_answer(text: str) -> str:
    """Extracts the answer from the Phi-4-mini-instruct's formatted output."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)  # Extracts content inside <answer>...</answer>
    if match:
        return match.group(1).strip()
    return "N/A"  # Return "N/A" if no <answer> tag is found

# Example test questions (use questions relevant to your task)
test_questions = [
    "What is the sum of 3 and 5?",
    "How do you solve a quadratic equation?",
    "What is the derivative of x^2?",
    "Henry and 3 of his friends order 7 pizzas for lunch. Each pizza is cut into 8 slices. If Henry and his friends want to share the pizzas equally, how many slices can each of them have?"
]
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


# Generate answers for each question and print the output
for question in test_questions:
    # Define the system message format
    system_message = """
    <|system|> 
    Respond in the following format and complete the full answer always:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    <|end|>
    """

    
    # llm = LLM(model="microsoft/Phi-4-mini-instruct", trust_remote_code=True)

    # messages = [
    #     {"role": "system", "content": "You are a helpful MATHS AI assistant."},
    #     {"role": "user", "content": question},
    # ]

    # sampling_params = SamplingParams(
    # max_tokens=500,
    # temperature=0.0,
    # )

    # output = llm.chat(messages=messages, sampling_params=sampling_params)
    # print(output[0].outputs[0].text)

    # # Encode the question
    # inputs = tokenizer(chat_format, return_tensors="pt")

    # # Generate the model's response
    # with torch.no_grad():
    #     outputs = model.generate(inputs["input_ids"], max_token=1000, temperature=0.0)

    # # Decode the model's response
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # # Extract the answer using the new extract_answer function
    # answer = extract_answer(response)

    # Print the question and the extracted answer

  

    
    
    messages = [
        {"role": "system", "content": """You are a helpful MATHS AI assistant. Respond in the following format and complete the full answer always:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ONLY INTEGER ANSWER
    </answer>
    <|end|>"""},
        {"role": "user", "content": question},
    ]
    
    
    
    generation_args = {
        "max_new_tokens": 1000,
        "return_full_text": False,
        "temperature": 0.8,
        "do_sample": False,
    }
    
    output = pipe(messages, **generation_args)
    print(output[0]['generated_text'])


    # print(f"Question: {question}")
    # print(f"Model Response: {output}")
    # print(f"Extracted Answer: {output[0].outputs[0].text}\n")

