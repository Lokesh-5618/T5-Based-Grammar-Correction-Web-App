from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Set path to the fine-tuned model directory
model_path = "E:/project/ML/final_t5_model"

# Load the tokenizer and model from the fine-tuned directory
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Move model to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Grammar Correction Model is ready! Type a sentence (or type 'exit' to quit).\n")

# Start the prediction loop
while True:
    sentence = input("Enter a sentence to correct: ").strip()
    
    if sentence.lower() in ["exit", "quit"]:
        print("Exiting grammar correction.")
        break

    if not sentence:
        print("Please enter a non-empty sentence.")
        continue

    # Prefix input for the T5 model
    input_text = "fix: " + sentence
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate the corrected sentence
    outputs = model.generate(
        inputs,
        max_length=128,
        num_beams=4,
        early_stopping=True
    )

    # Decode and display the result
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Corrected:", corrected)
