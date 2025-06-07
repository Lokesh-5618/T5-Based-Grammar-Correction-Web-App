from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Load model
model_path = "final_t5_model"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/", methods=["GET", "POST"])
def correct():
    corrected_text = ""
    if request.method == "POST":
        sentence = request.form["input_text"]
        input_text = "fix: " + sentence
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
        output = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
        corrected_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return render_template("index.html", corrected_text=corrected_text)

if __name__ == "__main__":
    app.run(debug=True)
