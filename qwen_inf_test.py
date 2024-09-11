from transformers import AutoModelForCausalLM, AutoTokenizer

file_name = "weights/qwen2-7b-instruct-q8_0.gguf"
model_id = "Qwen/Qwen2-VL-7B-Instruct-AWQ"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")


test_sentence = [
    "In order to load gguf files in transformers, you should specify the gguf_file argument to the from_pretrained methods of both tokenizers and models. Here is how one would load a tokenizer and a model, which can be loaded from the exact same file",
    "Now you have access to the full, unquantized version of the model in the PyTorch ecosystem, where you can combine it with a plethora of other tools."
]

inputs = tokenizer(test_sentence, return_tensors="pt")
output = model(inputs)

print(output)