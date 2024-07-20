from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc



args = dict(
  model_name_or_path="deepseek-ai/deepseek-coder-1.3b-base", # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path="/cos_mount/users/dibyanayan/deepseek_lora_ept", 
  template=None,# load the saved LoRA adapters                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
)
chat_model = ChatModel(args)

messages = []
print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

query = {"text": "\n\n def foo():   a = bar()\n   return a"}

messages = []
messages.append(query)

print('response')
for new_text in chat_model.stream_chat(messages):
    print(new_text, end="", flush=True)

torch_gc()
