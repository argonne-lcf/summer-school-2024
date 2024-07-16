import torch
import torch.nn as nn
import transformers
try:
    from groqflow import groqit
except:
    raise ImportError("GroqFlow module not found!")

model = transformers.GPT2Model(transformers.GPT2Config())
inputs = {
   "input_ids": torch.ones(1, 256, dtype=torch.long),
   "attention_mask": torch.ones(1, 256, dtype=torch.float),
}

gmodel = groqit(model, inputs, rebuild="never")
groq_output = gmodel(**inputs)

print(groq_output)

