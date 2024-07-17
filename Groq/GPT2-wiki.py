import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
try:
    from groqflow import groqit
except:
    raise ImportError("GroqFlow module not found!")

# Load the dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Convert dataset to PyTorch Dataset
class WikiTextDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.dataset = tokenized_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': self.dataset[idx]['input_ids'],
            'attention_mask': self.dataset[idx]['attention_mask']
        }

# Create a PyTorch Dataset
pytorch_dataset = WikiTextDataset(tokenized_dataset)

# Create a DataLoader
dataloader = DataLoader(pytorch_dataset, batch_size=1, shuffle=True)

# Initialize the GPT-2 model with the language modeling head
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set the model to evaluation mode
model.eval()

# Create dummy static inputs in a dictionary to pass into GroqFlow
inputs = {
   "input_ids": torch.ones(1, 256, dtype=torch.long),
   "attention_mask": torch.ones(1, 256, dtype=torch.float),
}

# Instantiate a groq model with the model and dummy inputs
groq_model = groqit(model, inputs, rebuild="never")

# Iterate over the DataLoader
for batch in dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Ensure no gradient computations
    with torch.no_grad():
        # Pass the batch through the model and generate output sequences
        cpu_outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256) # CPU Example
        groq_outputs = groq_model(**{
            "input_ids": input_ids, 
            "attention_mask": attention_mask
        }) # Running on Groq
        
    # Decode the generated sequences to text
    cpu_decoded_outputs = [tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in cpu_outputs]
    groq_decoded_outputs = [tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in cpu_outputs]
    
    # Print the generated sentences
    for i, sentence in enumerate(cpu_decoded_outputs):
        print(f"CPU generated sentence {i+1}:\n{sentence}\n")
        
    # Print the generated sentences
    for i, sentence in enumerate(groq_decoded_outputs):
        print(f"Groq generated sentence {i+1}:\n{sentence}\n")

    break  # Just to show the output for one batch
