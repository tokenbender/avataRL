import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


model_name = "gpt2"

# Load the pretrained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    model.resize_token_embeddings(len(tokenizer))


dataset = load_dataset("yahma/alpaca-cleaned", split="train")


small_dataset = dataset.shuffle(seed=42).select(range(10000))

def preprocess_function(examples):
    texts = []
    
    for instr, inp, out in zip(
        examples["instruction"],
        examples.get("input", [""] * len(examples["instruction"])),
        examples["output"]
    ):
        if inp.strip():
            text = f"Instruction: {instr}\nInput: {inp}\nResponse: {out}<|endoftext|>"
        else:
            text = f"Instruction: {instr}\nResponse: {out}<|endoftext|>"
        texts.append(text)
    return tokenizer(texts, truncation=True, max_length=512)

# Tokenize the dataset
tokenized_dataset = small_dataset.map(
    preprocess_function, batched=True, remove_columns=small_dataset.column_names
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


training_args = TrainingArguments(
    output_dir="./gpt2-alpaca-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,  # Enable mixed precision if supported
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


trainer.train()


model.save_pretrained("./gpt2-alpaca-finetuned")
tokenizer.save_pretrained("./gpt2-alpaca-finetuned")
