import torch
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
from datasets import load_dataset

accelerator = Accelerator()

modelpath = "TinyLlama/TinyLlama_v1.1"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    device_map={"": accelerator.process_index},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16,
)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token,
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
model.resize_token_embeddings(len(tokenizer))
model.config.eos_token_id = tokenizer.eos_token_id

# Add adapters to model
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["lm_head", "embed_tokens"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
model.config.use_cache = False

# Load Argilla dataset

bs = 1
ga_steps = 1
epochs = 5
steps_per_epoch = len(dataset_remote) // (accelerator.state.num_processes * bs * ga_steps)

trainer = SFTTrainer(
    model,
    train_dataset=dataset_remote,
    dataset_text_field="text",
    max_seq_length=512,
)

model.config.use_cache = False
trainer.train()
