import os
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

# Cấu hình LoRA
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ],
    
    lora_dropout=0.05, 
    bias="none",
    task_type=TaskType.CAUSAL_LM, 
)

training_config = {
    'model': "pnnbao-ump/VieNeu-TTS",  # Full model (larger than 0.3B)
    'run_name': "VieNeu-TTS-Vast-LoRA",
    'output_dir': os.path.join("finetune", "output"),

    # ADJUSTED FOR FULL MODEL: Larger model needs smaller batch
    'per_device_train_batch_size': 4,   # Reduced from 16 to avoid OOM
    'gradient_accumulation_steps': 8,   # Increased to maintain effective batch = 32

    'learning_rate': 2e-4,
    'max_steps': 5000,
    'logging_steps': 50,
    'save_steps': 500,
    'eval_steps': 500,

    'warmup_ratio': 0.05,
    'bf16': True,

    'use_4bit': False,
}

def get_training_args(config):
    return TrainingArguments(
        output_dir=os.path.join(config['output_dir'], config['run_name']),
        do_train=True,
        do_eval=False,
        max_steps=config['max_steps'],
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_ratio=config['warmup_ratio'],
        bf16=config['bf16'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",

        # RTX 5090 PERFORMANCE OPTIMIZATIONS (adjusted for full model)
        dataloader_num_workers=8,            # Reduced from 32 to avoid overhead
        dataloader_pin_memory=True,          # Pin memory for faster GPU transfer
        dataloader_prefetch_factor=2,        # Reduced from 4 for memory conservation
        gradient_checkpointing=False,        # Disable for speed
        optim="adamw_torch_fused",           # Faster fused optimizer
        torch_compile=False,                 # Disabled - causes memory issues with large models
        ddp_find_unused_parameters=False,
    )
