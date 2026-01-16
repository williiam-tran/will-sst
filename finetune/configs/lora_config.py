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

    # OPTIMIZED FOR FULL MODEL ON RTX 5090
    'per_device_train_batch_size': 3,   # Increased from 2 (safe increase)
    'batch_size': 2,
    'gradient_accumulation_steps': 8,   # Adjusted to maintain effective batch = 9

    'learning_rate': 2e-4,
    'max_steps': 5000,
    'logging_steps': 25,                # More frequent logging for monitoring
    'save_steps': 500,
    'eval_steps': 500,

    'warmup_ratio': 0.05,
    'bf16': True,
    'bf16_full_eval': True,             # Full bf16 for eval (faster)
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
        bf16_full_eval=config.get('bf16_full_eval', True),
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_strategy="no",
        save_strategy="steps",
        save_total_limit=2,
        report_to="none",
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        # OPTIMIZED FOR FULL MODEL ON RTX 5090
        dataloader_num_workers=12,           # Increased from 8 (more parallelism)
        dataloader_pin_memory=True,          # Pin memory for faster GPU transfer
        dataloader_prefetch_factor=4,        # Increased from 2 (prefetch more batches)
        dataloader_persistent_workers=True,  # Keep workers alive between epochs
        gradient_checkpointing=True,         # Save memory
        optim="adamw_torch_fused",           # Faster fused optimizer
        max_grad_norm=1.0,                   # Gradient clipping for stability
        logging_first_step=True,             # Log first step for monitoring
        logging_nan_inf_filter=False,        # Show all metrics including NaN/Inf
    )
