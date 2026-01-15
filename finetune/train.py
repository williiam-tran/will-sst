import os
import sys
import json
import torch
import random
import time
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    default_data_collator,
    TrainerCallback
)
from peft import get_peft_model

# Thêm thư mục gốc vào path để import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Optimize CUDA memory allocator
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'

# Enable TF32 for faster matmul on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels

from vieneu_utils.phonemize_text import phonemize_with_dict
from finetune.configs.lora_config import lora_config, training_config, get_training_args

# Performance monitoring callback
class PerformanceCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n⚡ Training started at {time.strftime('%H:%M:%S')}")
        print(f"⚡ Target steps: {args.max_steps}")
        print(f"⚡ Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
            remaining_steps = args.max_steps - state.global_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                gpu_util = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_memory * 100
                print(f"📊 Step {state.global_step}/{args.max_steps} | "
                      f"{steps_per_sec:.2f} steps/s | "
                      f"GPU: {gpu_mem:.1f}GB ({gpu_util:.0f}%) | "
                      f"ETA: {eta_minutes:.1f}min")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        avg_speed = state.global_step / total_time
        print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
        print(f"✅ Average speed: {avg_speed:.2f} steps/sec")

def preprocess_sample(sample, tokenizer, max_len=2048):
    speech_gen_start = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
    ignore_index = -100
    
    phones = sample["phones"]
    vq_codes = sample["codes"]
    
    codes_str = "".join([f"<|speech_{i}|>" for i in vq_codes])
    chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"""
    
    ids = tokenizer.encode(chat)
    
    # Pad nếu ngắn
    if len(ids) < max_len:
        ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
    elif len(ids) > max_len:
        ids = ids[:max_len]
    
    input_ids = torch.tensor(ids, dtype=torch.long)
    labels = torch.full_like(input_ids, ignore_index)
    
    speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
    if len(speech_gen_start_idx) > 0:
        speech_gen_start_idx = speech_gen_start_idx[0]
        labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]
    
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

class VieNeuDataset(Dataset):
    def __init__(self, metadata_path, tokenizer, max_len=2048):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if not os.path.exists(metadata_path):
             raise FileNotFoundError(f"Missing dataset file: {metadata_path}")
             
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    # filename|text|codes
                    self.samples.append({
                        "filename": parts[0],
                        "text": parts[1],
                        "codes": json.loads(parts[2])
                    })
        print(f"🦜 Đã tải {len(self.samples)} mẫu dữ liệu từ {metadata_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        
        try:
            phones = phonemize_with_dict(text)
        except Exception as e:
            print(f"⚠️ Lỗi khi xử lý text: {e}")
            phones = text 
            
        data_item = {
            "phones": phones,
            "codes": sample["codes"]
        }
        
        return preprocess_sample(data_item, self.tokenizer, self.max_len)

def run_training():
    model_name = training_config['model']
    print(f"🦜 Đang tải model gốc: {model_name}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    # Enable gradient checkpointing BEFORE applying LoRA (if using gradient checkpointing)
    if training_config.get('gradient_checkpointing', False) or hasattr(get_training_args(training_config), 'gradient_checkpointing') and get_training_args(training_config).gradient_checkpointing:
        print("🔧 Enabling gradient checkpointing for LoRA compatibility...")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Load Dataset
    dataset_path = os.path.join("finetune", "dataset", "metadata_encoded.csv")
    if not os.path.exists(dataset_path):
        print(f"⚠️ Không tìm thấy {dataset_path}. Vui lòng chạy prepare data trước.")
        return

    full_dataset = VieNeuDataset(dataset_path, tokenizer)

    print(f"🦜 Total samples: {len(full_dataset)} (eval disabled, training only)")

    # Apply LoRA
    print("🦜 Đang áp dụng LoRA adapters...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Trainer Setup
    args = get_training_args(training_config)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=full_dataset,
        eval_dataset=None,
        data_collator=default_data_collator,
        callbacks=[PerformanceCallback()],
    )
    
    print("🦜 Bắt đầu quá trình huấn luyện! (Chúc may mắn)")
    trainer.train()
    
    # Save Final Model
    save_path = os.path.join(training_config['output_dir'], training_config['run_name'])
    print(f"🦜 Đang lưu model LoRA tại: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    run_training()
