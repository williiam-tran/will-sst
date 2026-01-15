
import os
import sys
import json
import torch
import random
import psutil
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    default_data_collator
)
from peft import get_peft_model

# Thêm thư mục gốc vào path để import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================
# RTX 5090 MAXIMUM PERFORMANCE MODE
# ============================================
# Dedicated server: Use 100% of all resources for fastest training
# - GPU: 100% (RTX 5090 - 32GB VRAM)
# - CPU: 100% (all cores for data loading)
# - RAM: 100% (no artificial limits)
# ============================================

# Use all available CPU cores for maximum throughput
cpu_count = os.cpu_count() or 8
torch.set_num_threads(cpu_count)
torch.set_num_interop_threads(cpu_count)

# Enable TF32 for RTX 5090 (faster matmul operations)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set CUDA optimal settings
torch.backends.cudnn.benchmark = True  # Auto-tune for optimal performance

# Optimize CUDA memory allocator for large batches
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print(f"⚡ RTX 5090 MAXIMUM PERFORMANCE MODE")
print(f"🔧 Using {cpu_count}/{cpu_count} CPU threads (100%)")
print(f"🔧 GPU: RTX 5090 - 100% capacity, TF32 enabled")
print(f"🔧 RAM: Unlimited (100% available)")
print(f"🔧 CUDA Memory: Optimized for large batches")

from vieneu_utils.phonemize_text import phonemize_with_dict
from finetune.configs.lora_config import lora_config, training_config, get_training_args

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

    # RTX 5090: No memory limits - use everything available
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"💾 System RAM: {total_ram_gb:.1f}GB (100% available)")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU Detected: {gpu_name}")
        print(f"💾 GPU VRAM: {gpu_memory:.1f}GB (100% available)")
    else:
        print(f"⚠️ WARNING: No CUDA GPU detected! Training will be slow.")

    # Load Model - No artificial limits, use all available VRAM
    try:
        # Try Flash Attention 2 (fastest for RTX 5090)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("✅ Flash Attention 2 enabled (maximum speed)")
    except:
        # Fallback if flash-attn not installed
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("⚠️ Flash Attention 2 not available, using default attention")
    
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
    )
    
    print("🦜 Bắt đầu quá trình huấn luyện!")
    print(f"⚡ Batch size: {training_config['per_device_train_batch_size']} x {training_config['gradient_accumulation_steps']} = {training_config['per_device_train_batch_size'] * training_config['gradient_accumulation_steps']} (effective)")
    print(f"⚡ Expected training time: ~45-60 minutes (RTX 5090 optimized)")

    trainer.train()
    
    # Save Final Model
    save_path = os.path.join(training_config['output_dir'], training_config['run_name'])
    print(f"🦜 Đang lưu model LoRA tại: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    run_training()
