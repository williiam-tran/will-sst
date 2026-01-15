"""
Demo VieNeuSDK v1.1.3 - Full Features Guide
"""

import time
import soundfile as sf
from vieneu import Vieneu
from pathlib import Path

def main():
    print("🚀 Initializing VieNeu SDK (v1.1.3)...")
    
    tts = Vieneu(
        backbone_repo="pnnbao-ump/VieNeu-TTS",  # PyTorch version (LoRA compatible)
        backbone_device="mps",  # Use "mps" for Apple Silicon, "cuda" for NVIDIA GPU, or "cpu" for CPU-only
    )

    print("\n--- 1. Available Preset Voices ---")
    available_voices = tts.list_preset_voices()
    print("📋 Voices:", available_voices)

    # Select a preset voice
    current_voice = tts.get_preset_voice("Will")
    print("✅ Selected voice: Binh")


    # ---------------------------------------------------------
    # PART 2: CREATE & SAVE CUSTOM VOICE
    # ---------------------------------------------------------
    print("\n--- 2. Create Custom Voice ---")

    # Replace with your actual .wav file path and its exact transcript (including punctuation)
    sample_audio = Path(__file__).parent / "finetune/dataset/raw_audio/mobai_00001.wav"
    sample_text = "hôm qua thì tớ đang mở bài dở đúng hông? Để tớ mở bài tiếp nhớ! Ờm, lên đến đại học thì có mình tớ ở nơi hoang vu à, tớ không có đứa bạn cấp ba nào luôn, nên là kiểu mình."

    if sample_audio.exists():
        voice_name = "Will"

        print(f"🎙️ Cloning voice from: {sample_audio.name}")

        # 'clone_voice' now supports saving directly with 'name' argument
        custom_voice = tts.clone_voice(
            audio_path=sample_audio,
            text=sample_text,
            name=voice_name  # <-- Automatically saves voice to system
        )

        print(f"✅ Voice created and saved as: '{voice_name}'")

        # Verify functionality
        print("📋 Voice list after adding:", tts.list_preset_voices())

        # Switch to new voice
        current_voice = custom_voice
    else:
        print("⚠️ Sample audio not found. Skipping...")


    # ---------------------------------------------------------
    # PART 2.5: LOAD LORA ADAPTER (Advanced)
    # ---------------------------------------------------------
    print("\n--- 2.5. Load Fine-tuned LoRA Adapter ---")

    # Check if local LoRA exists
    lora_path = Path(__file__).parent / "finetune" / "output" / "VeNeu-TTS-Vast-LoRA"

    if lora_path.exists():
        print(f"🎯 Loading LoRA adapter from: {lora_path}")

        try:
            # Load LoRA adapter (can be local path or HuggingFace repo ID)
            tts.load_lora_adapter(str(lora_path))
            print("✅ LoRA adapter loaded successfully!")
            print("💡 The model now uses your fine-tuned voice characteristics")

            # Optional: Use a reference from your training set for best results
            # For this demo, we'll use the existing custom_voice
            # In production, use audio from your fine-tuning dataset

        except Exception as e:
            print(f"⚠️ Failed to load LoRA: {e}")
            print("💡 Continuing with base model...")
    else:
        print(f"⚠️ LoRA adapter not found at: {lora_path}")
        print("💡 To create your own LoRA adapter:")
        print("   1. Prepare your training data (audio + transcripts)")
        print("   2. Follow the fine-tuning guide: finetune/README.md")
        print("   3. Or use a HuggingFace repo ID: tts.load_lora_adapter('username/repo-name')")
        print("💡 Using base model with preset voice...")


    # ---------------------------------------------------------
    # PART 3: SYNTHESIS WITH ADVANCED PARAMETERS
    # ---------------------------------------------------------
    print("\n--- 3. Speech Synthesis ---")

    # Note: If LoRA adapter was loaded in PART 2.5, synthesis will use the fine-tuned model
    text_input = "Xin chào, tôi là Will. Tôi có thể giúp bạn đọc sách, làm chatbot thời gian thực, hoặc thậm chí clone giọng nói của bạn."
    
    # Generate with specific temperature
    print("🎧 Generating...")
    audio = tts.infer(
        text=text_input,
        voice=current_voice,
        temperature=0.9,  # Adjustable: Lower (0.1) -> Stable, Higher (1.0+) -> Expressive
        top_k=40,
        max_chars=256,  # Ensure proper text chunking
        silence_p=0.2  # Add 0.15s silence between chunks
    )
    sf.write("output.wav", audio, 24000)
    print("💾 Saved: output.wav")

    # ---------------------------------------------------------
    # OPTIONAL: Unload LoRA Adapter
    # ---------------------------------------------------------
    # If you want to return to the base model without restarting:
    # print("\n🔄 Unloading LoRA adapter...")
    # tts.unload_lora_adapter()
    # print("✅ Returned to base model")

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------
    tts.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
