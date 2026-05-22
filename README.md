**DPO for YouTube Title Generation (Qwen3-14B)**

This project fine-tunes the Qwen3-14B large language model using
**Direct Preference Optimization (DPO)** to generate high‑engagement
YouTube titles. DPO enables direct alignment from preference data
****without a separate reward model or reinforcement learning (RL) loop****.

**Project Overview**
Goal: Train a model to generate attractive, click‑worthy YouTube titles
by learning from human preference pairs (chosen vs. rejected titles).

Key features:
- DPO: single‑stage training from preference data
- ****LoRA efficient fine‑tuning****: only ~0.86% of parameters trained
- ****Unsloth acceleration**** for faster training and lower VRAM usage
- 4‑bit quantization to fit on consumer GPUs
- Full comparison: base model vs. DPO‑fine‑tuned model


**Project Objectives**
1. Understand the DPO framework and its advantages over RLHF
2. Implement DPO fine‑tuning using the Hugging Face TRL library
3. Process preference datasets in prompt/chosen/rejected format
4. Use LoRA for parameter‑efficient fine‑tuning
5. Evaluate model outputs qualitatively and quantitatively
6. Use modern optimization libraries (Unsloth) for speed

**System Requirements**
****Core libraries:****
transformers==4.57.6
trl==0.24.0
torch>=2.4.0
datasets
peft
accelerate
bitsandbytes
xformers
unsloth>=2026.4.6
llm-blender
mergekit
pandas

****Install commands:****
pip install datasets torch bitsandbytes xformers
pip install unsloth trl accelerate --upgrade
pip install llm-blender
pip install transformers==4.57.6
pip install trl mergekit

****Hardware:****
CUDA‑compatible GPU with at least 16GB VRAM (e.g., Tesla T4, A10)
Supports 4‑bit quantization


**Dataset**
Dataset name: EliasHossain/youtube-titles-dpo
Structure: prompt / chosen (preferred title) / rejected (worse title)
Size: 923 training samples, 103 validation samples

Example:
prompt: Given the YouTube video idea write an engaging title...
chosen: Power Law Scaling: The Key to YouTube Success
rejected: From Zero to Hero: Scaling YouTube with Power Laws


**Implementation Steps**
****1. Model Loading (Unsloth + 4‑bit)****
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-14B-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
tokenizer.pad_token = tokenizer.eos_token

****2. LoRA Configuration****
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
Trainable params: 128,450,560
Total params:     14,896,757,760
Trainable %:      0.8623%

****3. DPO Training Setup****
from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="./dpo_youtube_checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=5e-6,
    beta=0.1,
    fp16=True,
    load_best_model_at_end=True,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    tokenizer=tokenizer,
)
trainer.train()


**Training Results**
Train loss:        0.626 → 0.485
Eval loss:         0.616 → 0.558
Reward margin:     0.248 → 0.537 (****significant improvement****)
Preference accuracy: up to 85%

Summary:
- Stable training, no overfitting
- Reward margin doubled → clear preference learning
- 3 epochs give good baseline; ****5+ epochs recommended****


****Improvements from DPO:****
- More specific, audience‑aware titles
- Stronger action verbs and engagement hooks
- Style closer to high‑performance YouTube titles

****Limitations:****
- Some outputs unchanged from base
- Dataset noise reduces consistency
- Character‑length compliance can be improved


**Usage**
1. Set up environment and install dependencies
2. Log in to Hugging Face for model access
3. Run the DPO fine‑tuning pipeline
4. Generate titles using the generate_dpo() function
5. Compare base and DPO outputs manually


**Future Improvements**
1. Clean and standardize the preference dataset
2. Extend training to 5+ epochs
3. Tune beta, learning rate, and batch size
4. Enforce strict 30–75 character length
5. Deploy for batch inference or API service

**License**
Open source. Model and dataset follow their respective licenses.
