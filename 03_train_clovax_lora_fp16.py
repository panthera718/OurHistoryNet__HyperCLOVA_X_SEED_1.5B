import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
)

BASE_MODEL_PATH = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
DATA_PATH       = "history_sft_train.jsonl"
OUTPUT_DIR      = "./HyperCLOVAX-history-lora-fp16"
MAX_SEQ_LEN     = 4096

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable} / {total} "
          f"({100 * trainable / total:.2f}%)")

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[INFO] Loading base model in fp16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
    )

    # For memory saving
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    print(f"[INFO] Loading dataset from {DATA_PATH} ...")
    raw_datasets = load_dataset("json", data_files={"train": DATA_PATH})

    def format_example(example):
        instruction = example.get("instruction", "").strip()
        context     = example.get("input", "").strip()
        answer      = example.get("output", "").strip()

        if context:
            prompt = (
                "다음은 국사편찬위원회 우리역사넷(신편 한국사)에 기반한 질의응답이다.\n"
                "주어진 참고 자료를 우선적으로 활용하여, 우리역사넷의 서술에서 벗어나지 않도록 신중하게 답변하라.\n\n"
                f"질문: {instruction}\n\n"
                f"참고 자료: {context}\n\n"
                "답변: "
            )
        else:
            prompt = (
                "다음은 국사편찬위원회 우리역사넷(신편 한국사)에 기반한 질의응답이다.\n"
                "우리역사넷의 일반적인 서술 방식과 내용을 따라, 신중하게 답변하라.\n\n"
                f"질문: {instruction}\n\n"
                "답변: "
            )

        full_text = prompt + answer + tokenizer.eos_token
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("[INFO] Tokenizing dataset ...")
    tokenized_train = raw_datasets["train"].map(
        format_example,
        remove_columns=raw_datasets["train"].column_names,
        batched=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    print("[INFO] Start training (LoRA, fp16, no quantization) ...")
    trainer.train()

    print("[INFO] Saving LoRA adapter and tokenizer ...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[DONE] LoRA fine-tuning finished. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
