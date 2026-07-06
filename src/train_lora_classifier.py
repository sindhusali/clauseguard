"""
ClauseGuard — Step 2
LoRA fine-tune a RoBERTa clause classifier.
Run in Google Colab with a T4 GPU.
"""

import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "roberta-base"
CSV_PATH = "../data/clauseguard_classification_dataset.csv"
OUTPUT_DIR = "../models/clauseguard-lora-classifier"
MAX_LENGTH = 256


def main():
    df = pd.read_csv(CSV_PATH).dropna()
    print(f"Loaded {len(df)} rows, {df['label'].nunique()} classes.")

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    num_labels = len(label_encoder.classes_)

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label_id"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["query", "value"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, learning_rate=2e-4,
        per_device_train_batch_size=16, per_device_eval_batch_size=16,
        num_train_epochs=6, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1_macro",
        logging_steps=25, report_to="none", fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=data_collator, compute_metrics=compute_metrics,
    )

    trainer.train()
    final_metrics = trainer.evaluate()
    print("\nFinal validation metrics (quote these on your resume):")
    print(json.dumps(final_metrics, indent=2))

    model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    with open(f"{OUTPUT_DIR}/final_adapter/label_mapping.json", "w") as f:
        json.dump(dict(enumerate(label_encoder.classes_)), f, indent=2)

    print(f"\nSaved LoRA adapter to {OUTPUT_DIR}/final_adapter/")


if __name__ == "__main__":
    main()