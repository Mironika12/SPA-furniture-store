from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import load_dataset
import torch

# BIO-теги
label_list = ["O", "B-PRODUCT", "I-PRODUCT"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# Загрузка датасета
train_dataset = load_dataset("json", data_files="D:\proga\\test3\data\\train.jsonl")["train"]
val_dataset = load_dataset("json", data_files="D:\proga\\test3\data\\val.jsonl")["train"]

# Токенизатор
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# BIO-разметка на основе token offset mapping
def encode_entities_to_bio(text, entities, offset_mapping):
    labels = ["O"] * len(offset_mapping)
    for ent in entities:
        ent_start, ent_end, ent_label = ent["start"], ent["end"], ent["label"]
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start >= ent_end or tok_end <= ent_start:
                continue
            if tok_start >= ent_start and tok_end <= ent_end:
                if labels[i] == "O":
                    labels[i] = "B-" + ent_label if tok_start == ent_start else "I-" + ent_label
    return [label2id[label] for label in labels]

# Токенизация + выравнивание
def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_offsets_mapping=True
    )
    labels = encode_entities_to_bio(example["text"], example["entities"], tokenized["offset_mapping"])
    final_labels = []
    for offset, label in zip(tokenized["offset_mapping"], labels):
        if offset[0] == offset[1]:
            final_labels.append(-100)
        else:
            final_labels.append(label)
    tokenized["labels"] = final_labels
    tokenized.pop("offset_mapping")
    return tokenized

# Применяем токенизацию
tokenized_train = train_dataset.map(tokenize_and_align_labels)
tokenized_val = val_dataset.map(tokenize_and_align_labels)

# Модель
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_list),  # 3 класса
    id2label=id2label,
    label2id=label2id,
)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=10,
    save_steps=100,
    learning_rate=1.2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    max_grad_norm=2.5,
    warmup_ratio=0.03,
)

# Обучение
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)],
)

trainer.train()

# Сохраняем модель
model.save_pretrained("./ner_model_simple")
tokenizer.save_pretrained("./ner_model_simple")
print("Обучение завершено. Модель сохранена в 'ner_model_simple'")