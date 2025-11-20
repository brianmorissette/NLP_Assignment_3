"""
Knowledge Distillation on SQuAD

Goal:
Compare standard fine-tuning vs. knowledge distillation (KD) using a BERT teacher and student model.
Fill in missing parts (marked with TODO).

Suggested environment:
pip install transformers datasets evaluate torch accelerate
"""

import os
import math
import json
import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    set_seed,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
from torch.nn.utils.rnn import pad_sequence


# --------------------------
# Global Config
# --------------------------
class Config:
    teacher_model = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
    student_model = "bert-base-uncased"
    output_dir = "./checkpoints_kd_squad"

    max_length = 384
    doc_stride = 128
    batch_size = 8
    epochs = 2
    lr = 3e-5
    alpha_kd = 0.5
    beta_ce = 0.5
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
set_seed(cfg.seed)


# --------------------------
# Utility Functions
# --------------------------
def prepare_features(examples, tokenizer):
    """
    Prepare SQuAD features for both train and validation.
    """
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=cfg.max_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []
    example_ids = []   

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_mapping[i]
        example_ids.append(examples["id"][sample_idx])   

        answers = examples["answers"][sample_idx]
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        sequence_ids = tokenized.sequence_ids(i)

        # find context start/end
        context_start = None
        context_end = None
        for idx, s in enumerate(sequence_ids):
            if s == 1 and context_start is None:
                context_start = idx
            if s == 1:
                context_end = idx

        if not (offsets[context_start][0] <= start_char <= offsets[context_end][1]):
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_start = context_start
        while token_start <= context_end and offsets[token_start][0] <= start_char:
            token_start += 1
        token_start -= 1

        token_end = context_end
        while token_end >= context_start and offsets[token_end][1] >= end_char:
            token_end -= 1
        token_end += 1

        start_positions.append(token_start)
        end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized["example_id"] = example_ids   

    return tokenized


def make_collate_fn(tokenizer):
    pad_id = tokenizer.pad_token_id

    def collate_fn(batch):
        # Example ID stays list[str]
        example_ids = [b["example_id"] for b in batch]

        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]

        # Labels
        start_positions = torch.tensor([b["start_positions"] for b in batch])
        end_positions = torch.tensor([b["end_positions"] for b in batch])

        # Now pad
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_positions,
            "end_positions": end_positions,
            "example_id": example_ids,   # remains list[str]
        }

    return collate_fn




# --------------------------
# Losses
# --------------------------
def kl_loss(student_logits, teacher_logits, T=2.0):
    """KL divergence loss for distillation."""
    s = F.log_softmax(student_logits / T, dim=-1)
    t = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (T ** 2)


# --------------------------
# Training Functions
# --------------------------
def train_standard(student, dataloader, optimizer, scheduler, tokenizer):
    """Standard fine-tuning (no distillation)."""
    student.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Fine-tuning"):
        # Move only tensor fields to device
        batch = {
            k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        model_inputs = {k: v for k, v in batch.items() if k != "example_id"}
        outputs = student(**model_inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_kd(student, teacher, dataloader, optimizer, scheduler, tokenizer):
    """Knowledge Distillation (CE + KL).  -- FILL THIS FUNCTION --
    Implement KD training with:
      - teacher in eval + no grad
      - move only tensors to device
      - drop non-tensor keys (e.g., example_id) from model inputs
      - CE (hard) + KL (soft) losses
      - optimizer/scheduler steps
    """
    # TODO (1 pt): correct train/eval modes for teacher and

    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Distillation"):
        # TODO (2 pts): move only tensor fields to device; keep others (e.g., example_id) as-is

        # TODO (1 pt): drop fields not accepted by forward (e.g., example_id)

        # TODO (2 pts): teacher forward with no grad
 

        # TODO (1 pt): student forward

        # TODO (3 pts): losses = CE + KL(start/end) with provided kl_loss()

        # TODO (1 pt): optimize (backward, step, scheduler, zero_grad) and accumulate

    return total_loss / len(dataloader)


# --------------------------
# Evaluation
# --------------------------
def evaluate_model(model, tokenizer, dataloader, raw_dataset):
    model.eval()
    metric = evaluate.load("squad")
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move only tensor fields to device
            batch = {
                k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                        }
            model_inputs = {k: v for k, v in batch.items() if k != "example_id"}
            outputs = model(**model_inputs)
            start = torch.argmax(outputs.start_logits, dim=-1)
            end = torch.argmax(outputs.end_logits, dim=-1)
            for i in range(len(start)):
                ans = tokenizer.decode(batch["input_ids"][i][start[i]:end[i]+1], skip_special_tokens=True)
                preds.append({"id": batch["example_id"][i], "prediction_text": ans})
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in raw_dataset]
    return metric.compute(predictions=preds, references=refs)


# --------------------------
# Main
# --------------------------
def main():
    print("Loading data...")
    raw = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_model, use_fast=True)
    train_set = raw["train"].select(range(10000)).map(lambda ex: prepare_features(ex, tokenizer),
                                 batched=True, remove_columns=raw["train"].column_names)
    val_set = raw["validation"].select(range(50)).map(lambda ex: prepare_features(ex, tokenizer),
                                    batched=True, remove_columns=raw["validation"].column_names)
    collate_fn = make_collate_fn(tokenizer)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, collate_fn=collate_fn)

    print("Loading models...")
    teacher = AutoModelForQuestionAnswering.from_pretrained(cfg.teacher_model).to(cfg.device)
    student = AutoModelForQuestionAnswering.from_pretrained(cfg.student_model).to(cfg.device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, cfg.epochs * len(train_loader))

    print("\n=== Standard Fine-tuning ===")
    loss_ft = train_standard(student, train_loader, optimizer, scheduler, tokenizer)
    print(f"Fine-tune loss: {loss_ft:.4f}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, cfg.epochs * len(train_loader))

    print("\n=== Knowledge Distillation ===")
    loss_kd = train_kd(student, teacher, train_loader, optimizer, scheduler, tokenizer)
    print(f"Distillation loss: {loss_kd:.4f}")

    print("\nEvaluating fine-tuned student...")
    scores_ft = evaluate_model(student, tokenizer, val_loader, raw["validation"].select(range(50)))
    print(scores_ft)

    print("\nEvaluating distilled student...")
    scores_kd = evaluate_model(student, tokenizer, val_loader, raw["validation"].select(range(50)))
    print(scores_kd)


if __name__ == "__main__":
    main()
