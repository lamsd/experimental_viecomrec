#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PhoBERT SimCSE end-to-end:
- train  : unsupervised SimCSE fine-tuning for Vietnamese item texts
- encode : embed item texts (title + description)
- eval   : offline metrics (Recall@K / MRR@K / nDCG@K) on Validation/Test
- compare: compare base vs finetuned checkpoints side-by-side

Data files expected:
- items_prepared.csv  (columns: item_id, title, description)
- interactions_train.csv (columns: user_id, item_id, timestamp)
- interactions_val.csv   (columns: user_id, val_item, timestamp)
- interactions_test.csv  (columns: user_id, test_item, timestamp)
"""

import os
import math
import json
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def detect_device(prefer: str = None) -> torch.device:
    if prefer is not None:
        if prefer.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def l2_normalize(x: torch.Tensor, dim=1, eps=1e-9) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, L, H], attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)     # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)          # [B, 1]
    return summed / counts


# -----------------------------
# Dataset for unsupervised training
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.texts[idx]


def join_item_texts(items_csv: str, min_len: int = 5) -> List[str]:
    df = pd.read_csv(items_csv)
    for col in ["title", "description"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {items_csv}")
    s = (df["title"].fillna("").astype(str) + ". " + df["description"].fillna("").astype(str)).str.strip()
    texts = [t for t in s.tolist() if len(t) >= min_len]
    # dedup văn bản để training hiệu quả hơn
    texts = list(dict.fromkeys(texts))
    return texts


# -----------------------------
# PhoBERT encoder (with pooling)
# -----------------------------
class PhoBERTEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, normalize: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_path)
        self.normalize = normalize

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        emb = mean_pool(out.last_hidden_state, attention_mask)
        if self.normalize:
            emb = l2_normalize(emb)
        return emb


# -----------------------------
# SimCSE Loss (unsupervised, in-batch)
# -----------------------------
class SimCSELoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # z1, z2: [B, D], assumed L2-normalized
        bsz = z1.size(0)
        z = torch.cat([z1, z2], dim=0)         # [2B, D]
        sim = torch.matmul(z, z.t()) / self.temperature
        mask = torch.eye(2 * bsz, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)
        targets = torch.arange(bsz, 2 * bsz, device=z.device)
        targets = torch.cat([targets, torch.arange(0, bsz, device=z.device)], dim=0)  # [2B]
        loss = self.ce(sim, targets)
        return loss


def collate_texts(batch_texts: List[str], tokenizer, max_len: int, device: torch.device):
    enc = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    )
    return {k: v.to(device) for k, v in enc.items()}


# -----------------------------
# TRAIN
# -----------------------------
def cmd_train(
    items_csv: str,
    output_dir: str = "phobert-simcse",
    base_model: str = "vinai/phobert-base",
    device_str: str = None,
    batch_size: int = 64,
    max_len: int = 256,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_steps: int = 5000,
    grad_accum_steps: int = 1,
    temperature: float = 0.05,
    seed: int = 42,
    num_workers: int = 2,
    log_every: int = 50,
):
    seed_everything(seed)
    device = detect_device(device_str)
    print(f"[train] device={device}")

    texts = join_item_texts(items_csv, min_len=5)
    print(f"[train] loaded {len(texts)} unique texts.")

    if len(texts) < batch_size:
        raise ValueError(f"Not enough texts ({len(texts)}) for batch_size={batch_size}")

    dataset = TextDataset(texts)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = PhoBERTEncoder(base_model, normalize=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup_steps = int(max_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )
    loss_fn = SimCSELoss(temperature=temperature)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    os.makedirs(output_dir, exist_ok=True)
    step, running = 0, 0.0

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            b1 = collate_texts(batch, tokenizer, max_len, device)
            b2 = collate_texts(batch, tokenizer, max_len, device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                z1 = model(**b1)
                z2 = model(**b2)
                loss = loss_fn(z1, z2) / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if ((step + 1) % grad_accum_steps) == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * grad_accum_steps
            step += 1
            if step % log_every == 0:
                print(f"[train] step {step} | loss {running/log_every:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
                running = 0.0

    # save backbone + tokenizer
    model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump({
            "max_len": max_len,
            "temperature": temperature,
            "batch_size": batch_size,
            "max_steps": max_steps
        }, f, ensure_ascii=False, indent=2)
    print(f"[train] saved to {output_dir}")


# -----------------------------
# ENCODE (items → embeddings)
# -----------------------------
def _encode_texts(texts: List[str], model_dir: str, device: torch.device, max_len: int = 256, batch_size: int = 128) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    backbone = AutoModel.from_pretrained(model_dir).to(device)
    backbone.eval()

    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            out = backbone(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            emb = l2_normalize(emb)
            vecs.append(emb.cpu().float())
    return torch.cat(vecs, dim=0).numpy().astype(np.float32)

def cmd_encode(items_csv: str, model_dir: str, out_dir: str = "embeds", device_str: str = None, max_len: int = 256, batch_size: int = 128):
    device = detect_device(device_str)
    print(f"[encode] device={device}")
    df = pd.read_csv(items_csv)
    for col in ["item_id", "title", "description"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {items_csv}")
    texts = (df["title"].fillna("").astype(str) + ". " + df["description"].fillna("").astype(str)).tolist()
    item_ids = df["item_id"].tolist()

    os.makedirs(out_dir, exist_ok=True)
    embs = _encode_texts(texts, model_dir=model_dir, device=device, max_len=max_len, batch_size=batch_size)
    np.save(os.path.join(out_dir, "item_embeds.npy"), embs)
    # map item_id -> row index
    with open(os.path.join(out_dir, "item_index.json"), "w", encoding="utf-8") as f:
        json.dump({"item_ids": item_ids, "model_dir": model_dir, "max_len": max_len}, f, ensure_ascii=False, indent=2)
    print(f"[encode] saved {embs.shape} to {out_dir}/item_embeds.npy and item_index.json")


# -----------------------------
# EVAL (Validation / Test)
# -----------------------------
def _load_interactions(split_csv: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(split_csv)
    # normalize columns
    if split.lower().startswith("val"):
        assert {"user_id", "val_item"}.issubset(df.columns), "Validation CSV must have columns: user_id, val_item, timestamp"
        df = df.rename(columns={"val_item": "item_id"})
    elif split.lower().startswith("test"):
        assert {"user_id", "test_item"}.issubset(df.columns), "Test CSV must have columns: user_id, test_item, timestamp"
        df = df.rename(columns={"test_item": "item_id"})
    else:  # train
        assert {"user_id", "item_id"}.issubset(df.columns), "Train CSV must have columns: user_id, item_id, timestamp"
    return df[["user_id", "item_id"]].dropna()

def _build_user_profiles(train_df: pd.DataFrame, id2row: Dict[str, int], item_embs: np.ndarray) -> Dict[str, np.ndarray]:
    # profile = mean of user's train item embeddings
    user_prof = {}
    g = train_df.groupby("user_id")
    for u, sub in g:
        idxs = [id2row[i] for i in sub["item_id"].tolist() if i in id2row]
        if len(idxs) == 0:
            continue
        v = item_embs[idxs].mean(axis=0)
        # l2 norm (already normalized items, but mean may need renorm)
        v = v / (np.linalg.norm(v) + 1e-9)
        user_prof[u] = v.astype(np.float32)
    return user_prof

def _eval_split(
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    item_ids: List[str],
    item_embs: np.ndarray,
    ks: List[int] = [10, 20, 50],
    ensure_target_in_candidates: bool = True,
    exclude_seen_train: bool = True,
) -> Dict[str, float]:
    # mapping
    id2row = {it: i for i, it in enumerate(item_ids)}
    # candidate set: items seen in Train (to avoid cold-start item)
    train_items = set(train_df["item_id"].unique())
    cand_mask = np.array([it in train_items for it in item_ids], dtype=bool)
    cand_rows = np.where(cand_mask)[0]
    if len(cand_rows) == 0:
        raise ValueError("No candidate items (train_items empty).")

    # user -> seen train items
    seen_by_user = {u: set(v["item_id"].tolist()) for u, v in train_df.groupby("user_id")}
    # user profiles
    user_prof = _build_user_profiles(train_df, id2row, item_embs)

    M = item_embs.shape[0]
    # normalize item_embs already, but ensure float32
    E = item_embs.astype(np.float32)

    ranks = []
    dropped_no_profile = 0
    dropped_target_not_in_index = 0
    dropped_target_not_in_cand = 0

    for _, row in split_df.iterrows():
        u = row["user_id"]
        tgt = row["item_id"]
        if u not in user_prof:
            dropped_no_profile += 1
            continue
        if tgt not in id2row:
            dropped_target_not_in_index += 1
            continue

        prof = user_prof[u]  # (D,)
        seen = seen_by_user.get(u, set())

        # build candidate row ids
        cands = cand_rows
        if exclude_seen_train:
            # exclude user's seen items in train from candidates
            bad = [id2row[i] for i in seen if i in id2row]
            if bad:
                mask = np.ones(len(cands), dtype=bool)
                # remove bad indices
                bad_set = set(bad)
                cands = np.array([r for r in cands if r not in bad_set], dtype=int)

        # ensure target present
        tgt_row = id2row[tgt]
        if ensure_target_in_candidates and tgt_row not in cands:
            # add it back
            cands = np.concatenate([cands, np.array([tgt_row], dtype=int)], axis=0)

        # compute similarity: cosine = dot (both L2-normed)
        sims = E[cands].dot(prof)  # (Nc,)
        # higher is better
        order = np.argsort(-sims)  # descending
        ranked_rows = cands[order]
        # find rank of target
        try:
            pos = np.where(ranked_rows == tgt_row)[0][0]  # 0-based rank
        except IndexError:
            dropped_target_not_in_cand += 1
            continue

        ranks.append(int(pos))  # 0-based rank

    if len(ranks) == 0:
        raise ValueError("No evaluable events (all dropped).")

    # metrics
    def recall_at_k(k): return float(np.mean([1.0 if r < k else 0.0 for r in ranks]))
    def mrr_at_k(k):    # truncated MRR@k
        vals = []
        for r in ranks:
            if r < k:
                vals.append(1.0 / (r + 1))
            else:
                vals.append(0.0)
        return float(np.mean(vals))
    def ndcg_at_k(k):
        vals = []
        for r in ranks:
            if r < k:
                dcg = 1.0 / math.log2(r + 2)  # rank r => position r+1
            else:
                dcg = 0.0
            # ideal DCG (hit at rank 1) = 1/log2(1+1) = 1
            vals.append(dcg)
        return float(np.mean(vals))

    out = {
        "evaluated_events": len(ranks),
        "dropped_no_profile": dropped_no_profile,
        "dropped_target_not_in_index": dropped_target_not_in_index,
        "dropped_target_not_in_candidates": dropped_target_not_in_cand,
    }
    for k in ks:
        out[f"Recall@{k}"] = recall_at_k(k)
        out[f"MRR@{k}"]    = mrr_at_k(k)
        out[f"nDCG@{k}"]   = ndcg_at_k(k)
    return out


def cmd_eval(
    items_csv: str,
    embeds_dir: str,
    train_csv: str,
    split_csv: str,
    split: str = "Validation",  # "Validation" or "Test"
    ks: List[int] = [10, 20, 50],
    out_json: str = "metrics.json",
):
    # load embeds
    embeds_path = os.path.join(embeds_dir, "item_embeds.npy")
    index_path  = os.path.join(embeds_dir, "item_index.json")
    if not (os.path.exists(embeds_path) and os.path.exists(index_path)):
        raise FileNotFoundError(f"Missing embeds at {embeds_path} or {index_path}. Run 'encode' first.")

    E = np.load(embeds_path)  # (N, D)
    idx = json.load(open(index_path, "r", encoding="utf-8"))
    item_ids = idx["item_ids"]

    # sanity
    df_items = pd.read_csv(items_csv)
    if len(df_items) != len(item_ids):
        print("[warn] items_prepared.csv length != item_index length. Proceeding with index order.")

    train_df = _load_interactions(train_csv, "Train")
    split_df = _load_interactions(split_csv, split)

    metrics = _eval_split(split_df, train_df, item_ids, E, ks=ks)
    json.dump(metrics, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[eval:{split}] -> {out_json}")
    for k in ks:
        print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f} | MRR@{k}: {metrics[f'MRR@{k}']:.4f} | nDCG@{k}: {metrics[f'nDCG@{k}']:.4f}")
    print("  counts:", {k: v for k, v in metrics.items() if k.startswith("dropped") or k == "evaluated_events"})


# -----------------------------
# COMPARE (base vs finetuned)
# -----------------------------
def _encode_with_model(items_csv: str, model_dir: str, device: torch.device, tmp_dir: str, max_len: int = 256, batch_size: int = 128) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(items_csv)
    texts = (df["title"].fillna("").astype(str) + ". " + df["description"].fillna("").astype(str)).tolist()
    item_ids = df["item_id"].tolist()
    embs = _encode_texts(texts, model_dir=model_dir, device=device, max_len=max_len, batch_size=batch_size)
    os.makedirs(tmp_dir, exist_ok=True)
    np.save(os.path.join(tmp_dir, "item_embeds.npy"), embs)
    with open(os.path.join(tmp_dir, "item_index.json"), "w", encoding="utf-8") as f:
        json.dump({"item_ids": item_ids}, f)
    return item_ids, embs

def cmd_compare(
    items_csv: str,
    train_csv: str,
    split_csv: str,
    split: str,
    base_model: str = "vinai/phobert-base",
    tuned_ckpt: str = "phobert-simcse",
    device_str: str = None,
    ks: List[int] = [10, 20, 50],
):
    device = detect_device(device_str)
    # base
    item_ids_a, E_a = _encode_with_model(items_csv, base_model, device, tmp_dir=".tmp_base_embeds")
    # tuned
    item_ids_b, E_b = _encode_with_model(items_csv, tuned_ckpt, device, tmp_dir=".tmp_tuned_embeds")

    assert item_ids_a == item_ids_b, "Item order mismatch between encodings."
    train_df = _load_interactions(train_csv, "Train")
    split_df = _load_interactions(split_csv, split)

    m_a = _eval_split(split_df, train_df, item_ids_a, E_a, ks=ks)
    m_b = _eval_split(split_df, train_df, item_ids_b, E_b, ks=ks)

    print(f"[compare:{split}] BASE={base_model}  vs  TUNED={tuned_ckpt}")
    hdr = "K | Recall | MRR | nDCG"
    print("BASE".ljust(6), hdr)
    for k in ks:
        print("BASE".ljust(6), f"{k:>2} | {m_a[f'Recall@{k}']:.4f} | {m_a[f'MRR@{k}']:.4f} | {m_a[f'nDCG@{k}']:.4f}")
    print("TUNED".ljust(6), hdr)
    for k in ks:
        print("TUNED".ljust(6), f"{k:>2} | {m_b[f'Recall@{k}']:.4f} | {m_b[f'MRR@{k}']:.4f} | {m_b[f'nDCG@{k}']:.4f}")

    both = {"base": m_a, "tuned": m_b, "ks": ks, "base_model": base_model, "tuned_ckpt": tuned_ckpt}
    with open("compare_metrics.json", "w", encoding="utf-8") as f:
        json.dump(both, f, ensure_ascii=False, indent=2)
    print("[compare] saved compare_metrics.json")


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="PhoBERT SimCSE training + embedding + evaluation")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train SimCSE (unsupervised) on item texts")
    pt.add_argument("--items-csv", required=True, help="items_prepared.csv")
    pt.add_argument("--output-dir", default="phobert-simcse")
    pt.add_argument("--base-model", default="vinai/phobert-base")
    pt.add_argument("--device", default=None)
    pt.add_argument("--batch-size", type=int, default=64)
    pt.add_argument("--max-len", type=int, default=256)
    pt.add_argument("--lr", type=float, default=5e-5)
    pt.add_argument("--weight-decay", type=float, default=0.01)
    pt.add_argument("--warmup-ratio", type=float, default=0.06)
    pt.add_argument("--max-steps", type=int, default=5000)
    pt.add_argument("--grad-accum-steps", type=int, default=1)
    pt.add_argument("--temperature", type=float, default=0.05)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--num-workers", type=int, default=2)
    pt.add_argument("--log-every", type=int, default=50)

    # encode
    pe = sub.add_parser("encode", help="Encode items into embeddings")
    pe.add_argument("--items-csv", required=True)
    pe.add_argument("--model-dir", required=True, help="checkpoint dir (e.g., phobert-simcse) or model name")
    pe.add_argument("--out-dir", default="embeds")
    pe.add_argument("--device", default=None)
    pe.add_argument("--max-len", type=int, default=256)
    pe.add_argument("--batch-size", type=int, default=128)

    # eval
    pv = sub.add_parser("eval", help="Evaluate on Validation/Test")
    pv.add_argument("--items-csv", required=True)
    pv.add_argument("--embeds-dir", default="embeds")
    pv.add_argument("--train-csv", required=True)
    pv.add_argument("--split-csv", required=True)
    pv.add_argument("--split", choices=["Validation", "Test"], default="Validation")
    pv.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50])
    pv.add_argument("--out-json", default="metrics.json")

    # compare
    pc = sub.add_parser("compare", help="Compare base vs tuned")
    pc.add_argument("--items-csv", required=True)
    pc.add_argument("--train-csv", required=True)
    pc.add_argument("--split-csv", required=True)
    pc.add_argument("--split", choices=["Validation", "Test"], default="Validation")
    pc.add_argument("--base-model", default="vinai/phobert-base")
    pc.add_argument("--tuned-ckpt", default="phobert-simcse")
    pc.add_argument("--device", default=None)
    pc.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50])

    args = p.parse_args()

    if args.cmd == "train":
        cmd_train(
            items_csv=args.items_csv,
            output_dir=args.output_dir,
            base_model=args.base_model,
            device_str=args.device,
            batch_size=args.batch_size,
            max_len=args.max_len,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            max_steps=args.max_steps,
            grad_accum_steps=args.grad_accum_steps,
            temperature=args.temperature,
            seed=args.seed,
            num_workers=args.num_workers,
            log_every=args.log_every,
        )
    elif args.cmd == "encode":
        cmd_encode(
            items_csv=args.items_csv,
            model_dir=args.model_dir,
            out_dir=args.out_dir,
            device_str=args.device,
            max_len=args.max_len,
            batch_size=args.batch_size,
        )
    elif args.cmd == "eval":
        cmd_eval(
            items_csv=args.items_csv,
            embeds_dir=args.embeds_dir,
            train_csv=args.train_csv,
            split_csv=args.split_csv,
            split=args.split,
            ks=args.ks,
            out_json=args.out_json,
        )
    elif args.cmd == "compare":
        cmd_compare(
            items_csv=args.items_csv,
            train_csv=args.train_csv,
            split_csv=args.split_csv,
            split=args.split,
            base_model=args.base_model,
            tuned_ckpt=args.tuned_ckpt,
            device_str=args.device,
            ks=args.ks,
        )


if __name__ == "__main__":
    main()