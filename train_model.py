"""Train a lightweight Multinomial Naive Bayes fake-news classifier.

Uses only Python standard library and local Excel datasets.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DATASET_PATHS = [Path("aippp.xlsx"), Path("bharatfakenewskosh.xlsx")]
OUTPUT_MODEL_PATH = Path("model/news_nb_model.json")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
    "this", "these", "those", "or", "not", "you", "your", "i", "we", "they", "them", "their", "our",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.findall(".//a:t", NS)) for si in root.findall("a:si", NS)]


def get_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value_node = cell.find("a:v", NS)
    if value_node is None:
        return ""
    raw = value_node.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(raw)]
    return raw


def parse_dataset(path: Path) -> list[tuple[str, str, int]]:
    with zipfile.ZipFile(path) as zf:
        shared = load_shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = sheet.findall(".//a:sheetData/a:row", NS)
    if not rows:
        return []

    header_map: dict[str, str] = {}
    for cell in rows[0].findall("a:c", NS):
        col = "".join(ch for ch in cell.attrib.get("r", "") if ch.isalpha())
        header_map[col] = get_cell_value(cell, shared)

    col_by_name = {name: col for col, name in header_map.items()}
    id_col = col_by_name.get("id", "A")
    label_col = col_by_name.get("Label", "S")
    candidate_text_cols = [
        col_by_name.get("Eng_Trans_Statement", ""),
        col_by_name.get("Eng_Trans_News_Body", ""),
        col_by_name.get("Statement", ""),
        col_by_name.get("News Body", ""),
    ]
    candidate_text_cols = [c for c in candidate_text_cols if c]

    samples: list[tuple[str, str, int]] = []
    for row in rows[1:]:
        values: dict[str, str] = {}
        for cell in row.findall("a:c", NS):
            col = "".join(ch for ch in cell.attrib.get("r", "") if ch.isalpha())
            values[col] = get_cell_value(cell, shared)

        label_str = values.get(label_col, "")
        if label_str not in {"0", "1"}:
            continue

        text_parts = [values.get(col, "").strip() for col in candidate_text_cols if values.get(col, "").strip()]
        if not text_parts:
            continue

        sample_id = values.get(id_col, "").strip() or hashlib.md5(" ".join(text_parts).encode("utf-8")).hexdigest()[:12]
        samples.append((f"{path.stem}:{sample_id}", " \n ".join(text_parts), int(label_str)))

    return samples


def load_all_samples(paths: list[Path]) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    seen = set()
    for path in paths:
        if not path.exists():
            continue
        for item in parse_dataset(path):
            if item[0] in seen:
                continue
            seen.add(item[0])
            out.append(item)
    return out


def normalize_text(text: str) -> str:
    text = URL_RE.sub(" ", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(normalize_text(text)) if tok not in STOPWORDS and len(tok) > 2]


def split_samples(samples: list[tuple[str, str, int]]) -> tuple[list[tuple[str, list[str], int]], list[tuple[str, list[str], int]]]:
    train, valid = [], []
    for sample_id, text, label in samples:
        tokens = tokenize(text)
        if not tokens:
            continue
        bucket = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest(), 16) % 10
        entry = (sample_id, tokens, label)
        if bucket < 8:
            train.append(entry)
        else:
            valid.append(entry)
    return train, valid


def build_vocab(train_samples: Iterable[tuple[str, list[str], int]], max_features: int) -> set[str]:
    doc_freq: Counter[str] = Counter()
    for _, tokens, _ in train_samples:
        doc_freq.update(set(tokens))
    return {tok for tok, _ in doc_freq.most_common(max_features)}


def fit_nb(train_samples: list[tuple[str, list[str], int]], vocab: set[str], alpha: float):
    class_doc_counts = Counter()
    token_counts = {0: Counter(), 1: Counter()}
    total_tokens = Counter()

    for _, tokens, label in train_samples:
        class_doc_counts[label] += 1
        for tok in tokens:
            if tok in vocab:
                token_counts[label][tok] += 1
                total_tokens[label] += 1

    vocab_size = len(vocab)
    total_docs = class_doc_counts[0] + class_doc_counts[1]
    log_prior = {
        0: math.log((class_doc_counts[0] + alpha) / (total_docs + 2 * alpha)),
        1: math.log((class_doc_counts[1] + alpha) / (total_docs + 2 * alpha)),
    }

    denom0 = total_tokens[0] + alpha * vocab_size
    denom1 = total_tokens[1] + alpha * vocab_size
    unk_log_prob = {
        0: math.log(alpha / denom0),
        1: math.log(alpha / denom1),
    }

    token_log_prob = {}
    for tok in vocab:
        token_log_prob[tok] = {
            0: math.log((token_counts[0][tok] + alpha) / denom0),
            1: math.log((token_counts[1][tok] + alpha) / denom1),
        }

    return {
        "alpha": alpha,
        "vocab_size": vocab_size,
        "log_prior": log_prior,
        "unk_log_prob": unk_log_prob,
        "token_log_prob": token_log_prob,
    }


def predict(model, tokens: list[str]) -> int:
    score0 = model["log_prior"][0]
    score1 = model["log_prior"][1]

    for tok in tokens:
        probs = model["token_log_prob"].get(tok)
        if probs is None:
            score0 += model["unk_log_prob"][0]
            score1 += model["unk_log_prob"][1]
        else:
            score0 += probs[0]
            score1 += probs[1]

    return 1 if score1 >= score0 else 0


def evaluate(model, samples: list[tuple[str, list[str], int]]) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for _, tokens, label in samples:
        pred = predict(model, tokens)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "samples": total,
    }


def serialize_model(model, tuning, metrics, datasets):
    return {
        "model_type": "multinomial_naive_bayes",
        "version": "nb-v2-multidataset",
        "tuning": tuning,
        "validation": metrics,
        "datasets": datasets,
        "stopwords": sorted(STOPWORDS),
        "token_pattern": TOKEN_RE.pattern,
        "log_prior": {str(k): v for k, v in model["log_prior"].items()},
        "unk_log_prob": {str(k): v for k, v in model["unk_log_prob"].items()},
        "token_log_prob": {tok: {"0": probs[0], "1": probs[1]} for tok, probs in model["token_log_prob"].items()},
    }


def main() -> None:
    samples = load_all_samples(DATASET_PATHS)
    if not samples:
        raise SystemExit("No samples found in dataset files.")

    train_samples, valid_samples = split_samples(samples)
    if not train_samples or not valid_samples:
        raise SystemExit("Dataset split failed; need both train and validation samples.")

    print(f"Loaded samples: total={len(samples)}, train={len(train_samples)}, valid={len(valid_samples)}")

    best = None
    configs = [
        {"max_features": 12000, "alpha": 0.2},
        {"max_features": 20000, "alpha": 0.3},
        {"max_features": 30000, "alpha": 0.4},
        {"max_features": 40000, "alpha": 0.5},
    ]

    for cfg in configs:
        vocab = build_vocab(train_samples, cfg["max_features"])
        model = fit_nb(train_samples, vocab, cfg["alpha"])
        metrics = evaluate(model, valid_samples)
        print(f"cfg={cfg} -> acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        candidate = (metrics["f1"], metrics["accuracy"], cfg, model, metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    _, _, best_cfg, best_model, best_metrics = best
    print(f"Best config: {best_cfg} with f1={best_metrics['f1']:.4f}")

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    datasets_used = [str(p) for p in DATASET_PATHS if p.exists()]
    payload = serialize_model(best_model, best_cfg, best_metrics, datasets_used)
    OUTPUT_MODEL_PATH.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Saved tuned model to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
