"""Скачивание и фиксация обучающих корпусов DRREM.

Что делает:
  1. OpenOrca: тянет parquet ``1M-GPT4-Augmented`` (Open-Orca/OpenOrca) и берёт
     детерминированную равномерную выборку 100 000 строк (seed фиксирован),
     а не «первые 100k» — файл сгруппирован по источникам, префикс был бы смещён.
  2. GSM8K: конфиг ``main`` (openai/gsm8k), train и test целиком.
  3. Токенизатор Qwen2.5-1.5B: только файлы токенизатора, без весов.
  4. Считает статистику корпуса под оба токенизатора (байты и Qwen) и пишет
     ``data/manifest.json`` с sha256 всех артефактов — чтобы любая проба
     ссылалась на один и тот же зафиксированный срез.

Запуск (venv проекта CERBER, там уже есть pyarrow/tokenizers/huggingface_hub):
  ~/Coding/Python/CERBER/.venv/bin/python scripts/fetch_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"

OPENORCA_REPO = "Open-Orca/OpenOrca"
OPENORCA_FILE = "1M-GPT4-Augmented.parquet"
GSM8K_REPO = "openai/gsm8k"
GSM8K_FILES = ("main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet")
QWEN_REPO = "Qwen/Qwen2.5-1.5B"
QWEN_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")

SAMPLE_SIZE = 100_000
SEED = 20260918  # дата фиксации среза


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- скачивание


def fetch_openorca(force: bool) -> Path:
    out = DATA / "openorca_100k.parquet"
    if out.exists() and not force:
        log(f"OpenOrca срез уже есть: {out}")
        return out
    log(f"Скачиваю {OPENORCA_REPO}/{OPENORCA_FILE} (~1 ГБ)…")
    src = Path(
        hf_hub_download(OPENORCA_REPO, OPENORCA_FILE, repo_type="dataset", local_dir=RAW / "openorca")
    )
    log(f"Читаю {src} (только нужные столбцы)…")
    table = pq.read_table(src, columns=["id", "system_prompt", "question", "response"])
    n = table.num_rows
    log(f"Всего строк в GPT-4 части: {n}")
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(n, size=SAMPLE_SIZE, replace=False))
    sample = table.take(pa.array(idx))
    sample = sample.append_column("source_row", pa.array(idx, type=pa.int64()))
    pq.write_table(sample, out, compression="zstd")
    log(f"Записан срез: {out} ({out.stat().st_size / 1e6:.1f} МБ, {sample.num_rows} строк, seed={SEED})")
    return out


def fetch_gsm8k(force: bool) -> dict[str, Path]:
    outs: dict[str, Path] = {}
    for rel in GSM8K_FILES:
        split = "train" if "train" in rel else "test"
        out = DATA / f"gsm8k_{split}.parquet"
        outs[split] = out
        if out.exists() and not force:
            log(f"GSM8K {split} уже есть: {out}")
            continue
        log(f"Скачиваю {GSM8K_REPO}/{rel}…")
        src = Path(hf_hub_download(GSM8K_REPO, rel, repo_type="dataset", local_dir=RAW / "gsm8k"))
        table = pq.read_table(src)
        pq.write_table(table, out, compression="zstd")
        log(f"Записан {out} ({table.num_rows} строк)")
    return outs


def fetch_qwen_tokenizer(force: bool) -> Path:
    out = DATA / "tokenizer_qwen2.5"
    if (out / "tokenizer.json").exists() and not force:
        log(f"Токенизатор Qwen уже есть: {out}")
        return out
    log(f"Скачиваю файлы токенизатора {QWEN_REPO}…")
    snapshot_download(QWEN_REPO, allow_patterns=list(QWEN_TOKENIZER_FILES), local_dir=out)
    return out


# --------------------------------------------------------------------------- статистика


def _byte_stats(texts: list[str]) -> dict:
    """Энтропия байтовых униграмм/биграмм — «нулевые линейки» в битах на байт."""
    uni = np.zeros(256, dtype=np.int64)
    bi = np.zeros((256, 256), dtype=np.int64)
    total = 0
    for t in texts:
        b = np.frombuffer(t.encode("utf-8"), dtype=np.uint8)
        if b.size == 0:
            continue
        total += b.size
        uni += np.bincount(b, minlength=256)
        if b.size > 1:
            np.add.at(bi, (b[:-1], b[1:]), 1)
    p = uni / max(total, 1)
    h1 = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    row = bi.sum(axis=1, keepdims=True)
    cond = np.divide(bi, row, out=np.zeros_like(bi, dtype=np.float64), where=row > 0)
    joint = bi / max(bi.sum(), 1)
    mask = joint > 0
    h2 = float(-(joint[mask] * np.log2(cond[mask])).sum())
    return {
        "bytes_total": int(total),
        "unigram_entropy_bits_per_byte": round(h1, 4),
        "bigram_cond_entropy_bits_per_byte": round(h2, 4),
        "distinct_byte_values": int((uni > 0).sum()),
    }


def _qwen_stats(tok: Tokenizer, texts: list[str]) -> dict:
    counts: Counter[int] = Counter()
    lengths: list[int] = []
    bs = 2048
    for i in range(0, len(texts), bs):
        for enc in tok.encode_batch(texts[i : i + bs]):
            lengths.append(len(enc.ids))
            counts.update(enc.ids)
    total = sum(lengths)
    p = np.array(list(counts.values()), dtype=np.float64) / max(total, 1)
    h = float(-(p * np.log2(p)).sum())
    ln = np.array(lengths)
    return {
        "tokens_total": int(total),
        "distinct_tokens_used": len(counts),
        "unigram_entropy_bits_per_token": round(h, 4),
        "tokens_per_doc_mean": round(float(ln.mean()), 1),
        "tokens_per_doc_median": int(np.median(ln)),
        "tokens_per_doc_p95": int(np.percentile(ln, 95)),
        "tokens_per_doc_max": int(ln.max()),
    }


def _len_stats(texts: list[str]) -> dict:
    ln = np.array([len(t.encode("utf-8")) for t in texts])
    return {
        "docs": int(ln.size),
        "bytes_per_doc_mean": round(float(ln.mean()), 1),
        "bytes_per_doc_median": int(np.median(ln)),
        "bytes_per_doc_p95": int(np.percentile(ln, 95)),
        "bytes_per_doc_max": int(ln.max()),
    }


def corpus_stats(tok: Tokenizer, name: str, fields: dict[str, list[str]]) -> dict:
    """fields: имя поля → список строк. Считает по каждому полю и по склейке."""
    out: dict = {"name": name, "fields": {}}
    for fname, texts in fields.items():
        out["fields"][fname] = _len_stats(texts)
    joined = ["\n".join(parts) for parts in zip(*fields.values())]
    out["joined"] = {**_len_stats(joined), **_byte_stats(joined), "qwen": _qwen_stats(tok, joined)}
    return out


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="перекачать и пересчитать всё")
    ap.add_argument("--no-stats", action="store_true", help="только скачать, без статистики")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    RAW.mkdir(exist_ok=True)

    orca = fetch_openorca(args.force)
    gsm = fetch_gsm8k(args.force)
    tok_dir = fetch_qwen_tokenizer(args.force)

    manifest: dict = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "sources": {
            "openorca": f"{OPENORCA_REPO}/{OPENORCA_FILE}",
            "gsm8k": f"{GSM8K_REPO}/main",
            "qwen_tokenizer": QWEN_REPO,
        },
        "files": {},
    }
    for p in [orca, *gsm.values(), tok_dir / "tokenizer.json"]:
        manifest["files"][str(p.relative_to(ROOT))] = {"sha256": sha256(p), "bytes": p.stat().st_size}

    if not args.no_stats:
        tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
        manifest["qwen_vocab_size"] = tok.get_vocab_size()
        log("Статистика OpenOrca-100k…")
        t = pq.read_table(orca).to_pydict()
        manifest["stats_openorca_100k"] = corpus_stats(
            tok,
            "openorca_100k",
            {
                "system_prompt": [s or "" for s in t["system_prompt"]],
                "question": t["question"],
                "response": t["response"],
            },
        )
        manifest["stats_openorca_100k"]["id_prefixes"] = dict(
            Counter(i.split(".")[0] for i in t["id"]).most_common()
        )
        for split, path in gsm.items():
            log(f"Статистика GSM8K {split}…")
            g = pq.read_table(path).to_pydict()
            manifest[f"stats_gsm8k_{split}"] = corpus_stats(
                tok, f"gsm8k_{split}", {"question": g["question"], "answer": g["answer"]}
            )

    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    log(f"Манифест записан: {DATA / 'manifest.json'}")
    if not args.no_stats:
        o = manifest["stats_openorca_100k"]["joined"]
        log(
            f"OpenOrca-100k: {o['docs']} док., {o['bytes_total'] / 1e6:.1f} МБ, "
            f"{o['qwen']['tokens_total'] / 1e6:.2f} М токенов Qwen, "
            f"{o['qwen']['distinct_tokens_used']} различных токенов; "
            f"H1={o['unigram_entropy_bits_per_byte']} H2={o['bigram_cond_entropy_bits_per_byte']} бит/байт"
        )
        for split in gsm:
            g = manifest[f"stats_gsm8k_{split}"]["joined"]
            log(
                f"GSM8K {split}: {g['docs']} док., {g['bytes_total'] / 1e6:.2f} МБ, "
                f"{g['qwen']['tokens_total'] / 1e3:.0f} К токенов Qwen"
            )
        if not math.isfinite(o["unigram_entropy_bits_per_byte"]):
            raise SystemExit("энтропия не конечна — корпус пуст?")


if __name__ == "__main__":
    main()
