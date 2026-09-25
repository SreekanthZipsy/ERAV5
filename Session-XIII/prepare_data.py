"""Stream FineWeb-Edu, train an 8k byte-level BPE, and write uint16 token shards.

Output (in data/):
  tokenizer.json   8192-vocab byte-level BPE
  train.bin        >= TRAIN_TOKENS tokens (uint16)
  val.bin          VAL_TOKENS tokens (uint16)
"""

import argparse
import os
import time

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def doc_stream():
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds:
        yield row["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--train_tokens", type=int, default=52_000_000)
    ap.add_argument("--val_tokens", type=int, default=1_000_000)
    ap.add_argument("--tok_train_docs", type=int, default=40_000)
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    t0 = time.time()
    stream = doc_stream()
    tok_docs = [next(stream) for _ in range(args.tok_train_docs)]
    print(f"fetched {len(tok_docs)} docs for tokenizer training in {time.time()-t0:.0f}s")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(tok_docs, trainer=trainer)
    tok.save(os.path.join(DATA, "tokenizer.json"))
    eot = tok.token_to_id("<|endoftext|>")
    print(f"tokenizer trained in {time.time()-t0:.0f}s, vocab={tok.get_vocab_size()}")

    # tokenizer-training docs are reused as the start of the val split
    need = args.train_tokens + args.val_tokens
    chunks, total, ndocs = [], 0, 0

    def encode(batch):
        nonlocal total, ndocs
        for enc in tok.encode_batch(batch):
            ids = np.asarray(enc.ids + [eot], dtype=np.uint16)
            chunks.append(ids)
            total += len(ids)
        ndocs += len(batch)

    encode(tok_docs)
    batch = []
    for text in stream:
        if total >= need:
            break
        batch.append(text)
        if len(batch) == 2000:
            encode(batch)
            batch = []
            print(f"  {total/1e6:6.1f}M tokens, {ndocs} docs, {time.time()-t0:.0f}s", flush=True)
    if batch:
        encode(batch)

    all_ids = np.concatenate(chunks)
    val = all_ids[: args.val_tokens]
    train = all_ids[args.val_tokens : args.val_tokens + args.train_tokens]
    val.tofile(os.path.join(DATA, "val.bin"))
    train.tofile(os.path.join(DATA, "train.bin"))
    chars = sum(len(t) for t in tok_docs)
    toks = sum(len(c) for c in chunks[: len(tok_docs)])
    print(f"done: train={len(train)/1e6:.1f}M val={len(val)/1e6:.1f}M docs={ndocs} "
          f"chars/token={chars/toks:.2f} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
