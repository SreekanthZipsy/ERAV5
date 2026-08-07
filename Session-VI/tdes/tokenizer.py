"""Frozen character/word hybrid tokenizer with a content hash.

Tiny on purpose: prove freeze + hash integrity, not vocabulary quality.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .hashutil import sha256_json, sha256_text


SPECIAL = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
    "<mask_obs>": 4,  # agentic observation (context only, no loss)
}


@dataclass(frozen=True)
class TokenizerSpec:
    vocab: dict[str, int]
    version: str = "tdes-tok-v1"
    algo: str = "whitespace_char_fallback"

    @property
    def pad_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.vocab["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.vocab["<eos>"]

    @property
    def hash(self) -> str:
        return sha256_json({"version": self.version, "algo": self.algo, "vocab": self.vocab})


@dataclass
class FrozenTokenizer:
    spec: TokenizerSpec
    _id_to_token: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_id_to_token",
            {i: t for t, i in self.spec.vocab.items()},
        )

    @property
    def hash(self) -> str:
        return self.spec.hash

    @property
    def vocab_size(self) -> int:
        return len(self.spec.vocab)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.spec.bos_id)
        for tok in _tokenize(text):
            ids.append(self.spec.vocab.get(tok, self.spec.unk_id))
        if add_eos:
            ids.append(self.spec.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts = []
        for i in ids:
            t = self._id_to_token.get(i, "<unk>")
            if t in SPECIAL:
                continue
            parts.append(t)
        return " ".join(parts)

    def save(self, path: Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "version": self.spec.version,
            "algo": self.spec.algo,
            "vocab": self.spec.vocab,
            "tokenizer_hash": self.hash,
        }
        path.write_text(json.dumps(blob, indent=2, sort_keys=True), encoding="utf-8")
        return self.hash

    @classmethod
    def load(cls, path: Path) -> "FrozenTokenizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        spec = TokenizerSpec(vocab={k: int(v) for k, v in blob["vocab"].items()}, version=blob["version"], algo=blob["algo"])
        tok = cls(spec)
        if tok.hash != blob["tokenizer_hash"]:
            raise ValueError("tokenizer_hash mismatch — tokenizer is not frozen as recorded")
        return tok


def _tokenize(text: str) -> list[str]:
    # Keep special markers intact; otherwise whitespace + single-char fallback for rare glyphs.
    parts = re.findall(r"<[^>]+>|[A-Za-z0-9_]+|[^\s]", text)
    return parts


def build_tokenizer_from_texts(texts: list[str], extra_tokens: list[str] | None = None) -> FrozenTokenizer:
    vocab = dict(SPECIAL)
    next_id = max(vocab.values()) + 1
    for t in texts:
        for tok in _tokenize(t):
            if tok not in vocab:
                vocab[tok] = next_id
                next_id += 1
    for tok in extra_tokens or []:
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1
    # Freeze: content hash is identity.
    _ = sha256_text(json.dumps(vocab, sort_keys=True))
    return FrozenTokenizer(TokenizerSpec(vocab=vocab))
