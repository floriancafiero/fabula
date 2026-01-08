from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import re

from .schemas import Segment

_whitespace_re = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _whitespace_re.sub(" ", s).strip()


@dataclass
class ParagraphSegmenter:
    min_len: int = 1

    def segment(self, text: str) -> List[Segment]:
        segments: List[Segment] = []
        n_chars = len(text)
        idx = 0

        cursor = 0
        for block in text.split("\n\n"):
            raw = block
            cleaned = _clean(raw)
            start_char = cursor
            end_char = cursor + len(raw)

            cursor = end_char + 2  # rough, because we split on "\n\n"
            if not cleaned or len(cleaned) < self.min_len:
                continue

            rel_pos = start_char / max(n_chars, 1)
            segments.append(
                Segment(
                    idx=idx,
                    text=cleaned,
                    rel_pos=rel_pos,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            idx += 1

        return segments


@dataclass
class RegexSentenceSegmenter:
    pattern: str = r"(?<=[\.\!\?…])\s+"
    min_len: int = 1

    def segment(self, text: str) -> List[Segment]:
        parts = re.split(self.pattern, text)
        segments: List[Segment] = []
        n_chars = len(text)

        idx = 0
        cursor = 0
        for p in parts:
            cleaned = _clean(p)
            if not cleaned or len(cleaned) < self.min_len:
                cursor += len(p) + 1
                continue

            start_char = text.find(p, cursor)
            if start_char < 0:
                start_char = cursor
            end_char = start_char + len(p)
            cursor = end_char

            rel_pos = start_char / max(n_chars, 1)
            segments.append(
                Segment(
                    idx=idx,
                    text=cleaned,
                    rel_pos=rel_pos,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            idx += 1

        return segments


@dataclass
class SlidingWindowTokenSegmenter:
    """
    Token-window segmenter for long texts.

    Character offsets are not exact here; we expose token offsets instead.
    """
    tokenizer: any
    window_tokens: int = 256
    stride_tokens: int = 64
    min_tokens: int = 16
    decode_kwargs: Optional[dict] = None

    def segment(self, text: str) -> List[Segment]:
        if self.decode_kwargs is None:
            self.decode_kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": True}

        enc = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_tensors=None)
        input_ids: Sequence[int] = enc["input_ids"]
        n_tokens = len(input_ids)
        if n_tokens == 0:
            return []

        segments: List[Segment] = []
        idx = 0

        start = 0
        while start < n_tokens:
            end = min(start + self.window_tokens, n_tokens)
            length = end - start
            if length < self.min_tokens:
                break

            window_ids = input_ids[start:end]
            window_text = self.tokenizer.decode(window_ids, **self.decode_kwargs).strip()
            if window_text:
                rel_pos = start / max(n_tokens, 1)
                segments.append(
                    Segment(
                        idx=idx,
                        text=window_text,
                        rel_pos=rel_pos,
                        start_token=start,
                        end_token=end,
                    )
                )
                idx += 1

            if end == n_tokens:
                break
            start += self.stride_tokens

        return segments

