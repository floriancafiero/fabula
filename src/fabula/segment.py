from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import re
import warnings
import logging

from .schemas import Segment

_whitespace_re = re.compile(r"\s+")


def _clean(s: str) -> str:
    return _whitespace_re.sub(" ", s).strip()


@dataclass
class ParagraphSegmenter:
    min_len: int = 1
    verbose: bool = False

    def segment(self, text: str) -> List[Segment]:
        if self.verbose:
            print(f"[ParagraphSegmenter] Segmenting text of {len(text)} characters")
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

        if self.verbose:
            print(f"[ParagraphSegmenter] Created {len(segments)} segments")
        return segments


@dataclass
class RegexSentenceSegmenter:
    pattern: str = r"(?<=[\.\!\?…])\s+"
    min_len: int = 1
    verbose: bool = False

    def segment(self, text: str) -> List[Segment]:
        if self.verbose:
            print(f"[RegexSentenceSegmenter] Segmenting text of {len(text)} characters")
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

        if self.verbose:
            print(f"[RegexSentenceSegmenter] Created {len(segments)} segments")
        return segments


@dataclass
class SlidingWindowTokenSegmenter:
    """
    Token-window segmenter for long texts.

    Character offsets are not exact here; we expose token offsets instead.
    """
    window_tokens: int = 256
    stride_tokens: int = 64
    min_tokens: int = 16
    tokenizer: any = None
    decode_kwargs: Optional[dict] = None
    verbose: bool = False

    def segment(self, text: str) -> List[Segment]:
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided. Either pass it during initialization or use Fabula to automatically configure it.")

        if self.verbose:
            print(f"[SlidingWindowTokenSegmenter] Segmenting text of {len(text)} characters")

        if self.decode_kwargs is None:
            self.decode_kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": True}

        # Suppress the warning about sequence length since we handle segmentation ourselves
        # We need to suppress both Python warnings and transformers logging
        transformers_logger = logging.getLogger("transformers.tokenization_utils_base")
        original_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enc = self.tokenizer(text,
                                    add_special_tokens=False,
                                    return_attention_mask=False,
                                    return_tensors=None,
                                    truncation=False,
                                    max_length=None,
)
        finally:
            transformers_logger.setLevel(original_level)
        input_ids: Sequence[int] = enc["input_ids"]
        n_tokens = len(input_ids)
        if self.verbose:
            print(f"[SlidingWindowTokenSegmenter] Tokenized into {n_tokens} tokens")
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

        if self.verbose:
            print(f"[SlidingWindowTokenSegmenter] Created {len(segments)} segments")
        return segments


@dataclass
class DocumentChunkTokenSegmenter:
    """
    Chunk segmenter for long documents using tokenizer offsets.

    Uses token offsets to estimate character positions for alignment.
    """
    chunk_tokens: int = 512
    stride_tokens: int = 5
    min_tokens: int = 128
    tokenizer: any = None
    decode_kwargs: Optional[dict] = None
    verbose: bool = False

    def segment(self, text: str) -> List[Segment]:
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided. Either pass it during initialization or use Fabula to automatically configure it.")

        if self.verbose:
            print(f"[DocumentChunkTokenSegmenter] Segmenting text of {len(text)} characters")

        if self.decode_kwargs is None:
            self.decode_kwargs = {"skip_special_tokens": True, "clean_up_tokenization_spaces": True}

        if self.stride_tokens <= 0:
            raise ValueError("stride_tokens must be positive.")

        # Suppress the warning about sequence length since we handle segmentation ourselves
        # We need to suppress both Python warnings and transformers logging
        transformers_logger = logging.getLogger("transformers.tokenization_utils_base")
        original_level = transformers_logger.level
        transformers_logger.setLevel(logging.ERROR)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                enc = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_offsets_mapping=True,
                    return_tensors=None,
                    truncation=False,
                    max_length=None,
                )
        finally:
            transformers_logger.setLevel(original_level)
        input_ids: Sequence[int] = enc["input_ids"]
        offsets = enc.get("offset_mapping")
        if offsets is None:
            raise ValueError("Tokenizer must provide offsets for in-context chunking.")

        n_tokens = len(input_ids)
        if self.verbose:
            print(f"[DocumentChunkTokenSegmenter] Tokenized into {n_tokens} tokens")
        if n_tokens == 0:
            return []

        segments: List[Segment] = []
        idx = 0
        n_chars = len(text)

        start = 0
        while start < n_tokens:
            end = min(start + self.chunk_tokens, n_tokens)
            length = end - start
            if length < self.min_tokens:
                break

            chunk_ids = input_ids[start:end]
            chunk_text = self.tokenizer.decode(chunk_ids, **self.decode_kwargs).strip()
            if chunk_text:
                start_char = offsets[start][0]
                end_char = offsets[end - 1][1]
                rel_pos = start_char / max(n_chars, 1)
                segments.append(
                    Segment(
                        idx=idx,
                        text=chunk_text,
                        rel_pos=rel_pos,
                        start_char=start_char,
                        end_char=end_char,
                        start_token=start,
                        end_token=end,
                    )
                )
                idx += 1

            if end == n_tokens:
                break
            start += self.stride_tokens

        if self.verbose:
            print(f"[DocumentChunkTokenSegmenter] Created {len(segments)} segments")
        return segments
