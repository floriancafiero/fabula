# Fabula

Fabula is a Python package for analyzing how sentiment or emotions evolve across a document. It slices text into segments, scores each segment with a Transformers model, and optionally produces a smoothed narrative arc.

> **Language support**: The bundled defaults target contemporary **French** models, but you can supply any Hugging Face sequence-classification model that matches your analysis labels.

## Key capabilities

- **Per-segment scoring** for sentiment or emotion analysis.
- **Multiple segmentation strategies** (sentence, paragraph, token windows, document chunks).
- **Long-input handling** with chunk pooling and document-level interpolation.
- **Narrative arc generation** with resampling + smoothing.
- **CLI and Python API** for batch runs and scripting.

## Installation

Base install:

```bash
pip install fabula
```

Transformers support (required for real models):

```bash
pip install fabula[transformers]
```

Some Hugging Face models require SentencePiece:

```bash
pip install fabula[transformers-sp]
# or
pip install sentencepiece
```

Optional plotting support:

```bash
pip install fabula[plot]
```

## Quickstart

### Python

```python
from fabula.core import Fabula
from fabula.scorer import TransformersScorer

scorer = TransformersScorer(model="cmarkea/distilcamembert-base-sentiment")
fb = Fabula(scorer=scorer)

# Per-segment scoring
scores = fb.score("Bonjour. C'est une belle journée.")
print(scores[["rel_pos", "label", "score"]])

# Narrative arc
arc = fb.arc("Bonjour. C'est une belle journée.", n_points=50)
print(arc.x[:5], arc.y[:5])
```

### CLI

Score a document and return per-segment results:

```bash
fabula score my.txt --format json
```

Compute a narrative arc:

```bash
fabula arc my.txt --n-points 100 --smooth-window 9
```

## Models and analysis types

Fabula supports two analysis modes:

- **sentiment** (default)
- **emotion**

Default models (used when `--model` is not provided):

- Sentiment: `cmarkea/distilcamembert-base-sentiment`
- Emotion: `astrosbd/french_emotion_camembert`

You can override them with any Hugging Face model ID:

```bash
fabula score my.txt --analysis emotion --model j-hartmann/emotion-english-distilroberta-base
```

## Segmentation strategies

Choose a segmentation strategy with `--segment` (CLI) or provide a custom `segmenter` in the Python API. Each segment yields a relative position (`rel_pos`) within the document.

### 1) Sentence segmentation (default)

Splits on sentence-ending punctuation (regex-based).

```bash
fabula score my.txt --segment sentence
```

### 2) Paragraph segmentation

Splits on blank lines (`\n\n`). Useful for prose or articles with clear paragraph breaks.

```bash
fabula score my.txt --segment paragraph
```

### 3) Window segmentation (token sliding windows)

Uses the tokenizer to create overlapping windows. Requires a Transformers tokenizer (not available in `--dummy` mode).

```bash
fabula score my.txt --segment window --window-tokens 256 --stride-tokens 64 --min-tokens 16
```

### 4) Document chunking + interpolation

Scores sentences *and* coarse chunks, then blends their probabilities to preserve long-range context. Requires a tokenizer with offset mappings.

```bash
fabula score my.txt \
  --segment document \
  --chunk-tokens 1024 \
  --chunk-stride-tokens 1024 \
  --chunk-min-tokens 128 \
  --chunk-weight 0.3 \
  --chunk-attention-tau 0.1
```

- `chunk-weight` controls how much chunk scores influence sentence scores.
- `chunk-attention-tau` controls the distance decay for chunk influence.

## Long-input pooling

For very long segments, you can pool probabilities across overflowing chunks instead of truncating. Pooling options:

- `none` (default)
- `mean`
- `max`
- `attention` (softmax-weighted by max prob per chunk)

```bash
fabula score my.txt --pooling mean --pooling-stride-tokens 128
```

## Narrative arc generation

The `arc` command (or `Fabula.arc`) turns segment scores into a continuous curve.

- **Resampling**: points are interpolated to `--n-points`.
- **Smoothing**: set `--smooth-method` to `moving_average`, `gaussian`, or `none`.
- **Padding mode**: `reflect`, `edge`, or `constant`.

Example:

```bash
fabula arc my.txt \
  --n-points 200 \
  --smooth-method gaussian \
  --smooth-window 11 \
  --smooth-sigma 2.0 \
  --smooth-pad-mode reflect
```

### Plotting

The CLI can optionally plot the arc if matplotlib is installed.

```bash
fabula arc my.txt --plot arc.png
# or show interactively
fabula arc my.txt --plot -
```

## Output formats

### `fabula score`

Formats:

- `csv` (default)
- `json`
- `jsonl`

Each row includes:

- `idx`: segment index
- `rel_pos`: relative position in the document (0..1)
- `text`: segment text
- `label`: top predicted label
- `score`: scalar score for arcs
- `probs`: full label distribution
- `chunk_probs`: pooled chunk distribution (document mode only)
- `start_char`, `end_char`: character offsets (when available)
- `start_token`, `end_token`: token offsets (window/document modes)

### `fabula arc`

Formats:

- `csv` (default): `x`, `y`
- `json`: `x`, `y`, `raw_x`, `raw_y`

`raw_x` and `raw_y` are the original segment positions and scores before resampling/smoothing.

## CLI reference

The CLI has two subcommands: `score` and `arc`. Both share common options.

### Common options

- `input`: text file path or `-` for stdin
- `-o, --output`: output path or `-` for stdout
- `--encoding`: file encoding (default: `utf-8`)
- `--dummy`: use a tiny built-in scorer (no Transformers download)
- `--analysis`: `sentiment` or `emotion`
- `--model`: Hugging Face model ID (ignored with `--dummy`)
- `--device`: `cpu`, `cuda`, or `cuda:0`
- `--batch-size`: inference batch size
- `--max-length`: max tokens per segment
- `--pooling`: `none`, `mean`, `max`, `attention`
- `--pooling-stride-tokens`: stride for pooled chunking (defaults to `max_length/4`)
- `--segment`: `sentence`, `paragraph`, `window`, `document`
- `--window-tokens`, `--stride-tokens`, `--min-tokens`: window segmentation controls
- `--chunk-tokens`, `--chunk-stride-tokens`, `--chunk-min-tokens`: document chunking controls
- `--chunk-weight`: interpolation weight for chunk scores
- `--chunk-attention-tau`: attention pooling temperature for chunk scores

### `fabula score` options

- `--format`: `csv`, `json`, or `jsonl`

### `fabula arc` options

- `--format`: `csv` or `json`
- `--n-points`: number of resampled points
- `--smooth-window`: smoothing window size
- `--smooth-method`: `moving_average`, `gaussian`, or `none`
- `--smooth-sigma`: gaussian sigma (defaults to `window/6`)
- `--smooth-pad-mode`: `reflect`, `edge`, or `constant`
- `--score-col`: column to use as scalar score (default: `score`)
- `--no-fallback-to-maxprob`: disable fallback score for missing scalar values
- `--plot`: output file path or `-` to display

## Python API reference

### `fabula.core.Fabula`

```python
Fabula(
    scorer,
    segmenter=None,
    coarse_segmenter=None,
    analysis="sentiment",
    chunk_weight=0.3,
    chunk_attention_tau=0.1,
)
```

Methods:

- `score(text) -> pandas.DataFrame`
- `arc(text, n_points=100, smooth_window=7, smooth_method="moving_average", smooth_sigma=None, smooth_pad_mode="reflect", score_col="score", fallback_to_maxprob=True) -> ArcResult`

### Segmenters

- `RegexSentenceSegmenter(pattern=..., min_len=1)`
- `ParagraphSegmenter(min_len=1)`
- `SlidingWindowTokenSegmenter(tokenizer, window_tokens=256, stride_tokens=64, min_tokens=16)`
- `DocumentChunkTokenSegmenter(tokenizer, chunk_tokens=1024, stride_tokens=1024, min_tokens=128)`

### Scoring

- `TransformersScorer(model, device=None, batch_size=16, max_length=512, pooling="none", pooling_stride_tokens=None)`

## Practical recipes

### Quick sentiment arc

```bash
fabula arc my.txt --analysis sentiment --n-points 100
```

### Emotion arc with custom model

```bash
fabula arc my.txt --analysis emotion --model astrosbd/french_emotion_camembert
```

### Long document, fewer API calls

```bash
fabula score my.txt --segment document --chunk-tokens 2048 --chunk-weight 0.4
```

### Smoke-test without downloading models

```bash
fabula score my.txt --dummy --analysis sentiment
```

## Notes & limitations

- The default models are French. For other languages, pass a different Hugging Face model.
- `window` and `document` segmentation require a Transformers tokenizer (disable `--dummy`).
- `document` segmentation requires a tokenizer that returns offset mappings.
- `score` outputs `score=None` when the model labels do not support valence; `arc` can fall back to max probability unless disabled.

## License

Licensed under the MIT License.
