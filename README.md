# fabula
A Python package to study the evolution of sentiment and emotions throughout a document. 

Implementation only for contemporary French language for now.

## Installation
Base install:

```bash
pip install fabula
```

Transformers support:

```bash
pip install fabula[transformers]
```

Some Hugging Face models require SentencePiece. If you see an error about missing
`sentencepiece`, install the extra below or add the package directly:

```bash
pip install fabula[transformers-sp]
# or
pip install sentencepiece
```

## What Fabula produces

Fabula reads a document, splits it into segments, scores each segment, and then
optionally smooths the scores into a narrative arc.

Concepts:

- **Segment**: a sentence/paragraph/window/chunk of text.
- **Score**: a scalar derived from model probabilities (valence for sentiment,
  max-probability for emotion).
- **Arc**: the smoothed evolution of scores across the document.

## Quickstart

### Python

```python
from fabula.core import Fabula
from fabula.scorer import TransformersScorer

# 1) Choose analysis + model (sentiment or emotion).
# Sentiment is the default analysis mode.
scorer = TransformersScorer(model="cmarkea/distilcamembert-base-sentiment")

# 2) Build Fabula with a segmenter (defaults to sentence segmentation).
fb = Fabula(scorer=scorer)

text = "Bonjour. C'est une belle journée. Pourtant, je suis inquiet."

# 3) Score each segment.
df = fb.score(text)
print(df[["rel_pos", "label", "score"]])

# 4) Produce a smooth narrative arc.
arc = fb.arc(text, n_points=50)
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

## CLI reference

Common arguments (score and arc):

- `input`: file path or `-` for stdin
- `-o, --output`: output path or `-` for stdout
- `--encoding`: input file encoding
- `--dummy`: use the built-in dummy scorer (no transformers download)
- `--analysis`: `sentiment` or `emotion`
- `--model`: Hugging Face model id (ignored with `--dummy`)
- `--device`: `cpu`, `cuda`, `cuda:0` (default: auto)
- `--batch-size`: batch size for inference
- `--max-length`: max tokens per segment fed to the model
- `--pooling`: `none`, `mean`, `max`, `attention` (long-input pooling)
- `--pooling-stride-tokens`: stride for pooled chunking
- `--segment`: `sentence`, `paragraph`, `window`, `document`
- `--window-tokens`: token window size (segment=`window`)
- `--stride-tokens`: token stride (segment=`window`)
- `--min-tokens`: min tokens for window segments
- `--chunk-tokens`: chunk token size (segment=`document`)
- `--chunk-stride-tokens`: chunk stride (segment=`document`)
- `--chunk-min-tokens`: min tokens for document chunks
- `--chunk-weight`: interpolation weight for chunk scores
- `--chunk-attention-tau`: attention pooling temperature for chunk blending

Score-only arguments:

- `--format`: `csv`, `json`, `jsonl`

Arc-only arguments:

- `--format`: `csv`, `json`
- `--n-points`: number of arc points
- `--smooth-window`: smoothing window size
- `--smooth-method`: `moving_average`, `gaussian`, `none`
- `--smooth-sigma`: gaussian sigma (only with `gaussian`)
- `--smooth-pad-mode`: `reflect`, `edge`, `constant`
- `--score-col`: column to use as scalar score
- `--no-fallback-to-maxprob`: disable fallback scalar when score is missing
- `--plot`: output plot file path or `-` to display (requires matplotlib)

## Sentiment vs emotion

Use `--analysis sentiment` when you want a positive/negative valence curve.
Use `--analysis emotion` when you want the strongest emotion intensity per
segment. Choose a matching model with `--model`.

Examples:

```bash
fabula arc my.txt --analysis sentiment
fabula arc my.txt --analysis emotion --model astrosbd/french_emotion_camembert
```

## How segmentation works

Segmentation controls what each model pass sees. Shorter segments capture local
sentiment shifts; longer segments preserve context.

Fabula supports multiple segmenters via `--segment`:

- `sentence` (default)
- `paragraph`
- `window` (token sliding windows)
- `document` (sentence segments with coarse document chunks)

### Window segmentation (token sliding windows)

Use window segmentation for long texts with overlap:

```bash
fabula score my.txt --segment window --window-tokens 256 --stride-tokens 64
```

### Document chunking + interpolation

The `document` mode scores sentences and coarse document chunks, then blends
their probabilities. Use this when sentence-only scoring misses long-range
context.

```bash
fabula score my.txt \
  --segment document \
  --chunk-tokens 1024 \
  --chunk-stride-tokens 1024 \
  --chunk-weight 0.3 \
  --chunk-attention-tau 0.1
```

## Long-input pooling

For long inputs, you can pool scores across overflowing chunks instead of
truncating a single sequence. Configure pooling and stride:

```bash
fabula score my.txt --pooling mean --pooling-stride-tokens 128
```

## Graphic output

Fabula does not generate plots by default. The CLI can save a plot when you use
`fabula arc` with `--plot` (requires the optional matplotlib extra).

## Output structure

`fabula score` returns one row per segment with:

- `rel_pos`: segment position in the document (0..1)
- `label`: top predicted label
- `score`: scalar score used for arcs
- `probs`: probability distribution
- `chunk_probs`: pooled chunk probabilities (only in `document` mode)

## Common recipes

### Quick sentiment arc

```bash
fabula arc my.txt --analysis sentiment --n-points 100
```

### Emotion arc

```bash
fabula arc my.txt --analysis emotion --n-points 100
```

### Long document, fewer API calls

```bash
fabula score my.txt --segment document --chunk-tokens 2048 --chunk-weight 0.4
```
