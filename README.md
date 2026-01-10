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
from fabula.plot import plot_arc_series

scorer = TransformersScorer(model="cmarkea/distilcamembert-base-sentiment")
fb = Fabula(scorer=scorer)

# Per-segment scoring
scores = fb.score("Bonjour. C'est une belle journée.")
print(scores[["rel_pos", "label", "score"]])

# Narrative arc
arc = fb.arc("Bonjour. C'est une belle journée.", n_points=50)
print(arc.x[:5], arc.y[:5])

# Multi-dimensional emotion arc (per-label probabilities)
fb_emotion = Fabula(scorer=TransformersScorer(model="astrosbd/french_emotion_camembert"), analysis="emotion")
emotion_arc = fb_emotion.arc("Bonjour. Quelle surprise.", score_col="probs")
plot_arc_series(emotion_arc, title="Emotions across the story", legend_title="Emotions")
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

Compute a multi-dimensional emotion arc (one column per label):

```bash
fabula arc my.txt --analysis emotion --score-col probs --format csv
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

### Multi-dimensional arcs (emotion or multi-label scores)

If your segments include **multiple labels** (e.g., emotions), Fabula can emit
**one arc per label** instead of collapsing everything into a single scalar.

Two ways to request vector arcs:

1. **Per-label probabilities** from the `probs` column:

```bash
fabula arc my.txt --analysis emotion --score-col probs --format csv
```

This produces a CSV like:

```
x,JOIE,TRISTESSE,PEUR,SURPRISE
0.0,0.12,0.55,0.07,0.26
0.01,0.15,0.52,0.06,0.27
...
```

2. **Explicit columns** using `--score-cols` (comma-separated). This is useful
if you already have multiple numeric columns in your scoring output and want
independent arcs for each:

```bash
fabula arc my.txt --score-cols valence,arousal,dominance
```

### Plotting multi-dimensional arcs

The scalar plot function fills under a single curve. For multi-dimensional arcs,
use `plot_arc_series` to get multiple lines and a legend:

```python
from fabula import Fabula, plot_arc_series
from fabula.scorer import TransformersScorer

fb = Fabula(scorer=TransformersScorer(model="astrosbd/french_emotion_camembert"), analysis="emotion")
arc = fb.arc("Bonjour. Quelle surprise.", score_col="probs")
plot_arc_series(arc, title="Smoothed Evolution of Emotions", legend_title="Emotions")
```

### Plotting

The CLI can optionally plot the arc if matplotlib is installed. For quick teaching visuals,
use `--plot-raw` to show the underlying segment scores alongside the smoothed curve.
For multi-dimensional arcs (e.g., `--score-col probs`), the CLI will draw multiple lines
with a legend instead of a filled scalar curve.

```bash
fabula arc my.txt --plot arc.png
# or show interactively
fabula arc my.txt --plot -
fabula arc my.txt --plot arc.png --plot-raw
```

You can also create plots programmatically:

```python
from fabula import Fabula, plot_arc

arc = Fabula().arc(\"...\")
plot_arc(arc, subtitle=\"Chapter 1: Opening scene\", raw_points=True, save_path=\"arc.png\")
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

- `csv` (default): `x`, `y` for scalar arcs, or `x` + one column per label for vector arcs
- `json`: scalar arcs include `x`, `y`, `raw_x`, `raw_y`; vector arcs include `x`, `y_series`, `raw_x`, `raw_y_series`

`raw_x` and `raw_y` are the original segment positions and scores before resampling/smoothing. For vector arcs,
the `*_series` fields store one list per label.

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
- `--score-col`: column to use as scalar score (default: `score`; use `probs` for per-label arcs)
- `--score-cols`: comma-separated columns for vector arcs (overrides `--score-col`)
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
- `arc(text, n_points=100, smooth_window=7, smooth_method="moving_average", smooth_sigma=None, smooth_pad_mode="reflect", score_col="score", score_cols=None, fallback_to_maxprob=True) -> ArcResult`

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

### Emotion arc as multi-column CSV

```bash
fabula arc my.txt --analysis emotion --score-col probs --format csv
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
