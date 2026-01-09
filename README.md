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

## Quickstart

### Python

```python
from fabula.core import Fabula
from fabula.scorer import TransformersScorer

scorer = TransformersScorer(model="cmarkea/distilcamembert-base-sentiment")
fb = Fabula(scorer=scorer)

df = fb.score("Bonjour. C'est une belle journée.")
print(df[["rel_pos", "label", "score"]])

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

## Segmentation strategies

Fabula supports multiple segmenters via `--segment`:

- `sentence` (default)
- `paragraph`
- `window` (token sliding windows)
- `document` (sentence segments with coarse document chunks)

Window segmentation options:

```bash
fabula score my.txt --segment window --window-tokens 256 --stride-tokens 64
```

Document chunking with interpolation between sentence and chunk scores:

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
