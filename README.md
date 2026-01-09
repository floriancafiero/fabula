# fabula
A Python package to study the evolution of a narration through sentiment and emotion analysis. 

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
