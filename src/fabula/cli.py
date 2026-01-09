from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .core import Fabula
from .segment import ParagraphSegmenter, RegexSentenceSegmenter, SlidingWindowTokenSegmenter


DEFAULT_MODELS = {
    "sentiment": "cmarkea/distilcamembert-base-sentiment",
    "emotion": "astrosbd/french_emotion_camembert",
}


@dataclass
class DummyScorer:
    """
    Tiny scorer for CLI smoke tests (no transformers, no downloads).
    Produces stable probabilities so arc/score pipelines can be tested.
    """
    analysis: str = "sentiment"

    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for t in texts:
            # tiny heuristic to vary a bit
            if any(w in t.lower() for w in ["triste", "peur", "colère", "haine"]):
                if self.analysis == "emotion":
                    out.append({"TRISTESSE": 0.7, "PEUR": 0.2, "JOIE": 0.1})
                else:
                    out.append({"POSITIVE": 0.2, "NEGATIVE": 0.8})
            else:
                if self.analysis == "emotion":
                    out.append({"JOIE": 0.7, "SURPRISE": 0.2, "NEUTRE": 0.1})
                else:
                    out.append({"POSITIVE": 0.7, "NEGATIVE": 0.3})
        return out


def _read_text(input_path: str, encoding: str = "utf-8") -> str:
    if input_path == "-":
        return sys.stdin.read()
    p = Path(input_path)
    return p.read_text(encoding=encoding)


def _write_text(output_path: str, content: str) -> None:
    if output_path == "-":
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        return
    Path(output_path).write_text(content, encoding="utf-8")


def _df_to_json_records(df: pd.DataFrame) -> str:
    # ensure probs dict is JSON-serializable
    rows = []
    for _, r in df.iterrows():
        d = r.to_dict()
        rows.append(d)
    return json.dumps(rows, ensure_ascii=False)


def _df_to_jsonl(df: pd.DataFrame) -> str:
    lines = []
    for _, r in df.iterrows():
        d = r.to_dict()
        lines.append(json.dumps(d, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def _load_transformers_scorer(
    model: str,
    device: Optional[str],
    batch_size: int,
    max_length: int,
    temperature: float,
):
    # import lazily so CLI can run in dummy mode without transformers installed
    from .scorer import TransformersScorer
    return TransformersScorer(
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        temperature=temperature,
    )


def _make_segmenter(kind: str, scorer_or_none, window_tokens: int, stride_tokens: int, min_tokens: int):
    if kind == "sentence":
        return RegexSentenceSegmenter()
    if kind == "paragraph":
        return ParagraphSegmenter()
    if kind == "window":
        if scorer_or_none is None or getattr(scorer_or_none, "tokenizer", None) is None:
            raise ValueError("Window segmentation requires a transformers tokenizer (disable --dummy).")
        return SlidingWindowTokenSegmenter(
            tokenizer=scorer_or_none.tokenizer,
            window_tokens=window_tokens,
            stride_tokens=stride_tokens,
            min_tokens=min_tokens,
        )
    raise ValueError(f"Unknown segmenter kind: {kind}")


def _resolve_model(analysis: str, model: Optional[str]) -> str:
    if model is not None:
        return model
    return DEFAULT_MODELS[analysis]


def cmd_score(args: argparse.Namespace) -> int:
    text = _read_text(args.input, encoding=args.encoding)

    model = _resolve_model(args.analysis, args.model)
    scorer = DummyScorer(analysis=args.analysis) if args.dummy else _load_transformers_scorer(
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
    )

    segmenter = _make_segmenter(
        kind=args.segment,
        scorer_or_none=None if args.dummy else scorer,
        window_tokens=args.window_tokens,
        stride_tokens=args.stride_tokens,
        min_tokens=args.min_tokens,
    )

    fb = Fabula(scorer=scorer, segmenter=segmenter, analysis=args.analysis)
    df = fb.score(text)

    fmt = args.format.lower()
    if fmt == "csv":
        content = df.to_csv(index=False)
    elif fmt == "json":
        content = _df_to_json_records(df)
    elif fmt == "jsonl":
        content = _df_to_jsonl(df)
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    _write_text(args.output, content)
    return 0


def cmd_arc(args: argparse.Namespace) -> int:
    text = _read_text(args.input, encoding=args.encoding)

    model = _resolve_model(args.analysis, args.model)
    scorer = DummyScorer(analysis=args.analysis) if args.dummy else _load_transformers_scorer(
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
    )

    segmenter = _make_segmenter(
        kind=args.segment,
        scorer_or_none=None if args.dummy else scorer,
        window_tokens=args.window_tokens,
        stride_tokens=args.stride_tokens,
        min_tokens=args.min_tokens,
    )

    fb = Fabula(scorer=scorer, segmenter=segmenter, analysis=args.analysis)
    arc = fb.arc(
        text,
        n_points=args.n_points,
        smooth_window=args.smooth_window,
        smooth_method=args.smooth_method,
        smooth_sigma=args.smooth_sigma,
        smooth_pad_mode=args.smooth_pad_mode,
        uncertainty_samples=args.uncertainty_samples,
        uncertainty_ci=args.uncertainty_ci,
        uncertainty_concentration=args.uncertainty_concentration,
        score_col=args.score_col,
        fallback_to_maxprob=args.fallback_to_maxprob,
    )

    fmt = args.format.lower()
    if fmt == "csv":
        out_df = pd.DataFrame({"x": arc.x, "y": arc.y})
        if arc.y_low is not None and arc.y_high is not None:
            out_df["y_low"] = arc.y_low
            out_df["y_high"] = arc.y_high
        content = out_df.to_csv(index=False)
    elif fmt == "json":
        payload = {"x": arc.x, "y": arc.y, "raw_x": arc.raw_x, "raw_y": arc.raw_y}
        if arc.y_low is not None and arc.y_high is not None:
            payload["y_low"] = arc.y_low
            payload["y_high"] = arc.y_high
        content = json.dumps(payload, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    _write_text(args.output, content)

    if args.plot is not None:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise ImportError("Plotting requires matplotlib (pip install fabula[plot]).") from e

        plt.figure()
        plt.plot(arc.x, arc.y)
        plt.xlabel("Relative position")
        plt.ylabel("Score")
        plt.title("Fabula arc")

        if args.plot == "-":
            plt.show()
        else:
            plt.savefig(args.plot, bbox_inches="tight")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fabula", description="Transformers-based narrative arcs for literature.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("input", help="Input text file path, or '-' for stdin.")
        sp.add_argument("-o", "--output", default="-", help="Output path, or '-' for stdout. (default: '-')")
        sp.add_argument("--encoding", default="utf-8", help="Input file encoding. (default: utf-8)")

        sp.add_argument("--dummy", action="store_true", help="Use dummy scorer (no transformers download).")

        sp.add_argument("--analysis", choices=["sentiment", "emotion"], default="sentiment",
                        help="Analysis type (default: sentiment).")
        sp.add_argument("--model", default=None,
                        help="Hugging Face model id (ignored with --dummy). "
                             "Defaults to cmarkea/distilcamembert-base-sentiment for sentiment "
                             "and astrosbd for emotion.")
        sp.add_argument("--device", default=None, help="cpu, cuda, cuda:0 (default: auto).")
        sp.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.")
        sp.add_argument("--max-length", type=int, default=512, help="Max tokens per segment fed to the model.")
        sp.add_argument(
            "--temperature",
            type=float,
            default=1.0,
            help="Softmax temperature for probability calibration (default: 1.0).",
        )

        sp.add_argument("--segment", choices=["sentence", "paragraph", "window"], default="sentence",
                        help="Segmentation strategy (default: sentence).")
        sp.add_argument("--window-tokens", type=int, default=256, help="Token window size (segment=window).")
        sp.add_argument("--stride-tokens", type=int, default=64, help="Token stride (segment=window).")
        sp.add_argument("--min-tokens", type=int, default=16, help="Min tokens for window segments.")

    sp_score = sub.add_parser("score", help="Score segments and output per-segment data.")
    add_common(sp_score)
    sp_score.add_argument("--format", choices=["csv", "json", "jsonl"], default="csv", help="Output format.")
    sp_score.set_defaults(func=cmd_score)

    sp_arc = sub.add_parser("arc", help="Compute a smoothed narrative arc from segment scores.")
    add_common(sp_arc)
    sp_arc.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format.")
    sp_arc.add_argument("--n-points", type=int, default=100, help="Resample the arc to N points.")
    sp_arc.add_argument("--smooth-window", type=int, default=9, help="Smoothing window size.")
    sp_arc.add_argument(
        "--smooth-method",
        choices=["moving_average", "gaussian", "none"],
        default="moving_average",
        help="Smoothing method (default: moving_average).",
    )
    sp_arc.add_argument(
        "--smooth-sigma",
        type=float,
        default=None,
        help="Gaussian sigma (default: window/6). Only used with --smooth-method=gaussian.",
    )
    sp_arc.add_argument(
        "--smooth-pad-mode",
        choices=["reflect", "edge", "constant"],
        default="reflect",
        help="Padding mode for smoothing (default: reflect).",
    )
    sp_arc.add_argument(
        "--uncertainty-samples",
        type=int,
        default=0,
        help="Number of Monte Carlo samples for uncertainty bands (default: 0).",
    )
    sp_arc.add_argument(
        "--uncertainty-ci",
        type=float,
        default=0.95,
        help="Confidence interval for uncertainty bands (default: 0.95).",
    )
    sp_arc.add_argument(
        "--uncertainty-concentration",
        type=float,
        default=50.0,
        help="Dirichlet concentration for sampling from probabilities (default: 50.0).",
    )
    sp_arc.add_argument("--score-col", default="score", help="Column to use as scalar score.")
    sp_arc.add_argument("--no-fallback-to-maxprob", dest="fallback_to_maxprob", action="store_false",
                        help="Disable fallback scalar using max(prob) when score is missing.")
    sp_arc.add_argument("--plot", default=None,
                        help="Plot to a file (e.g., arc.png) or '-' to display interactively (requires matplotlib).")
    sp_arc.set_defaults(func=cmd_arc)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
