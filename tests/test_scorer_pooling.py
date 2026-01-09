import pytest

from fabula.scorer import TransformersScorer


def _make_scorer(pooling: str) -> TransformersScorer:
    scorer = TransformersScorer.__new__(TransformersScorer)
    scorer.pooling = pooling
    return scorer


def test_pool_chunk_probs_mean():
    scorer = _make_scorer("mean")
    probs = [[0.2, 0.8], [0.6, 0.4]]
    pooled = scorer._pool_chunk_probs(probs)
    assert pooled == pytest.approx([0.4, 0.6])


def test_pool_chunk_probs_max():
    scorer = _make_scorer("max")
    probs = [[0.2, 0.8], [0.6, 0.4]]
    pooled = scorer._pool_chunk_probs(probs)
    assert pooled == [0.6, 0.8]


def test_pool_chunk_probs_attention():
    scorer = _make_scorer("attention")
    probs = [[0.9, 0.1], [0.2, 0.8]]
    pooled = scorer._pool_chunk_probs(probs)
    assert pooled[0] > pooled[1]


def test_pool_chunk_probs_none():
    scorer = _make_scorer("none")
    probs = [[0.2, 0.8], [0.6, 0.4]]
    pooled = scorer._pool_chunk_probs(probs)
    assert pooled == [0.2, 0.8]
