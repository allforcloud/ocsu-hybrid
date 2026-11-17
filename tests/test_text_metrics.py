import pytest


def test_text_metrics_exact_match():
    pytest.importorskip("rouge_score")
    pytest.importorskip("nltk")
    from ocsu_hybrid.evaluate.text import TextScorer

    try:
        scorer = TextScorer()
    except Exception as exc:  # missing NLTK corpora / SciBERT assets
        pytest.skip(str(exc))
    metrics = scorer.evaluate([dict(gt="The molecule is ethanol.",
                                    pred="The molecule is ethanol.")])
    assert metrics["BLEU-4"] == pytest.approx(1.0)
    assert metrics["ROUGE-L"] == pytest.approx(1.0)
