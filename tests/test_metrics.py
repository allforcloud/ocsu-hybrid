import pytest

from ocsu_hybrid.evaluate.smiles import canonicalize, score_smiles, tanimoto


def test_canonicalize_ignores_cistrans():
    assert canonicalize("C/C=C/C", ignore_cistrans=True)[0] == "CC=CC"


def test_canonicalize_invalid():
    value, ok = canonicalize("not-a-smiles")
    assert not ok
    assert value == "not-a-smiles"


def test_tanimoto_identical():
    assert tanimoto("CCO", "OCC") == pytest.approx(1.0)


def test_score_smiles():
    records = [dict(images=["a.png"], gt="CCO"), dict(images=["b.png"], gt="CCC")]
    preds = {"a.png": "OCC", "b.png": "CCN"}
    metrics, rows = score_smiles(records, preds)
    assert metrics["n"] == 2
    assert metrics["acc_s"] == 0.5
    assert metrics["acc_g"] == 0.5
    assert rows[0]["exact"] is False
