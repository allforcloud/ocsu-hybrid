"""Unit tests for multi-view candidate selection."""
from __future__ import annotations

from ocsu_hybrid.fg.dictionary import get_func_groups
from ocsu_hybrid.pipeline import Pipeline


def test_selects_view_with_matching_functional_groups():
    rows = [dict(
        image="x.png",
        smiles="CC(=O)O",
        raw="CC(=O)O",
        candidates=[
            dict(view="base", smiles="CC(=O)O", confidence=0.95),
            dict(view="pad0.08", smiles="CCO", confidence=0.80),
        ],
    )]
    groups = [get_func_groups("CCO")]
    selected = Pipeline.select_candidates(rows, groups)
    assert selected[0]["smiles"] == "CCO"
    assert selected[0]["selected_view"] == "pad0.08"


def test_keeps_single_view_result_when_no_candidates():
    rows = [dict(image="x.png", smiles="CCO", candidates=[])]
    selected = Pipeline.select_candidates(rows, [["alcohol groups"]])
    assert selected[0]["smiles"] == "CCO"
    assert "selected_view" not in selected[0]
