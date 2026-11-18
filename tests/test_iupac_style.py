from ocsu_hybrid.iupac.style import gated_name, normalize_name


def test_heteroaryl_locants():
    assert normalize_name(
        "6-methoxy-2-pyridinyl") == "6-methoxypyridin-2-yl"
    assert normalize_name(
        "3-(4-methyl-2-pyridinyl)pyridine") == "3-(4-methylpyridin-2-yl)pyridine"


def test_salt_separators():
    assert normalize_name("sodium 2-amino-benzoate") == "sodium;2-amino-benzoate"
    assert normalize_name("lithium bromide") == "lithium;bromide"
    assert normalize_name("ethyl(trimethyl)azanium iodide") == \
        "ethyl(trimethyl)azanium;iodide"


def test_cis_trans_and_isotope():
    assert normalize_name("trans-(1R,3R)-cyclopropane") == "(1R,3R)-cyclopropane"
    assert normalize_name("2-[3-(18F)fluoro-phenyl]") == \
        "2-[3-(18F)fluoranyl-phenyl]"


def test_only_accepts_pubchem_synonym():
    original = "6-methoxy-2-pyridinyl"
    candidate = "6-methoxypyridin-2-yl"
    assert gated_name(original, [candidate]) == candidate
    assert gated_name(original, ["something else"]) == original
