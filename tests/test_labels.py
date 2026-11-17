from ocsu_hybrid.fg.labels import mean_f1, parse_fg


def test_parse_single_group():
    assert parse_fg("The functional group ester is highlighted.") == ["ester"]


def test_parse_multiple_groups():
    text = ("This molecule contains several functional groups, including esters, "
            "ketones, carbonyl methyl ester, methoxy, side-chain aldehydes, alkenes, "
            "cyclopentane, and cyclohexane.")
    assert parse_fg(text) == [
        "esters", "ketones", "carbonyl methyl ester", "methoxy",
        "side-chain aldehydes", "alkenes", "cyclopentane", "cyclohexane",
    ]


def test_mean_f1():
    metrics = mean_f1([["a", "b"], ["c"]], [["a"], ["c", "d"]])
    assert metrics["f1"] == (2 / 3 + 2 / 3) / 2
    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["exact"] == 0.0
