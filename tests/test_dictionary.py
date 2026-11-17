from ocsu_hybrid.fg.dictionary import dictionary_stats, get_func_groups


def test_dictionary_compiles():
    stats = dictionary_stats()
    assert stats["broken"] == []
    assert stats["compiled"] == 262


def test_aspirin_groups():
    groups = set(get_func_groups("CC(=O)Oc1ccccc1C(=O)O"))
    assert {"benzene", "esters", "carboxylic acids"} <= groups


def test_invalid_smiles():
    assert get_func_groups("not-a-smiles") == []
    assert get_func_groups(None) == []
