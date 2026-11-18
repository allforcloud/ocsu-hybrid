import json

from ocsu_hybrid.iupac.pubchem import PubChemResolver


def test_cache_hit_does_not_touch_network(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(dict(smiles="CCO", name="ethanol",
                                     source="smiles")) + "\n", encoding="utf-8")
    resolver = PubChemResolver(cache, sleep_s=0)
    name, source = resolver.resolve("OCC")  # canonicalises to CCO
    assert (name, source) == ("ethanol", "smiles")


def test_empty_input(tmp_path):
    resolver = PubChemResolver(tmp_path / "cache.jsonl", sleep_s=0)
    assert resolver.resolve("") == (None, None)
    assert resolver.resolve("not-a-smiles") == (None, None)


def test_cached_miss_is_retried(tmp_path, monkeypatch):
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(dict(smiles="CCO", name=None,
                                     source=None)) + "\n", encoding="utf-8")
    resolver = PubChemResolver(cache, sleep_s=0)
    monkeypatch.setattr(resolver, "_iupac_from_props",
                        lambda section, key: "ethanol")
    assert resolver.resolve("CCO") == ("ethanol", "smiles")
