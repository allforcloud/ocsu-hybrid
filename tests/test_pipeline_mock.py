from ocsu_hybrid.pipeline import Pipeline


class DummyOCR:
    def recognize_batch(self, images):
        return [dict(image=str(image), smiles="CC(=O)Oc1ccccc1C(=O)O",
                     raw="CC(=O)Oc1ccccc1C(=O)O", error=None, latency_s=0.01)
                for image in images]


def test_pipeline_without_network(tmp_path):
    pipeline = Pipeline(fg_backend="dictionary", use_iupac=False,
                        use_description=False)
    pipeline._ocr = DummyOCR()  # avoid loading MolScribe in a unit test
    result = pipeline.analyze("examples/aspirin.png")
    assert result["valid"] is True
    assert "benzene" in result["functional_groups"]
    assert result["iupac"]["status"] == "skipped"
    assert result["description"]["status"] == "skipped"
    assert result["timings_s"]["ocsr"] >= 0.0
