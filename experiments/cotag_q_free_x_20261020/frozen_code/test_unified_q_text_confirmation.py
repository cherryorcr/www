import torch

from run_unified_q_text_confirmation_20261008 import baseline_provenance, config, save_memory, load_memory


def test_candidate_changes_one_config_field():
    from dataclasses import asdict
    base, candidate = config(271108, 30)
    assert {k for k in asdict(base) if asdict(base)[k] != asdict(candidate)[k]} == {"text_materialization"}


def test_baseline_metadata_does_not_modify_features_and_compressed_archive_roundtrips(tmp_path):
    block = dict(x=torch.randn(3, 4), text_emb=torch.randn(3, 8), texts=["a", "b", "c"])
    result = baseline_provenance(block, dict(text_source_ids=[[1], [2, 3], [4]]))
    assert "text_source_ids" not in block
    assert result["x"] is block["x"] and result["text_emb"] is block["text_emb"]
    path = tmp_path / "memory.pt.gz"
    save_memory(path, result)
    restored = load_memory(path)
    torch.testing.assert_close(restored["x"], result["x"], rtol=0, atol=0)
    assert restored["text_source_ids"] == result["text_source_ids"]
