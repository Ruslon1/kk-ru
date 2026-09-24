import json
from pathlib import Path

from kk_ru.train import _load_resume_state


def test_resume_state_includes_position(tmp_path: Path):
    checkpoint = tmp_path / "step-12"
    checkpoint.mkdir()
    (checkpoint / "meta.json").write_text(
        json.dumps({"step": 12, "epoch": 2, "batch_index": 7}), encoding="utf-8"
    )

    assert _load_resume_state(str(checkpoint)) == {
        "step": 12,
        "epoch": 2,
        "batch_index": 7,
    }


def test_old_checkpoint_resumes_from_start_of_first_epoch(tmp_path: Path):
    checkpoint = tmp_path / "step-12"
    checkpoint.mkdir()
    (checkpoint / "meta.json").write_text(json.dumps({"step": 12}), encoding="utf-8")

    assert _load_resume_state(str(checkpoint)) == {
        "step": 12,
        "epoch": 0,
        "batch_index": 0,
    }
