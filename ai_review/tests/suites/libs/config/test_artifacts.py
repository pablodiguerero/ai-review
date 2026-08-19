from pathlib import Path

from ai_review.libs.config.artifacts import ArtifactsConfig


def test_artifacts_config_defaults() -> None:
    config = ArtifactsConfig()
    assert config.llm_dir == Path("./artifacts/llm")
    assert config.vcs_dir == Path("./artifacts/vcs")
    assert config.llm_enabled is False
    assert config.vcs_enabled is False


def test_artifacts_config_does_not_create_directories(tmp_path: Path) -> None:
    llm_dir = tmp_path / "nested" / "llm"
    vcs_dir = tmp_path / "nested" / "vcs"

    config = ArtifactsConfig(llm_dir=llm_dir, vcs_dir=vcs_dir)

    assert config.llm_dir == llm_dir
    assert config.vcs_dir == vcs_dir
    assert not llm_dir.exists()
    assert not vcs_dir.exists()
    assert not (tmp_path / "nested").exists()
