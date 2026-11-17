import pytest

from ocsu_hybrid.credentials import (
    github_credentials,
    hf_credentials,
    load_local,
    mask,
    normalize_repo,
    validate_token,
)


def test_load_local(tmp_path):
    path = tmp_path / "local.yaml"
    path.write_text(
        "huggingface:\n  username: alice\n  token: hf_test\n"
        "github:\n  repo: alice/ocsu-hybrid\n  private: true\n",
        encoding="utf-8",
    )
    data = load_local(path)
    assert data["huggingface"]["username"] == "alice"
    assert data["github"]["private"] is True


def test_hf_env_override(tmp_path, monkeypatch):
    path = tmp_path / "local.yaml"
    path.write_text("huggingface:\n  token: hf_file\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert hf_credentials(path)[1] == "hf_env"


def test_github_defaults(tmp_path):
    cfg = github_credentials(tmp_path / "missing.yaml")
    assert cfg["repo"] == "ocsu-hybrid"
    assert cfg["branch"] == "main"
    assert cfg["private"] is False
    assert cfg["auth"] == "ssh"
    assert cfg["create_repo"] == "auto"


def test_normalize_repo_urls():
    assert normalize_repo(
        "https://huggingface.co/allforcloud9527/ocsu-hybrid-fg-convnext-t",
        "huggingface") == "allforcloud9527/ocsu-hybrid-fg-convnext-t"
    assert normalize_repo(
        "https://github.com/allforcloud/ocsu-hybrid.git",
        "github") == "allforcloud/ocsu-hybrid"
    assert normalize_repo(
        "git@github.com:allforcloud/ocsu-hybrid.git",
        "github") == "allforcloud/ocsu-hybrid"


@pytest.mark.parametrize("token", ["hf_abc", "github_pat_abc", "ghp_abc"])
def test_validate_token_accepts_tokens(token):
    kind = "huggingface" if token.startswith("hf_") else "github"
    assert validate_token(token, kind) == token


def test_validate_token_rejects_password():
    with pytest.raises(ValueError, match="not an account password"):
        validate_token("my-secret-password", "github")


def test_validate_token_requires_value():
    with pytest.raises(ValueError, match="Missing"):
        validate_token("", "huggingface")


def test_mask_does_not_reveal_secret():
    value = "hf_abcdefghijklmnop"
    masked = mask(value)
    assert masked.startswith("hf_a")
    assert masked.endswith("mnop")
    assert value not in masked
