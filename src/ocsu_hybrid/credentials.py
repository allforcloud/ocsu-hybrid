"""Local-only credentials for publishing the model and the repository.

The real file is ``configs/local.yaml``. It is gitignored and should be mode
``0600``. Environment variables ``HF_TOKEN`` and ``GITHUB_TOKEN`` override the
YAML values, which is the safer option on shared machines.

Do not put account passwords in this file. Use a Hugging Face write token and a
GitHub personal access token (PAT); both can be revoked independently.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import paths

CONFIG_PATH = paths.PROJECT_ROOT / "configs" / "local.yaml"

HF_TOKEN_PREFIXES = ("hf_",)
GITHUB_TOKEN_PREFIXES = ("ghp_", "github_pat_", "gho_", "ghu_", "ghs_", "ghr_")


def load_local(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the local YAML file, returning an empty mapping when absent."""
    config_path = Path(path) if path else CONFIG_PATH
    if not config_path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read configs/local.yaml. "
            "Install it with `pip install PyYAML`."
        ) from exc
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return data


def normalize_repo(value: Optional[str], kind: str) -> Optional[str]:
    """Accept a repository id or a browser/SSH URL and return ``owner/name``."""
    if not value:
        return value
    repo = value.strip()
    prefixes = {
        "huggingface": (
            "https://huggingface.co/", "http://huggingface.co/", "hf://",
        ),
        "github": (
            "https://github.com/", "http://github.com/",
            "ssh://git@github.com/", "git@github.com:",
        ),
    }[kind]
    for prefix in prefixes:
        if repo.startswith(prefix):
            repo = repo[len(prefix):]
            break
    if repo.endswith(".git"):
        repo = repo[:-4]
    return repo.strip("/") or None


def _cached_hf_token() -> Optional[str]:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def hf_credentials(path: str | Path | None = None
                   ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(username, token, fg_repo)`` with env and cached-login fallback."""
    section = load_local(path).get("huggingface") or {}
    token = (os.environ.get("HF_TOKEN") or section.get("token")
             or _cached_hf_token() or None)
    return (section.get("username") or None, token,
            normalize_repo(section.get("fg_repo"), "huggingface"))


def github_credentials(path: str | Path | None = None) -> Dict[str, Any]:
    """Return GitHub settings.

    ``auth`` defaults to ``ssh`` (an existing key/agent is enough to push).
    ``token`` is only needed for API repository creation or HTTPS auth.
    """
    section = load_local(path).get("github") or {}
    token = os.environ.get("GITHUB_TOKEN") or section.get("token") or None
    auth = str(section.get("auth") or "ssh").lower()
    if auth not in ("ssh", "token"):
        raise ValueError("github.auth must be 'ssh' or 'token'")
    return dict(
        username=section.get("username") or None,
        auth=auth,
        token=token,
        repo=normalize_repo(section.get("repo"), "github") or "ocsu-hybrid",
        branch=section.get("branch") or "main",
        private=bool(section.get("private", False)),
        create_repo=str(section.get("create_repo") or "auto").lower(),
    )


def validate_token(token: Optional[str], kind: str) -> str:
    """Reject empty values and values that look like account passwords."""
    if not token:
        raise ValueError(
            f"Missing {kind} token. Fill configs/local.yaml or set "
            f"{'HF_TOKEN' if kind == 'huggingface' else 'GITHUB_TOKEN'}."
        )
    prefixes = HF_TOKEN_PREFIXES if kind == "huggingface" else GITHUB_TOKEN_PREFIXES
    if not token.startswith(prefixes):
        raise ValueError(
            f"The {kind} value does not look like an access token. "
            "Use a token, not an account password. "
            f"Expected prefixes: {', '.join(prefixes)}"
        )
    return token


def mask(value: Optional[str]) -> str:
    """Safe representation for logs."""
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"
