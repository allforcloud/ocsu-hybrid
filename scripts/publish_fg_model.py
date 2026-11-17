#!/usr/bin/env python3
"""Publish the ConvNeXt-T functional-group checkpoint to Hugging Face.

Reads ``configs/local.yaml`` (gitignored), creates the model repository if it
does not exist, and uploads ``model.pt`` plus the model card.

    python scripts/publish_fg_model.py --dry-run
    python scripts/publish_fg_model.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsu_hybrid import paths  # noqa: E402
from ocsu_hybrid.credentials import (  # noqa: E402
    hf_credentials,
    mask,
    normalize_repo,
    validate_token,
)

DEFAULT_REPO_NAME = "ocsu-hybrid-fg-convnext-t"


def resolve_repo(repo: str | None, username: str | None) -> str:
    if repo:
        repo_id = normalize_repo(repo, "huggingface") or ""
    elif username:
        repo_id = f"{username}/{DEFAULT_REPO_NAME}"
    else:
        raise SystemExit(
            "No Hugging Face repository configured. Set `huggingface.fg_repo` "
            "or `huggingface.username` in configs/local.yaml, or pass --repo."
        )
    if "/" not in repo_id:
        if not username:
            raise SystemExit("A bare repository name needs huggingface.username")
        repo_id = f"{username}/{repo_id}"
    if repo_id.startswith("your-org/"):
        raise SystemExit(
            "The repository is still the placeholder `your-org/...`. "
            "Set your real Hugging Face username or fg_repo in configs/local.yaml."
        )
    return repo_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="default: configs/local.yaml")
    ap.add_argument("--repo", default=None, help="override huggingface.fg_repo")
    ap.add_argument("--model", default=None, help="default: models/fg_classifier/model.pt")
    ap.add_argument("--card", default=str(ROOT / "docs" / "fg_model_card.md"))
    ap.add_argument("--private", action="store_true",
                    help="create a private HF repository (default: public)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    username, token, configured_repo = hf_credentials(args.config)
    repo_id = resolve_repo(args.repo or configured_repo, username)
    model_path = Path(args.model) if args.model else paths.fg_ckpt()
    card_path = Path(args.card)

    if not model_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {model_path}\n"
            "Train it with scripts/train_fg_classifier.py, download it with "
            "scripts/download_models.py --fg, or pass --model /path/to/model.pt."
        )
    if not card_path.exists():
        raise SystemExit(f"Model card not found: {card_path}")

    print(f"[hf] repository : {repo_id}")
    print(f"[hf] checkpoint : {model_path} "
          f"({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[hf] model card : {card_path}")
    print(f"[hf] token      : {mask(token)}")
    print(f"[hf] visibility : {'private' if args.private else 'public'}")

    if args.dry_run:
        print("[hf] dry run: no repository created and nothing uploaded")
        return 0

    try:
        token = validate_token(token, "huggingface")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("huggingface_hub is missing: pip install -e '.[all]'") from exc

    api = HfApi(token=token)
    try:
        who = api.whoami(token=token)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Hugging Face authentication failed: {exc}") from exc
    print(f"[hf] authenticated as {who.get('name') or who.get('user')}")

    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=args.private, exist_ok=True, token=token)
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo="model.pt",
                    repo_id=repo_id, token=token,
                    commit_message="Upload OCSU-Hybrid FG ConvNeXt-T checkpoint")
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                    repo_id=repo_id, token=token,
                    commit_message="Add model card")
    print(f"[hf] done: https://huggingface.co/{repo_id}")
    print("[hf] update README.md if you want the public repository id in the docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
