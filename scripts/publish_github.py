#!/usr/bin/env python3
"""Create a GitHub repository and push this project.

Reads ``configs/local.yaml`` (gitignored). The default ``github.auth: ssh``
pushes with the local SSH key and never puts a token in a URL. A GitHub token is
only used for API repository creation or ``auth: token``.

    python scripts/publish_github.py --dry-run
    python scripts/publish_github.py

The script never force-pushes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsu_hybrid.credentials import (  # noqa: E402
    github_credentials,
    mask,
    normalize_repo,
    validate_token,
)

API = "https://api.github.com"


def _sanitize(text: str, token: Optional[str]) -> str:
    return text.replace(token, "***") if token else text


def _env() -> dict:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _api(method: str, path: str, token: str,
         payload: Optional[dict] = None) -> Optional[dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        API + path, data=data, method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "ocsu-hybrid",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        detail = _sanitize(exc.read().decode("utf-8", errors="replace")[:800], token)
        raise SystemExit(f"GitHub API {method} {path} failed ({exc.code}): {detail}") from exc


def _run(args: list[str], description: str, token: Optional[str] = None) -> str:
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          check=False, env=_env())
    if proc.returncode != 0:
        stderr = _sanitize((proc.stderr or proc.stdout or "").strip()[-2000:], token)
        raise SystemExit(f"{description} failed:\n{stderr}")
    return (proc.stdout or "").strip()


def _git_config(key: str) -> str:
    proc = subprocess.run(["git", "config", key], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    return (proc.stdout or "").strip()


def _ensure_identity() -> list[str]:
    """Return git -c flags when the user has no global identity configured."""
    flags = []
    if not _git_config("user.name"):
        flags += ["-c", "user.name=OCSU-Hybrid"]
    if not _git_config("user.email"):
        flags += ["-c", "user.email=ocsu-hybrid@users.noreply.github.com"]
    return flags


def _resolve_repo(repo: str, login: Optional[str]) -> tuple[str, str]:
    value = normalize_repo(repo, "github") or "ocsu-hybrid"
    if "/" in value:
        owner, name = value.split("/", 1)
    elif login:
        owner, name = login, value
    else:
        raise SystemExit(
            "A bare repository name needs github.username or a GitHub token. "
            "Set `github.repo: owner/name` in configs/local.yaml."
        )
    if not owner or not name:
        raise SystemExit(f"Invalid GitHub repository: {repo!r}")
    return owner, name


def _credentials_ignored() -> bool:
    """Check .gitignore without requiring an initialized repository."""
    if (ROOT / ".git").exists():
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "configs/local.yaml"], cwd=ROOT,
            capture_output=True, check=False)
        return proc.returncode == 0
    ignore_file = ROOT / ".gitignore"
    if not ignore_file.exists():
        return False
    lines = {line.strip().lstrip("/") for line in
             ignore_file.read_text(encoding="utf-8").splitlines()}
    return "configs/local.yaml" in lines


def _repo_exists_ssh(owner: str, name: str) -> bool:
    url = f"git@github.com:{owner}/{name}.git"
    proc = subprocess.run(["git", "ls-remote", url], cwd=ROOT,
                          capture_output=True, text=True, check=False, env=_env())
    if proc.returncode == 0:
        return True
    message = (proc.stderr or proc.stdout or "").strip()
    if "Repository not found" in message or "does not exist" in message:
        return False
    raise SystemExit(
        f"Cannot access {url}. Check the SSH key with `ssh -T git@github.com`.\n"
        f"{message[-1500:]}"
    )


def _gh_ready() -> bool:
    if not shutil.which("gh"):
        return False
    proc = subprocess.run(["gh", "auth", "status"], cwd=ROOT,
                          capture_output=True, check=False)
    return proc.returncode == 0


def _create_with_gh(owner: str, name: str, private: bool) -> None:
    visibility = "--private" if private else "--public"
    _run(["gh", "repo", "create", f"{owner}/{name}", visibility,
          "--source", str(ROOT), "--remote", "origin"],
         "gh repo create")
    print(f"[github] created with gh: https://github.com/{owner}/{name}")


def _create_with_api(owner: str, name: str, login: str, private: bool,
                     token: str) -> None:
    payload = dict(
        name=name,
        private=private,
        description="Modular OCSU pipeline: MolScribe + ConvNeXt-T + PubChem + BioT5+",
        auto_init=False,
    )
    if owner == login:
        created = _api("POST", "/user/repos", token, payload)
    else:
        created = _api("POST", f"/orgs/{owner}/repos", token, payload)
    if not created:
        raise SystemExit(f"Failed to create repository {owner}/{name}")
    print(f"[github] created with API: {created['html_url']} "
          f"({'private' if private else 'public'})")


def _ensure_repo(owner: str, name: str, login: Optional[str], private: bool,
                 cfg: dict, token: Optional[str]) -> None:
    if cfg["auth"] == "ssh":
        try:
            exists = _repo_exists_ssh(owner, name)
        except SystemExit:
            raise
        if exists:
            print(f"[github] repository exists: https://github.com/{owner}/{name}")
            return
    else:
        existing = _api("GET", f"/repos/{owner}/{name}", token)
        if existing:
            print(f"[github] repository exists: {existing['html_url']}")
            if bool(existing.get("private")) != private:
                print("[github] note: existing repository visibility differs from config")
            return

    mode = cfg["create_repo"]
    if mode not in ("auto", "gh", "api", "never"):
        raise SystemExit("github.create_repo must be auto, gh, api or never")
    if mode == "never":
        raise SystemExit(
            f"Repository https://github.com/{owner}/{name} does not exist. "
            "Create it manually, then rerun this script."
        )
    if mode in ("auto", "gh") and _gh_ready():
        _create_with_gh(owner, name, private)
        return
    if mode in ("auto", "api") and token:
        if not login:
            who = _api("GET", "/user", token)
            login = who.get("login") if who else None
        if not login:
            raise SystemExit("Could not determine the GitHub account from the token")
        _create_with_api(owner, name, login, private, token)
        return
    raise SystemExit(
        f"Repository https://github.com/{owner}/{name} does not exist and no "
        "creation method is available. Install/authenticate `gh`, set a GitHub "
        "token, or create the empty repository manually."
    )


def _set_remote(owner: str, name: str, auth: str) -> None:
    remote = (f"git@github.com:{owner}/{name}.git" if auth == "ssh"
              else f"https://github.com/{owner}/{name}.git")
    if subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT,
                      capture_output=True, check=False).returncode == 0:
        _run(["git", "remote", "set-url", "origin", remote], "git remote set-url")
    else:
        _run(["git", "remote", "add", "origin", remote], "git remote add origin")


def _push(owner: str, name: str, branch: str, cfg: dict,
          token: Optional[str]) -> None:
    if cfg["auth"] == "ssh":
        _run(["git", "push", "-u", "origin", f"{branch}:{branch}"],
             "git push over SSH")
        return
    push_url = f"https://x-access-token:{token}@github.com/{owner}/{name}.git"
    proc = subprocess.run(["git", "push", push_url, f"{branch}:{branch}"],
                          cwd=ROOT, capture_output=True, text=True,
                          check=False, env=_env())
    if proc.returncode != 0:
        detail = _sanitize((proc.stderr or proc.stdout or "").strip()[-2500:], token)
        raise SystemExit("git push failed (no force-push attempted):\n" + detail)
    _run(["git", "config", f"branch.{branch}.remote", "origin"],
         "git config branch.remote")
    _run(["git", "config", f"branch.{branch}.merge", f"refs/heads/{branch}"],
         "git config branch.merge")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="default: configs/local.yaml")
    ap.add_argument("--repo", default=None, help="override github.repo")
    ap.add_argument("--branch", default=None, help="override github.branch")
    ap.add_argument("--auth", choices=["ssh", "token"], default=None)
    ap.add_argument("--private", action="store_true",
                    help="create a private repository (default: public)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not shutil.which("git"):
        raise SystemExit("git is not installed")
    if not (ROOT / "README.md").exists():
        raise SystemExit(f"{ROOT} does not look like the project root")

    cfg = github_credentials(args.config)
    cfg["auth"] = args.auth or cfg["auth"]
    repo_value = args.repo or cfg["repo"]
    branch = args.branch or cfg["branch"]
    private = args.private or cfg["private"]

    token = cfg["token"] if cfg["auth"] == "token" else None
    login = cfg["username"]
    if cfg["auth"] == "token":
        try:
            token = validate_token(token, "github")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        who = _api("GET", "/user", token)
        login = who.get("login") if who else None
        if not login:
            raise SystemExit("Could not authenticate with GitHub")
        if cfg["username"] and cfg["username"] != login:
            print(f"[github] warning: config username {cfg['username']} != "
                  f"token owner {login}; using the token owner")
    else:
        token = cfg["token"] or None  # optional, only for API repo creation

    owner, name = _resolve_repo(repo_value, login)
    print(f"[github] repository : https://github.com/{owner}/{name}")
    print(f"[github] owner/name : {owner}/{name}")
    print(f"[github] branch     : {branch}")
    print(f"[github] visibility : {'private' if private else 'public'}")
    print(f"[github] auth       : {cfg['auth']}")
    print(f"[github] token      : {mask(token)}")

    if not _credentials_ignored():
        raise SystemExit(
            "configs/local.yaml is not ignored by git. Add "
            "`configs/local.yaml` to .gitignore before publishing."
        )
    if (ROOT / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "configs/local.yaml"],
            cwd=ROOT, capture_output=True, check=False)
        if tracked.returncode == 0:
            raise SystemExit(
                "configs/local.yaml is already tracked. Remove it from git history "
                "before publishing: git rm --cached configs/local.yaml"
            )

    if args.dry_run:
        print("[github] dry run: no repository created, nothing committed or pushed")
        return 0

    if not (ROOT / ".git").exists():
        _run(["git", "init"], "git init")

    _ensure_repo(owner, name, login, private, cfg, token)

    _run(["git", "add", "-A"], "git add -A")
    staged = _run(["git", "diff", "--cached", "--name-only"],
                  "git diff --cached").splitlines()
    if "configs/local.yaml" in staged:
        _run(["git", "reset", "--", "configs/local.yaml"],
             "git reset configs/local.yaml")
        raise SystemExit("Refusing to commit configs/local.yaml; it was staged")

    has_head = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT,
        capture_output=True, check=False).returncode == 0
    identity = _ensure_identity()
    if staged:
        message = "Update OCSU-Hybrid" if has_head else "Initial commit: OCSU-Hybrid"
        _run(["git"] + identity + ["commit", "-m", message], "git commit")
        print(f"[github] committed: {message}")
    elif not has_head:
        _run(["git"] + identity + ["commit", "--allow-empty",
                                   "-m", "Initial commit: OCSU-Hybrid"],
             "git commit --allow-empty")
        print("[github] created an empty initial commit")

    _run(["git", "branch", "-M", branch], f"git branch -M {branch}")
    _set_remote(owner, name, cfg["auth"])
    _push(owner, name, branch, cfg, token)
    print(f"[github] pushed: https://github.com/{owner}/{name}")
    print("[github] the credential file was not committed")
    print("[github] if you forked the project, update the repository URLs in "
          "README.md and pyproject.toml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
