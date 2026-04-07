from __future__ import annotations

import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.lib import layout


def git(
    root: Path,
    *args: str,
    capture_output: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def working_tree_dirty(root: Path) -> bool:
    result = git(root, "status", "--porcelain")
    return bool(result.stdout.strip())


def current_branch(root: Path) -> str:
    return git(root, "branch", "--show-current").stdout.strip()


def branch_exists(root: Path, branch: str) -> bool:
    result = git(root, "show-ref", "--verify", f"refs/heads/{branch}", check=False)
    return result.returncode == 0


def commit_changes(root: Path, message: str | None = None) -> None:
    """Stage all changes and commit if the working tree is dirty."""
    root_path = root if isinstance(root, Path) else Path(root)
    if not working_tree_dirty(root_path):
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    commit_message = message or f"Auto-commit results [{timestamp}]"
    git(root_path, "add", "-A")
    git(root_path, "commit", "-m", commit_message, capture_output=False)


def deploy_dashboard(root: Path | None = None, allow_dirty: bool = False, push: bool = True) -> None:
    root_path = layout.repo_root(root)
    source_path = layout.website_index_path(root_path)
    if not source_path.exists():
        raise SystemExit(f"Dashboard file not found: {source_path}")
    if working_tree_dirty(root_path) and not allow_dirty:
        raise SystemExit(
            "Refusing to deploy with a dirty git tree. "
            "Commit or stash changes first, or rerun with --allow-dirty."
        )

    html = source_path.read_text(encoding="utf-8")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with tempfile.TemporaryDirectory(prefix="deploy-gh-pages-", dir=root_path.parent) as tmpdir:
        worktree_path = Path(tmpdir)
        if branch_exists(root_path, "gh-pages"):
            git(root_path, "worktree", "add", str(worktree_path), "gh-pages")
        else:
            git(root_path, "worktree", "add", "-b", "gh-pages", str(worktree_path))
        try:
            target_path = worktree_path / "index.html"
            target_path.write_text(html, encoding="utf-8")
            git(worktree_path, "add", "index.html")
            diff = git(worktree_path, "diff", "--cached", "--quiet", capture_output=False, check=False)
            if diff.returncode == 1:
                git(worktree_path, "commit", "-m", f"Auto-deploy website [{timestamp}]")
                if push:
                    git(worktree_path, "push", "--force", "origin", "HEAD:gh-pages", capture_output=False)
                print("Dashboard deployed to gh-pages.")
            elif diff.returncode == 0:
                print("Dashboard already up to date on gh-pages.")
            else:
                raise subprocess.CalledProcessError(diff.returncode, diff.args)
        finally:
            git(root_path, "worktree", "remove", "--force", str(worktree_path), capture_output=False)
