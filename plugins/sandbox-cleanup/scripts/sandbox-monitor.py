#!/usr/bin/env python3
"""Sandbox artifact cleanup for Claude Code.

PreToolUse hook that monitors bwrap sandbox processes and removes the 0-byte
placeholder files Claude creates for --ro-bind /dev/null mounts.

Process model:
  Hook runs as: claude → /bin/sh -c '<hook>' → python3 sandbox-monitor.py
  Shell exec-optimizes, so our parent is claude directly.

  Claude spawns bwrap via: claude → zsh -c -l 'bwrap ...'
  The zsh exec-optimizes into bwrap, so the child transitions:
    comm=.claude-wrapped → comm=zsh → comm=bwrap (within ~3ms)

  The bwrap cmdline (as a single arg to zsh -c) contains the full command
  including: eval '<command>' — which we match on to identify our bwrap.

  For fast commands (e.g. ls), the entire bwrap lifecycle is ~16ms.
  We tight-poll (no sleep) to catch it before it exits.

Algorithm:
  1. Read hook input, find claude PID (our parent), snapshot existing children
  2. Double-fork a daemon so the hook returns immediately
  3. Tight-poll claude's children for one whose cmdline contains our command
  4. Parse the matched cmdline to extract --ro-bind /dev/null <path> artifacts
  5. Wait for the bwrap process to exit
  6. Remove any artifacts that are still 0-byte files
"""

import json
import logging
import logging.handlers
import os
import shlex
import stat
import sys
import time
from pathlib import Path

FIND_CHILD_TIMEOUT_S = 5
FIND_CHILD_POLL_S = 0
EXIT_POLL_S = 0.1

LOG_PATH = Path(__file__).resolve().parent / "sandbox-monitor.log"

logger = logging.getLogger("sandbox-monitor")


def setup_logging() -> None:
    handler = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=1)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


# -- /proc helpers --


def get_ppid(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        after_comm = text[text.rfind(")") + 2 :]
        return int(after_comm.split()[1])
    except (OSError, ValueError, IndexError):
        return None


def get_children(pid: int) -> set[int]:
    try:
        text = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
        return {int(p) for p in text.split()} if text else set()
    except (OSError, ValueError):
        return set()


def get_cmdline(pid: int) -> list[str]:
    try:
        data = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [a.decode("utf-8", errors="replace") for a in data.split(b"\0") if a]
    except OSError:
        return []


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


# -- Settings --


def is_sandbox_enabled() -> bool:
    """Check if sandbox is enabled by merging Claude settings files.

    Reads in order (later overrides earlier):
      $HOME/.claude/settings.json
      $HOME/.claude/settings.local.json
      $CLAUDE_PROJECT_DIR/.claude/settings.json
      $CLAUDE_PROJECT_DIR/.claude/settings.local.json
    """
    home = Path.home() / ".claude"
    project = Path(d) / ".claude" if (d := os.environ.get("CLAUDE_PROJECT_DIR")) else None

    paths = [home / "settings.json", home / "settings.local.json"]
    if project:
        paths += [project / "settings.json", project / "settings.local.json"]

    enabled = False
    for p in paths:
        try:
            data = json.loads(p.read_text())
            if "sandbox" in data and "enabled" in data["sandbox"]:
                enabled = data["sandbox"]["enabled"]
        except (OSError, json.JSONDecodeError, TypeError):
            continue

    return enabled


# -- Core logic --


def find_bwrap_child(
    claude_pid: int, command: str, existing: set[int]
) -> tuple[int, list[str]] | None:
    """Tight-poll claude's children for one whose cmdline contains our command.

    Returns (pid, cmdline) or None on timeout.
    The cmdline is captured from the zsh wrapper phase: [zsh, -c, -l, <bwrap ...>].
    We match on `command` appearing anywhere in the joined cmdline.
    """
    deadline = time.monotonic() + FIND_CHILD_TIMEOUT_S

    while time.monotonic() < deadline:
        for pid in get_children(claude_pid) - existing:
            cmdline = get_cmdline(pid)
            if not cmdline:
                continue
            if command in " ".join(cmdline):
                return pid, cmdline

        time.sleep(FIND_CHILD_POLL_S)

    return None


def parse_artifacts(cmdline: list[str]) -> list[Path]:
    """Extract /dev/null bind-mount paths from a bwrap cmdline.

    The cmdline is [zsh, -c, -l, <bwrap command string>].
    We shlex.split the bwrap string and scan for --ro-bind /dev/null <path>.
    """
    if len(cmdline) < 4:
        return []

    try:
        args = shlex.split(cmdline[3])
    except ValueError:
        return []

    artifacts = []
    i = 0
    while i < len(args) - 2:
        if args[i] == "--ro-bind" and args[i + 1] == "/dev/null":
            artifacts.append(Path(args[i + 2]))
            i += 3
        else:
            i += 1
    return artifacts


def cleanup_artifacts(artifacts: list[Path]) -> int:
    """Remove 0-byte regular files. Returns count of files removed."""
    removed = 0
    for p in artifacts:
        try:
            st = p.lstat()
            if stat.S_ISREG(st.st_mode) and st.st_size == 0:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def daemonize() -> bool:
    """Double-fork to detach. Returns True in the daemon, False in the parent."""
    if os.fork() > 0:
        return False
    os.setsid()
    if os.fork() > 0:
        os._exit(0)

    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    if devnull > 2:
        os.close(devnull)

    return True


def monitor(claude_pid: int, command: str, existing: set[int]) -> None:
    logger.info("monitor: claude_pid=%d command=%r", claude_pid, command)

    result = find_bwrap_child(claude_pid, command, existing)
    if result is None:
        logger.warning("timeout: bwrap child not found")
        return

    bwrap_pid, cmdline = result
    logger.info("found bwrap pid=%d", bwrap_pid)

    artifacts = parse_artifacts(cmdline)
    if not artifacts:
        logger.info("no artifacts to track")
        return

    logger.info("tracking %d artifacts", len(artifacts))

    while pid_alive(bwrap_pid):
        time.sleep(EXIT_POLL_S)

    logger.info("bwrap %d exited", bwrap_pid)

    removed = cleanup_artifacts(artifacts)
    logger.info("cleanup: removed %d/%d artifacts", removed, len(artifacts))


def main() -> None:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    if not is_sandbox_enabled():
        sys.exit(0)

    command = hook_input.get("tool_input", {}).get("command", "")
    if not command:
        sys.exit(0)

    ppid = get_ppid(os.getpid())
    if ppid is None or ppid <= 1:
        sys.exit(0)

    existing = get_children(ppid)

    if not daemonize():
        sys.exit(0)

    setup_logging()

    try:
        monitor(ppid, command, existing)
    except Exception:
        logger.exception("crash")


if __name__ == "__main__":
    main()
