"""Runner for executing skill evaluations using claude-agent-sdk."""

import json
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolUseBlock,
)

from skill_evaluation.models import EvaluationSpec


class ResponseStreamLogger:
    """Middleware for logging Claude SDK messages to JSONL file."""

    def __init__(self, parent: AsyncIterator[Message], output_file: Path | None):
        """
        Initialize response stream logger.

        Args:
            parent: Async iterator of messages from ClaudeSDKClient
            output_file: Path to write JSONL output, or None to disable logging
        """
        self.parent = parent
        self.output_file = output_file
        self._file_handle = open(output_file, "w") if output_file else None

    def __aiter__(self) -> AsyncIterator[Message]:
        return self

    async def __anext__(self) -> Message:
        try:
            message = await self.parent.__anext__()

            if self._file_handle:
                json.dump(asdict(message), self._file_handle, default=str)
                self._file_handle.write("\n")
                self._file_handle.flush()

            return message
        except StopAsyncIteration:
            if self._file_handle:
                self._file_handle.close()
            raise


class ResponseStreamVerbosePrinter:
    """Middleware for printing live verbose output from message stream."""

    def __init__(self, parent: AsyncIterator[Message]):
        """
        Initialize verbose printer.

        Args:
            parent: Async iterator of messages (from ClaudeSDKClient or another middleware)
        """
        self.parent = parent
        self.turn_count = 0

    def __aiter__(self) -> AsyncIterator[Message]:
        return self

    async def __anext__(self) -> Message:
        message = await self.parent.__anext__()

        if isinstance(message, AssistantMessage):
            self.turn_count += 1
            self._print_assistant_message(message)
        elif isinstance(message, ResultMessage):
            self._print_result_message(message)

        return message

    def _print_assistant_message(self, message: AssistantMessage) -> None:
        """Print concise summary of assistant message content."""
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                tool_input_preview = str(block.input)[:60]
                if len(str(block.input)) > 60:
                    tool_input_preview += "..."
                print(
                    f"[Turn {self.turn_count}] ToolUse: {block.name} -> {tool_input_preview}",
                    flush=True,
                )
            elif isinstance(block, TextBlock):
                print(f"[Turn {self.turn_count}] Text: {block.text}", flush=True)

    def _print_result_message(self, message: ResultMessage) -> None:
        """Print result message summary."""
        status = "ERROR" if message.is_error else "SUCCESS"
        duration_s = message.duration_ms / 1000
        print(
            f"[Result] {status} in {duration_s:.1f}s, {message.num_turns} turns",
            flush=True,
        )
        if message.is_error and message.result:
            print(f"         Error: {message.result[:100]}", flush=True)


class ResponseStreamSkillActivationChecker:
    """Middleware that enforces skill activation within timeout period."""

    def __init__(
        self,
        parent: AsyncIterator[Message],
        timeout_turns: int = 5,
        verbose: bool = False,
        interrupt: Callable[[], Awaitable[None]] | None = None,
    ):
        """
        Initialize skill activation checker.

        Args:
            parent: Async iterator of messages
            timeout_turns: Number of turns before timeout
            verbose: Whether to print activation status
            interrupt: Callable that interrupts the client when timeout occurs
        """
        self.parent = parent
        self.timeout_turns = timeout_turns
        self.verbose = verbose
        self.turn_count = 0
        self.skill_activated = False
        self.interrupt = interrupt
        self._timed_out = False

    def __aiter__(self) -> AsyncIterator[Message]:
        return self

    async def __anext__(self) -> Message:
        message = await self.parent.__anext__()

        if isinstance(message, AssistantMessage):
            self.turn_count += 1

            if self._contains_skill_tool_call(message):
                self.skill_activated = True
                if self.verbose:
                    print(f"[Turn {self.turn_count}] ✓ Skill activated", flush=True)

            elif (
                not self.skill_activated
                and self.turn_count >= self.timeout_turns
                and not self._timed_out
            ):
                print(
                    f"\n⚠ Skill not activated within {self.timeout_turns} turns. Terminating evaluation.",
                    flush=True,
                )
                self._timed_out = True
                if self.interrupt:
                    await self.interrupt()

        return message

    def _contains_skill_tool_call(self, message: AssistantMessage) -> bool:
        """Check if message contains Skill tool invocation."""
        for block in message.content:
            if isinstance(block, ToolUseBlock) and block.name == "Skill":
                return True
        return False


def setup_workspace(
    spec: EvaluationSpec, skill_name: str, skills_source_dir: Path
) -> Path:
    """
    Create temporary workspace with skill and test files.

    Args:
        spec: Evaluation specification
        skill_name: Name of the skill to load
        skills_source_dir: Directory containing skills to copy (repository-relative, e.g., ./skills)

    Returns:
        Path to temporary workspace
    """
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="skill-eval-"))

    # Set up .claude/skills/ directory
    claude_skills_dir = temp_dir / ".claude" / "skills"
    claude_skills_dir.mkdir(parents=True)

    # Copy the skill, excluding evaluation.json
    skill_src = skills_source_dir / skill_name
    if not skill_src.exists():
        raise FileNotFoundError(f"Skill not found: {skill_name} at {skill_src}")

    skill_dest = claude_skills_dir / skill_name
    shutil.copytree(
        skill_src,
        skill_dest,
        ignore=lambda src, names: ["evaluation.json"]
        if "evaluation.json" in names
        else [],
    )

    # Copy test files if specified
    for file_path_str in spec.files:
        file_path = Path(file_path_str)
        if file_path.exists():
            dest_path = temp_dir / file_path.name
            shutil.copy2(file_path, dest_path)

    return temp_dir


async def run_evaluation(
    spec: EvaluationSpec,
    skill_name: str,
    skills_source_dir: Path,
    verbose: bool = False,
    output_file: Path | None = None,
) -> bool:
    """
    Run an evaluation by setting up Claude with the specified skill and query.

    Args:
        spec: The evaluation specification
        skill_name: Name of the skill to evaluate
        skills_source_dir: Source directory for skills (repository-relative path)
        verbose: Show live output during evaluation
        output_file: Path to stream messages to as they arrive (JSONL format)

    Returns:
        True if evaluation completed normally, False if early-exit occurred
    """
    # Set up isolated workspace
    work_dir = setup_workspace(spec, skill_name, skills_source_dir)

    try:
        # Configure SDK options
        options = ClaudeAgentOptions(
            model="sonnet",
            cwd=str(work_dir),
            setting_sources=["project"],  # Load skills from .claude/ in workspace
            system_prompt={"type": "preset", "preset": "claude_code"},
            permission_mode="bypassPermissions",  # Auto-approve for evaluation
        )

        prefixed_query = f"<context> You should write files only to the current project directory (your cwd). </context>\n{spec.query}"

        executed_successfully = True
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prefixed_query)

            stream = client.receive_response()
            stream = ResponseStreamLogger(stream, output_file)
            stream = ResponseStreamSkillActivationChecker(
                stream,
                timeout_turns=5,
                verbose=verbose,
                interrupt=client.interrupt,
            )
            if verbose:
                stream = ResponseStreamVerbosePrinter(stream)

            async for message in stream:
                if isinstance(message, SystemMessage):
                    if skill_name not in message.data.get("skills", []):
                        print(
                            "\n⚠ Skill not loaded. Terminating evaluation.", flush=True
                        )
                        executed_successfully = False
                        await client.interrupt()

                elif isinstance(message, ResultMessage):
                    executed_successfully = message.subtype != "error_during_execution"

        return executed_successfully

    finally:
        # Cleanup workspace
        if work_dir.exists():
            shutil.rmtree(work_dir)
