"""Runner for executing skill evaluations using claude-agent-sdk."""

import json
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    SystemMessage,
    ToolUseBlock,
)

from skill_evaluation.models import EvaluationSpec


def print_message_live(message, turn_count: int) -> None:
    """Print a concise summary of a message for live monitoring."""
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                tool_input_preview = str(block.input)[:60]
                if len(str(block.input)) > 60:
                    tool_input_preview += "..."
                print(f"[Turn {turn_count}] ToolUse: {block.name} -> {tool_input_preview}", flush=True)
            elif isinstance(block, TextBlock):
                text_preview = block.text
                print(f"[Turn {turn_count}] Text: {text_preview}", flush=True)
    elif isinstance(message, ResultMessage):
        status = "ERROR" if message.is_error else "SUCCESS"
        duration_s = message.duration_ms / 1000
        print(f"[Result] {status} in {duration_s:.1f}s, {message.num_turns} turns", flush=True)
        if message.is_error and message.result:
            print(f"         Error: {message.result[:100]}", flush=True)


def setup_workspace(spec: EvaluationSpec, skills_source_dir: Path) -> Path:
    """
    Create temporary workspace with skills and test files.

    Args:
        spec: Evaluation specification
        skills_source_dir: Directory containing skills to copy (e.g., ~/.claude/skills)

    Returns:
        Path to temporary workspace
    """
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="skill-eval-"))

    # Set up .claude/skills/ directory
    claude_skills_dir = temp_dir / ".claude" / "skills"
    claude_skills_dir.mkdir(parents=True)

    # Copy each required skill
    for skill_name in spec.skills:
        skill_src = skills_source_dir / skill_name
        if skill_src.exists():
            skill_dest = claude_skills_dir / skill_name
            shutil.copytree(skill_src, skill_dest)
        else:
            raise FileNotFoundError(f"Skill not found: {skill_name} at {skill_src}")

    # Copy test files if specified
    for file_path_str in spec.files:
        file_path = Path(file_path_str)
        if file_path.exists():
            dest_path = temp_dir / file_path.name
            shutil.copy2(file_path, dest_path)

    return temp_dir


async def run_evaluation(
    spec: EvaluationSpec,
    skills_source_dir: Path | None = None,
    verbose: bool = False,
    output_file: Path | None = None
) -> None:
    """
    Run an evaluation by setting up Claude with the specified skills and query.

    Args:
        spec: The evaluation specification
        skills_source_dir: Source directory for skills (defaults to ~/.claude/skills)
        verbose: Show live output during evaluation
        output_file: Path to stream messages to as they arrive (JSONL format)
    """
    # Default to user's skills directory
    if skills_source_dir is None:
        skills_source_dir = Path.home() / ".claude" / "skills"

    # Set up isolated workspace
    work_dir = setup_workspace(spec, skills_source_dir)

    try:
        # Configure SDK options
        options = ClaudeAgentOptions(
            model="sonnet",
            cwd=str(work_dir),
            setting_sources=["project"],  # Load skills from .claude/ in workspace
            system_prompt={"type": "preset", "preset": "claude_code"},
            permission_mode="bypassPermissions",  # Auto-approve for evaluation
        )

        # Track turn count for verbose output
        turn_count = 0

        # Build query with minimal file location instruction
        prefixed_query = f"<context> You should write files only to the current project directory (your cwd). </context>\n{spec.query}"

        # Use context manager for safe file handling
        file_context = open(output_file, "w") if output_file else nullcontext()

        with file_context as output_file_handle:
            # Write query metadata as first line if streaming
            if output_file:
                json.dump({"type": "metadata", "query": prefixed_query}, output_file_handle, default=str)
                output_file_handle.write("\n")
                output_file_handle.flush()

            async with ClaudeSDKClient(options=options) as client:
                await client.query(prefixed_query)

                # Stream all messages from response
                async for message in client.receive_response():
                    # Stream to file if enabled (one JSON object per line)
                    if output_file:
                        json.dump(asdict(message), output_file_handle, default=str)
                        output_file_handle.write("\n")
                        output_file_handle.flush()

                    if isinstance(message, SystemMessage):
                        print(message.data)

                    # Track turns for verbose output
                    if isinstance(message, AssistantMessage):
                        turn_count += 1

                    # Print live if verbose mode
                    if verbose:
                        print_message_live(message, turn_count)

    finally:
        # Cleanup workspace
        if work_dir.exists():
            shutil.rmtree(work_dir)
