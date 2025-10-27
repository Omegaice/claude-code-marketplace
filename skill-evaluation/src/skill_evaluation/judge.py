"""LLM-as-a-judge for evaluating skill outputs."""

import json
import re
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from skill_evaluation.models import EvaluationSpec


async def judge_evaluation(
    spec: EvaluationSpec,
    runner_output_file: Path
) -> tuple[bool, float, str]:
    """
    Use LLM-as-a-judge to evaluate whether the captured output meets expectations.

    Args:
        spec: The evaluation specification with expected behaviors
        runner_output_file: Path to the runner-output.json file to evaluate

    Returns:
        Tuple of (passed, score, reasoning)
        - passed: Whether the evaluation passed
        - score: Score from 0.0 to 1.0
        - reasoning: LLM's reasoning for the score
    """
    # Build detailed judge prompt
    judge_prompt = f"""You are evaluating whether a Claude Code skill performed correctly.

ORIGINAL TASK:
{spec.query}

SKILLS LOADED:
{', '.join(spec.skills)}

EXPECTED BEHAVIORS:
{chr(10).join(f"{i+1}. {behavior}" for i, behavior in enumerate(spec.expected_behavior))}

EVALUATION INSTRUCTIONS:
Read the runner output file at: {runner_output_file.name}

This file is in JSONL (JSON Lines) format where each line is a separate JSON object:
- First line: metadata with the query
- Subsequent lines: messages from the evaluation run (init, assistant messages, tool results, etc.)

You can use jq via Bash to explore it efficiently:
- `cat {runner_output_file.name} | wc -l` - Count total lines
- `head -1 {runner_output_file.name} | jq '.query'` - Get the query
- `tail -n +2 {runner_output_file.name} | jq -s 'length'` - Count messages (skip metadata line)
- `tail -n +2 {runner_output_file.name} | jq -s '.[0].data.skills'` - Check which skills were available (from init message)
- `tail -n +2 {runner_output_file.name} | jq 'select(.content) | .content[] | select(.name) | .name'` - List tool names used
- `tail -n +2 {runner_output_file.name} | jq 'select(.content) | .content[] | select(.name == "Skill")'` - Find Skill tool invocations
- `tail -n 1 {runner_output_file.name} | jq '.'` - Get final message

Read the file to analyze the evaluation run and determine if expected behaviors were met:

EVALUATION CRITERIA:
- Did the skill load and activate correctly?
  * Skills are listed in the init message (second line of the file)
  * Skills are ACTIVATED by invoking the Skill tool: {{"name": "Skill", "input": {{"command": "skill-name"}}}}
  * Check the messages for a tool_use with name="Skill" to verify activation
  * DO NOT assume activation just from task completion - verify the Skill tool was invoked
- Were the expected behaviors demonstrated?
- Was the task completed successfully?
- Rate each expected behavior as met or not met
- Provide an overall score from 0.0 to 1.0 (0.0 = complete failure, 1.0 = perfect execution)
- Consider partial credit for partially met behaviors

IMPORTANT:
- Be specific about which expected behaviors were met or not met
- Provide concrete evidence from the captured output (line numbers, tool names, file paths)
- Verify skill activation by finding the Skill tool invocation, not just checking the init message
- A score >= 0.7 is considered passing

Respond with ONLY a JSON object (no markdown, no explanation):
{{
    "score": 0.0-1.0,
    "passed": true or false,
    "reasoning": "2-4 sentence explanation with specific evidence about which behaviors were met",
    "behaviors_met": ["list", "of", "met", "behavior", "indices"],
    "behaviors_not_met": ["list", "of", "unmet", "behavior", "indices"]
}}"""

    # Use SDK to get judgment - give judge Read and Bash (for jq) access
    options = ClaudeAgentOptions(
        model="opus",
        allowed_tools=["Read", "Bash"],
        cwd=str(runner_output_file.parent),  # Set cwd to result directory
        system_prompt="You are an impartial evaluator. Be precise, evidence-based, and output only valid JSON.",
        permission_mode="bypassPermissions",  # Auto-approve for judge
    )

    judgment_text = ""

    async with ClaudeSDKClient(options=options) as client:
        await client.query(judge_prompt)

        # Collect judge's response
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        judgment_text += block.text

            elif isinstance(message, ResultMessage):
                if message.is_error:
                    raise RuntimeError(f"Judge query failed: {message.result}")

    # Parse JSON response with fallback for markdown and text before JSON
    try:
        # Try direct parsing first
        judgment = json.loads(judgment_text.strip())
    except json.JSONDecodeError:
        # Try to extract from markdown code blocks
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', judgment_text, re.DOTALL)
        if match:
            try:
                judgment = json.loads(match.group(1))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse judge JSON response: {e}\nRaw: {judgment_text[:500]}")
        else:
            # Try to find JSON object anywhere in the text (judge might output text before JSON)
            # Find the first { that starts a JSON object
            start_idx = judgment_text.find('{')
            if start_idx != -1:
                # Try to parse from each { position
                for i in range(start_idx, len(judgment_text)):
                    if judgment_text[i] == '{':
                        try:
                            # Try to parse from this position to the end
                            judgment = json.loads(judgment_text[i:])
                            break
                        except json.JSONDecodeError:
                            continue
                else:
                    raise RuntimeError(f"No valid JSON found in judge response\nRaw: {judgment_text[:500]}")
            else:
                raise RuntimeError(f"No JSON object found in judge response\nRaw: {judgment_text[:500]}")

    # Extract values
    score = float(judgment.get("score", 0.0))
    passed = bool(judgment.get("passed", False))
    reasoning = str(judgment.get("reasoning", "No reasoning provided"))

    return passed, score, reasoning
