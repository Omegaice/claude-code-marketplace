"""Skill evaluation tool for Claude Code plugins."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from skill_evaluation.models import EvaluationSpec, EvaluationResult
from skill_evaluation.runner import run_evaluation
from skill_evaluation.judge import judge_evaluation


async def evaluate_skill(
    skill_dir: Path,
    skills_repo_dir: Path,
    output_dir: Path | None = None,
    verbose: bool = False
) -> EvaluationResult:
    """
    Run a skill evaluation from a skill directory containing evaluation.json.

    Args:
        skill_dir: Path to the skill directory containing evaluation.json
        skills_repo_dir: Path to the repository's skills directory (e.g., ./skills)
        output_dir: Optional output directory for results (defaults to ./results)
        verbose: Show live output during evaluation

    Returns:
        EvaluationResult with the outcome
    """
    # Load evaluation spec from skill directory
    eval_file = skill_dir / "evaluation.json"
    if not eval_file.exists():
        raise FileNotFoundError(f"evaluation.json not found in {skill_dir}")

    with open(eval_file) as f:
        spec_data = json.load(f)

    spec = EvaluationSpec(**spec_data)

    # Create output directory and file path before running evaluation
    if output_dir is None:
        output_dir = Path.cwd() / "results"

    eval_name = skill_dir.name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = output_dir / f"{eval_name}-{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    runner_output_file = result_dir / "runner-output.jsonl"

    print(f"Running evaluation with skills: {', '.join(spec.skills)}", flush=True)
    if verbose:
        print(f"{'─'*60}", flush=True)
    await run_evaluation(spec, skills_source_dir=skills_repo_dir, verbose=verbose, output_file=runner_output_file)

    # Judge the results by reading the file
    print("Evaluating results with LLM judge...")
    passed, score, reasoning = await judge_evaluation(spec, runner_output_file)

    # Save judge output and summary
    judge_output = {
        "passed": passed,
        "score": score,
        "reasoning": reasoning,
        "threshold": 0.7,
    }
    judge_output_file = result_dir / "judge-output.json"
    judge_output_file.write_text(json.dumps(judge_output, indent=2))

    summary = {
        "skill_directory": str(skill_dir),
        "timestamp": timestamp,
        "skills": spec.skills,
        "query": spec.query,
        "expected_behavior": spec.expected_behavior,
        "result": {
            "passed": passed,
            "score": score,
            "reasoning": reasoning,
        },
        "runner_output_file": str(runner_output_file),
        "judge_output_file": str(judge_output_file),
    }
    summary_file = result_dir / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print(f"\nResults saved to: {result_dir}")

    return EvaluationResult(
        spec=spec,
        passed=passed,
        score=score,
        reasoning=reasoning
    )


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Claude Code skills")
    parser.add_argument("skill_dir", type=Path, help="Path to skill directory containing evaluation.json")
    parser.add_argument(
        "--skills-repo",
        type=Path,
        help="Path to skills repository directory (defaults to ./skills relative to cwd)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show live output during evaluation"
    )

    args = parser.parse_args()

    if not args.skill_dir.exists():
        print(f"Error: Skill directory not found: {args.skill_dir}")
        sys.exit(1)

    if not args.skill_dir.is_dir():
        print(f"Error: {args.skill_dir} is not a directory")
        sys.exit(1)

    eval_file = args.skill_dir / "evaluation.json"
    if not eval_file.exists():
        print(f"Error: evaluation.json not found in {args.skill_dir}")
        sys.exit(1)

    skills_repo_dir = args.skills_repo if args.skills_repo else Path.cwd() / "skills"
    if not skills_repo_dir.exists():
        print(f"Error: Skills repository directory not found: {skills_repo_dir}")
        sys.exit(1)

    try:
        result = asyncio.run(evaluate_skill(args.skill_dir, skills_repo_dir, verbose=args.verbose))

        # Print results
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"Status: {'PASSED' if result.passed else 'FAILED'}")
        print(f"Score: {result.score:.2f}")
        print(f"\nReasoning:")
        print(result.reasoning)
        print(f"{'='*60}\n")

        sys.exit(0 if result.passed else 1)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.", file=sys.stderr)
        sys.exit(130)  # Standard exit code for Ctrl+C
