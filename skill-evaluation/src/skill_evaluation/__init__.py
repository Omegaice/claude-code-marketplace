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
    eval_file: Path,
    output_dir: Path | None = None,
    verbose: bool = False
) -> EvaluationResult:
    """
    Run a skill evaluation from a JSON specification file.

    Args:
        eval_file: Path to the evaluation JSON file
        output_dir: Optional output directory for results (defaults to ./results)
        verbose: Show live output during evaluation

    Returns:
        EvaluationResult with the outcome
    """
    # Load evaluation spec
    with open(eval_file) as f:
        spec_data = json.load(f)

    spec = EvaluationSpec(**spec_data)

    # Create output directory and file path before running evaluation
    if output_dir is None:
        output_dir = Path.cwd() / "results"

    eval_name = eval_file.stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = output_dir / f"{eval_name}-{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    runner_output_file = result_dir / "runner-output.jsonl"

    # Run the evaluation (creates temp workspace with skills)
    # Results will be streamed to runner_output_file as they arrive (JSONL format)
    print(f"Running evaluation with skills: {', '.join(spec.skills)}", flush=True)
    if verbose:
        print(f"{'─'*60}", flush=True)
    await run_evaluation(spec, verbose=verbose, output_file=runner_output_file)

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
        "evaluation_file": str(eval_file),
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
    parser.add_argument("eval_file", type=Path, help="Path to evaluation JSON file")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show live output during evaluation"
    )

    args = parser.parse_args()

    if not args.eval_file.exists():
        print(f"Error: Evaluation file not found: {args.eval_file}")
        sys.exit(1)

    try:
        # Run evaluation
        result = asyncio.run(evaluate_skill(args.eval_file, verbose=args.verbose))

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
