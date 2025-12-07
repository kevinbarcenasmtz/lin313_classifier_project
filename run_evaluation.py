#!/usr/bin/env python3
"""
Standalone script for running systematic shot count evaluation.

Usage:
    python run_evaluation.py --api-key YOUR_KEY --output results.json
    python run_evaluation.py --api-key YOUR_KEY --shot-counts 1 5 10 --trials 2
"""

import argparse
import json
import sys
from pathlib import Path

from data_loading import load_homo_mex_dataset, prepare_label_vectors, validate_dataset
from classification import split_train_test
from evaluation import (
    run_shot_count_evaluation,
    save_evaluation_results,
    load_evaluation_results
)
from constants import (
    EVALUATION_SHOT_COUNTS,
    DEFAULT_NUM_TRIALS,
    EVALUATION_RANDOM_SEEDS,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS
)
from logger_config import logger


def print_summary_table(results):
    """Print a formatted summary table of results."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()
    
    df = results.aggregated_metrics
    print(df.to_string(index=False))
    print()
    
    print(f"Total Cost: ${results.total_cost:.4f}")
    print(f"Total Experiments: {len(results.configs)}")
    print(f"Test Set Size: {results.metadata.get('test_set_size', 'N/A')}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic shot count evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (1, 5, 10, 15, 20 shots, 2 trials each)
  python run_evaluation.py --api-key sk-...

  # Run specific shot counts
  python run_evaluation.py --api-key sk-... --shot-counts 1 10 20

  # Run with 3 trials per configuration
  python run_evaluation.py --api-key sk-... --trials 3

  # Use different model
  python run_evaluation.py --api-key sk-... --model gpt-4.1-nano
        """
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='OpenAI API key'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output JSON file path (default: evaluation_results.json)'
    )
    
    parser.add_argument(
        '--shot-counts',
        type=int,
        nargs='+',
        default=EVALUATION_SHOT_COUNTS,
        help=f'Shot counts to evaluate (default: {EVALUATION_SHOT_COUNTS})'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=f'Number of trials per shot count (default: {DEFAULT_NUM_TRIALS})'
    )
    
    parser.add_argument(
        '--random-seeds',
        type=int,
        nargs='+',
        default=None,
        help=f'Random seeds for trials (default: {EVALUATION_RANDOM_SEEDS})'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model name (default: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f'Temperature setting (default: {DEFAULT_TEMPERATURE})'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Max tokens for responses (default: {DEFAULT_MAX_TOKENS})'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=DEFAULT_TEST_SIZE,
        help=f'Test set size fraction (default: {DEFAULT_TEST_SIZE})'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='data/Annotated LGBTQ+ Phobia Tweets.xlsx',
        help='Path to dataset Excel file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint JSON file (not yet implemented)'
    )
    
    args = parser.parse_args()
    
    # Validate shot counts
    for shot_count in args.shot_counts:
        if shot_count < 1:
            print(f"Error: Shot count must be >= 1, got {shot_count}", file=sys.stderr)
            sys.exit(1)
    
    # Validate random seeds
    if args.random_seeds is None:
        random_seeds = EVALUATION_RANDOM_SEEDS[:args.trials]
        if len(random_seeds) < args.trials:
            # Generate additional seeds
            random_seeds = random_seeds + [
                DEFAULT_RANDOM_SEED + i for i in range(len(random_seeds), args.trials)
            ]
    else:
        if len(args.random_seeds) < args.trials:
            print(f"Warning: Only {len(args.random_seeds)} seeds provided for {args.trials} trials. "
                  f"Generating additional seeds.", file=sys.stderr)
            random_seeds = args.random_seeds + [
                DEFAULT_RANDOM_SEED + i for i in range(len(args.random_seeds), args.trials)
            ]
        else:
            random_seeds = args.random_seeds[:args.trials]
    
    # Estimate cost
    from api_utils import estimate_api_cost
    estimated_test_size = int(862 * args.test_size)  # Approximate
    total_experiments = len(args.shot_counts) * args.trials
    avg_shot_count = sum(args.shot_counts) / len(args.shot_counts)
    
    cost_estimate = estimate_api_cost(
        model_name=args.model,
        num_test_tweets=estimated_test_size,
        num_ablations=total_experiments,
        few_shot_examples=int(avg_shot_count)
    )
    
    print("=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Shot Counts: {args.shot_counts}")
    print(f"Trials per Configuration: {args.trials}")
    print(f"Random Seeds: {random_seeds}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Test Size: {args.test_size}")
    print(f"Total Experiments: {total_experiments}")
    print(f"Estimated Cost: ${cost_estimate['total_cost']:.2f}")
    print("=" * 80)
    print()
    
    response = input("Continue with evaluation? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Evaluation cancelled.")
        sys.exit(0)
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        df = load_homo_mex_dataset(args.data_file)
        df = prepare_label_vectors(df)
        validation = validate_dataset(df)
        
        if not validation['valid'] and validation.get('errors'):
            print("Dataset validation errors:", file=sys.stderr)
            for error in validation['errors']:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)
        
        if validation.get('warnings'):
            print("Dataset validation warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        print(f"Dataset loaded: {len(df)} tweets")
        
    except FileNotFoundError:
        print(f"Error: Dataset file not found: {args.data_file}", file=sys.stderr)
        print("Please ensure the Excel file exists at the specified path.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    # Split train/test
    print("Splitting dataset into train/test sets...")
    train_df, test_df = split_train_test(df, test_size=args.test_size, random_seed=DEFAULT_RANDOM_SEED)
    print(f"Train: {len(train_df)} tweets, Test: {len(test_df)} tweets")
    
    # Progress callback
    def progress_callback(current, total, message):
        print(f"[{current}/{total}] {message}")
    
    # Run evaluation
    print("\nStarting evaluation...")
    print("-" * 80)
    
    try:
        results = run_shot_count_evaluation(
            train_df=train_df,
            test_df=test_df,
            shot_counts=args.shot_counts,
            num_trials=args.trials,
            api_key=args.api_key,
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            random_seeds=random_seeds,
            progress_callback=progress_callback
        )
        
        # Save results
        output_path = Path(args.output)
        save_evaluation_results(results, str(output_path))
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        print_summary_table(results)
        
        # Also save CSV version of aggregated metrics
        csv_path = output_path.with_suffix('.csv')
        results.aggregated_metrics.to_csv(csv_path, index=False)
        print(f"Aggregated metrics CSV saved to: {csv_path}")
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        print("Partial results may be available in session state.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}", file=sys.stderr)
        logger.exception("Evaluation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

