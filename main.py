"""
FlyBrain Metal64 simulation and benchmark runner.

Usage:
    # One 0.1-second trial on Apple silicon with MLX float64
    python main.py

    # P9 forward-walking experiment instead
    python main.py --experiment p9

    # Specific durations and trial count on MLX
    python main.py --t_run 0.1 1 10 --n_run 1

    # Specific backends (combinable)
    python main.py --mlx                                  # MLX on Apple Metal
    python main.py --brian2-cpu                          # Brian2 CPU only
    python main.py --brian2cuda-gpu                      # Brian2CUDA GPU only
    python main.py --pytorch                             # PyTorch only
    python main.py --nestgpu                             # NEST GPU only
    python main.py --genn                                # GeNN only
    python main.py --brian2genn                          # Brian2GeNN GPU only
    python main.py --brian2-cpu --pytorch --nestgpu      # Brian2 CPU + PyTorch + NEST GPU

    # Brian2GeNN uses a separate environment because it pins Brian2<2.6
    python main.py --brian2genn --paper --run-label nature_2026_05

    # Opt in to the full paper grid: 5 rounds of durations and trial counts
    python main.py --paper --run-label nature_2026_05

    # Background with log
    nohup python main.py > data/results/benchmarks.log 2>&1 &
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import sys
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))

from benchmark import (
    BenchmarkLogger, T_RUN_VALUES_SEC, PAPER_T_RUN_VALUES_SEC,
    N_RUN_VALUES, PAPER_ROUNDS, BACKEND_NAMES, EXPERIMENTS,
    SPIKE_IO_ENV_VAR, get_experiment, get_spike_output_dir, run_benchmarks,
    spike_io_enabled,
)


BACKEND_OPTIONS = (
    ('brian2-cpu', 'cpu', 'Run Brian2 C++ standalone (CPU)'),
    ('brian2cuda-gpu', 'gpu', 'Run Brian2CUDA (GPU)'),
    ('pytorch', 'pytorch', 'Run PyTorch benchmark'),
    ('nestgpu', 'nestgpu', 'Run NEST GPU benchmark'),
    ('genn', 'genn', 'Run GeNN benchmark'),
    ('brian2genn', 'brian2genn', 'Run Brian2GeNN benchmark'),
    ('mlx', 'mlx', 'Run float64 Eon LIF on Apple Metal with VF64'),
)


def main():
    parser = argparse.ArgumentParser(
        description='FlyBrain Metal64: run the Drosophila brain model. '
                    'Default: one 0.1-second trial on Apple Metal with MLX.'
    )
    parser.add_argument('--t_run', type=float, nargs='+', default=None,
                        help='Simulation duration(s) in seconds. '
                             f'Allowed: {T_RUN_VALUES_SEC}. Default: 0.1 for '
                             'MLX-only runs; all for other backends; '
                             f'{PAPER_T_RUN_VALUES_SEC} with --paper')
    parser.add_argument('--n_run', type=int, nargs='+', default=None,
                        help='Number of trials. Default: 1 for MLX-only runs; '
                             f'{N_RUN_VALUES} for other backends or --paper')
    parser.add_argument('--rounds', type=int, default=None,
                        help='Repeat the full backend/parameter suite N times. '
                             'Default: 1, or 5 with --paper')
    parser.add_argument('--round-start', '--round_start', dest='round_start',
                        type=int, default=1,
                        help='First round number to write. Useful for resuming '
                             'an interrupted labeled run')
    parser.add_argument('--run-label', '--run_label', dest='run_label',
                        type=str, default=None,
                        help='Label for grouping round-specific spike outputs '
                             'under data/results/<label>/')
    parser.add_argument('--paper', action='store_true',
                        help='Run the paper benchmark suite: t_run '
                             f'{PAPER_T_RUN_VALUES_SEC}, n_run {N_RUN_VALUES}, '
                             f'{PAPER_ROUNDS} rounds unless --rounds is set')
    parser.add_argument('--log_file', type=str, default='data/results/benchmarks.log',
                        help='Log file path. Default: data/results/benchmarks.log')
    parser.add_argument('--no_log_file', action='store_true',
                        help='Disable file logging (console only)')
    parser.add_argument('--disable-spike-io', '--disable_spike_io',
                        action='store_true',
                        help='Disable spike probing/recording and spike parquet '
                             'output for timing-only benchmarks')

    parser.add_argument('--experiment', type=str, default=None,
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment to run. '
                             f'Available: {list(EXPERIMENTS.keys())}. '
                             'Default: sugar')

    for flag, _, help_text in BACKEND_OPTIONS:
        parser.add_argument('--' + flag, action='store_true', help=help_text)

    args = parser.parse_args()

    if args.disable_spike_io:
        os.environ[SPIKE_IO_ENV_VAR] = '1'

    backends = [key for flag, key, _ in BACKEND_OPTIONS
                if getattr(args, flag.replace('-', '_'))]
    if not backends:
        backends = ['mlx']

    # Import brian2cuda once to register the cuda_standalone device
    if 'gpu' in backends:
        import brian2cuda  # noqa: F401

    rounds = args.rounds
    if rounds is None:
        rounds = PAPER_ROUNDS if args.paper else 1
    if rounds < 1:
        print("Error: --rounds must be >= 1")
        return
    if args.round_start < 1:
        print("Error: --round-start must be >= 1")
        return

    # Validate t_run values
    t_run_values = args.t_run
    n_run_values = args.n_run
    if args.paper:
        if t_run_values is None:
            t_run_values = PAPER_T_RUN_VALUES_SEC
        if n_run_values is None:
            n_run_values = N_RUN_VALUES
    elif backends == ['mlx']:
        if t_run_values is None:
            t_run_values = [0.1]
        if n_run_values is None:
            n_run_values = [1]

    if t_run_values:
        for val in t_run_values:
            if val not in T_RUN_VALUES_SEC:
                print(
                    f"Error: --t_run {val} is not in allowed values: "
                    f"{T_RUN_VALUES_SEC}"
                )
                return

    experiment = get_experiment(args.experiment)

    if backends == ['mlx'] and not args.run_label:
        args.run_label = datetime.now().strftime('mlx_%Y%m%d_%H%M%S_%f')

    log_file = None if args.no_log_file else args.log_file
    logger = BenchmarkLogger(log_file=log_file)

    try:
        logger.log_raw("")
        logger.log("Starting benchmark suite")
        logger.log(
            f"Backends: {', '.join(BACKEND_NAMES[b] for b in backends)}"
        )
        t_run_display = t_run_values if t_run_values else T_RUN_VALUES_SEC
        logger.log(f"t_run values: {t_run_display}s")
        logger.log(f"n_run values: {n_run_values or N_RUN_VALUES}")
        logger.log(f"rounds: {rounds}")
        logger.log(f"round start: {args.round_start}")
        if args.run_label:
            logger.log(f"run label: {args.run_label}")
        logger.log(
            "spike probing/output: "
            f"{'enabled' if spike_io_enabled() else 'disabled'}"
        )
        if spike_io_enabled():
            logger.log(f"Spike output directory: {get_spike_output_dir(args.run_label)}")
        logger.log(f"Log file: {log_file if log_file else 'disabled'}")

        run_benchmarks(
            backends=backends,
            t_run_values=t_run_values,
            n_run_values=n_run_values,
            experiment=experiment,
            logger=logger,
            rounds=rounds,
            run_label=args.run_label,
            round_start=args.round_start,
        )

    finally:
        logger.close()


if __name__ == '__main__':
    main()
