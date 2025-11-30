"""Main CLI entry point for kompot."""
import argparse
import sys
import os

# DO NOT import subcommand modules here - they import NumPy which must come
# AFTER setting thread limit environment variables

from .utils import setup_logging


def _set_early_thread_limits(args_list):
    """
    Set thread limit environment variables BEFORE any NumPy imports.

    This parses --threads from command line args without fully parsing,
    allowing us to set thread limits before NumPy initialization.

    Parameters
    ----------
    args_list : list
        Command line arguments (typically sys.argv[1:])
    """
    # Look for --threads or --use-gpu in args
    n_threads = None

    for i, arg in enumerate(args_list):
        if arg == '--threads' and i + 1 < len(args_list):
            try:
                n_threads = int(args_list[i + 1])
            except ValueError:
                pass  # Will be handled by proper argparse later
            break
        elif arg.startswith('--threads='):
            try:
                n_threads = int(arg.split('=')[1])
            except ValueError:
                pass
            break

    # Set thread limits if specified
    if n_threads is not None:
        n_threads_str = str(n_threads)
        os.environ['OMP_NUM_THREADS'] = n_threads_str
        os.environ['MKL_NUM_THREADS'] = n_threads_str
        os.environ['OPENBLAS_NUM_THREADS'] = n_threads_str
        os.environ['BLAS_NUM_THREADS'] = n_threads_str
        # Note: We can't log yet as logging isn't set up


def main():
    """Main CLI entry point."""
    # Set thread limits BEFORE any imports that might load NumPy
    _set_early_thread_limits(sys.argv[1:])

    # NOW we can safely import modules that use NumPy
    from .de import add_de_parser
    from .da import add_da_parser
    from .dm import add_dm_parser

    parser = argparse.ArgumentParser(
        prog='kompot',
        description='Kompot: Differential abundance and expression analysis for single-cell data',
        epilog='For more information, visit https://kompot.readthedocs.io'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.6.0'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available analysis commands',
        dest='command',
        help='Use "kompot <command> --help" for command-specific help'
    )

    # Add subcommands
    add_dm_parser(subparsers)
    add_de_parser(subparsers)
    add_da_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # If no command provided, print help
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Run the appropriate command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
