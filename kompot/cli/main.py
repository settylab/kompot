"""Main CLI entry point for kompot."""
import argparse
import sys

from .de import add_de_parser
from .da import add_da_parser
from .dm import add_dm_parser
from .utils import setup_logging


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='kompot',
        description='Kompot: Differential abundance and expression analysis for single-cell data',
        epilog='For more information, visit https://kompot.readthedocs.io'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 0.5.2'
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
