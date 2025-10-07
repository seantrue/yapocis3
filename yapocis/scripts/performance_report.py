#!/usr/bin/env python3
#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""
CLI tool for managing yapocis performance data and reports.

Usage:
    python scripts/performance_report.py list [--backend=BACKEND] [--test=TEST]
    python scripts/performance_report.py stats [--backend=BACKEND] [--test=TEST]
    python scripts/performance_report.py compare <test_name>
    python scripts/performance_report.py trend <test_name> [--backend=BACKEND]
    python scripts/performance_report.py comprehensive [--test=TEST]
    python scripts/performance_report.py purge [--days=N] [--backend=BACKEND] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yapocis.utils.performance import (
    load_performance_data,
    purge_performance_data,
    generate_summary_table,
    generate_checkpoint_comparison_table,
    generate_backend_comparison_report,
    generate_longitudinal_report,
    generate_comprehensive_report,
    plot_backend_comparison,
    plot_longitudinal_trend,
)


def cmd_list(args):
    """List performance data matching filters."""
    data = load_performance_data(
        backend=args.backend,
        test_name=args.test,
        limit=args.limit
    )

    if not data:
        print("No performance data found.")
        return

    print(f"Found {len(data)} performance records:\n")
    print(generate_summary_table(data))


def cmd_stats(args):
    """Show performance statistics."""
    data = load_performance_data(
        backend=args.backend,
        test_name=args.test,
        limit=200
    )

    if not data:
        print("No performance data found.")
        return

    # Group by test and backend
    test_backend_times = {}
    for entry in data:
        test = entry["metadata"]["test_name"]
        backend = entry["metadata"]["backend"]
        total_time = entry["timing"]["total"]

        key = (test, backend)
        if key not in test_backend_times:
            test_backend_times[key] = []
        test_backend_times[key].append(total_time)

    # Display statistics
    print(f"Performance Statistics ({len(data)} total runs)\n")
    print("=" * 80)

    current_test = None
    for (test, backend), times in sorted(test_backend_times.items()):
        if test != current_test:
            if current_test is not None:
                print()
            print(f"\n{test}")
            print("-" * 80)
            current_test = test

        mean = sum(times) / len(times)
        if len(times) > 1:
            variance = sum((t - mean) ** 2 for t in times) / len(times)
            std = variance ** 0.5
            min_time = min(times)
            max_time = max(times)
            print(f"  {backend:10s}: {mean:.3f}s ± {std:.3f}s  "
                  f"(min: {min_time:.3f}s, max: {max_time:.3f}s, runs: {len(times)})")
        else:
            print(f"  {backend:10s}: {mean:.3f}s (1 run)")

    print("\n" + "=" * 80)


def cmd_compare(args):
    """Generate backend comparison report for a test."""
    try:
        report_path = generate_backend_comparison_report(args.test_name)
        print(f"Backend comparison report generated: {report_path}")
        print(f"\nView the report: open {report_path}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_trend(args):
    """Generate longitudinal trend report for a test."""
    try:
        report_path = generate_longitudinal_report(
            args.test_name,
            backend=args.backend
        )
        print(f"Longitudinal trend report generated: {report_path}")
        print(f"\nView the report: open {report_path}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_comprehensive(args):
    """Generate comprehensive performance report."""
    try:
        report_path = generate_comprehensive_report(test_name=args.test)
        print(f"Comprehensive report generated: {report_path}")
        print(f"\nView the report: open {report_path}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_purge(args):
    """Purge old performance data."""
    count, files = purge_performance_data(
        older_than_days=args.days,
        backend=args.backend,
        test_name=args.test,
        dry_run=args.dry_run
    )

    if args.dry_run:
        print(f"Would delete {count} files (dry run):")
    else:
        print(f"Deleted {count} files:")

    if args.verbose and files:
        for filename in files:
            print(f"  - {filename}")
    elif count > 0:
        print(f"  (use --verbose to see filenames)")


def main():
    parser = argparse.ArgumentParser(
        description="Manage yapocis performance data and reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all performance data
  python scripts/performance_report.py list

  # List data for specific backend
  python scripts/performance_report.py list --backend=metal

  # Show statistics
  python scripts/performance_report.py stats --test=graphics

  # Generate backend comparison report
  python scripts/performance_report.py compare test_graphics4.test_batch

  # Generate trend report for specific backend
  python scripts/performance_report.py trend test_graphics4.test_batch --backend=ocl

  # Generate comprehensive report
  python scripts/performance_report.py comprehensive

  # Purge data older than 30 days (dry run)
  python scripts/performance_report.py purge --days=30 --dry-run

  # Delete old data for specific backend
  python scripts/performance_report.py purge --days=7 --backend=numba
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    list_parser = subparsers.add_parser("list", help="List performance data")
    list_parser.add_argument("--backend", help="Filter by backend")
    list_parser.add_argument("--test", help="Filter by test name (partial match)")
    list_parser.add_argument("--limit", type=int, default=50, help="Maximum results (default: 50)")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show performance statistics")
    stats_parser.add_argument("--backend", help="Filter by backend")
    stats_parser.add_argument("--test", help="Filter by test name (partial match)")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Generate backend comparison report")
    compare_parser.add_argument("test_name", help="Test name to compare")

    # Trend command
    trend_parser = subparsers.add_parser("trend", help="Generate longitudinal trend report")
    trend_parser.add_argument("test_name", help="Test name to analyze")
    trend_parser.add_argument("--backend", help="Filter by specific backend")

    # Comprehensive command
    comp_parser = subparsers.add_parser("comprehensive", help="Generate comprehensive report")
    comp_parser.add_argument("--test", help="Filter by test name (partial match)")

    # Purge command
    purge_parser = subparsers.add_parser("purge", help="Delete old performance data")
    purge_parser.add_argument("--days", type=int, help="Delete data older than N days")
    purge_parser.add_argument("--backend", help="Only delete data for specific backend")
    purge_parser.add_argument("--test", help="Only delete data for specific test")
    purge_parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    purge_parser.add_argument("--verbose", "-v", action="store_true", help="Show filenames")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    commands = {
        "list": cmd_list,
        "stats": cmd_stats,
        "compare": cmd_compare,
        "trend": cmd_trend,
        "comprehensive": cmd_comprehensive,
        "purge": cmd_purge,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
