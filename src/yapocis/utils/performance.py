#  Copyright Sean D. True & Swapwizard LLC (c) 2011-2025.
"""Performance tracking and reporting for yapocis tests using DuckDB."""

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import duckdb

from .internals import get_performance_dir

# Performance data directory
PERFORMANCE_DIR = Path(get_performance_dir())
REPORTS_DIR = PERFORMANCE_DIR / "reports"
DB_PATH = PERFORMANCE_DIR / "performance.duckdb"

# Ensure directories exist
PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_connection():
    """Get a DuckDB connection and ensure schema exists."""
    conn = duckdb.connect(str(DB_PATH))

    # Create tables if they don't exist
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS runs_seq START 1
    """)

    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS checkpoints_seq START 1
    """)

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS performance_runs
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     DEFAULT
                     nextval
                 (
                     'runs_seq'
                 ),
                     test_name VARCHAR NOT NULL,
                     test_file VARCHAR NOT NULL,
                     backend VARCHAR NOT NULL,
                     device_name VARCHAR,
                     driver_version VARCHAR,
                     branch VARCHAR NOT NULL,
                     commit VARCHAR NOT NULL,
                     file_status VARCHAR NOT NULL,
                     timestamp TIMESTAMP NOT NULL,
                     timestamp_unix DOUBLE NOT NULL,
                     total_time DOUBLE NOT NULL,
                     min_time DOUBLE,
                     max_time DOUBLE,
                     mean_time DOUBLE,
                     slowest_operation VARCHAR,
                     fastest_operation VARCHAR
                     )
                 """)

    # Migrate existing schema: add device_name and driver_version columns if they don't exist
    try:
        conn.execute("ALTER TABLE performance_runs ADD COLUMN device_name VARCHAR")
    except Exception:
        pass  # Column already exists

    try:
        conn.execute("ALTER TABLE performance_runs ADD COLUMN driver_version VARCHAR")
    except Exception:
        pass  # Column already exists

    conn.execute("""
                 CREATE TABLE IF NOT EXISTS performance_checkpoints
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     DEFAULT
                     nextval
                 (
                     'checkpoints_seq'
                 ),
                     run_id INTEGER NOT NULL,
                     checkpoint_name VARCHAR NOT NULL,
                     elapsed_time DOUBLE NOT NULL,
                     FOREIGN KEY
                 (
                     run_id
                 ) REFERENCES performance_runs
                 (
                     id
                 )
                     )
                 """)

    # Create indexes for common queries
    conn.execute("""
                 CREATE INDEX IF NOT EXISTS idx_test_backend_branch
                     ON performance_runs(test_name, backend, branch, commit, file_status)
                 """)

    conn.execute("""
                 CREATE INDEX IF NOT EXISTS idx_timestamp
                     ON performance_runs(timestamp_unix DESC)
                 """)

    return conn


def get_git_info(file_path: str) -> dict[str, str]:
    """
    Extract git information for the current repository state.

    Args:
        file_path: Path to the test file being tracked

    Returns:
        Dictionary with keys: branch, commit, file_status (clean/modified/untracked)

    Note:
        file_status is "clean" ONLY if there are NO uncommitted changes or untracked files
        anywhere in the repository. This ensures "clean" results are fully replicable.
    """
    info = {
        "branch":      "unknown",
        "commit":      "unknown",
        "file_status": "unknown"
    }

    try:
        repo_dir = os.path.dirname(file_path)

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_dir
        )
        info["branch"] = result.stdout.strip()

        # Get current commit (short hash)
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_dir
        )
        info["commit"] = result.stdout.strip()

        # Check entire repository status (not just the file)
        # Status is "clean" ONLY if there are NO changes or untracked files anywhere
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_dir
        )

        # If there's ANY output, repository has uncommitted changes
        if not result.stdout.strip():
            info["file_status"] = "clean"
        else:
            # Check if there are untracked files
            has_untracked = any(line.startswith('??') for line in result.stdout.strip().split('\n'))
            info["file_status"] = "untracked" if has_untracked else "modified"

    except subprocess.CalledProcessError:
        # Git command failed - likely not in a git repo or git not available
        pass
    except Exception:
        # Other error - keep defaults
        pass

    return info


def save_performance_data(
        test_name: str,
        file_path: str,
        backend: str,
        timing_data: dict[str, float],
        metadata: dict | None = None,
        test_start_time: float | None = None,
        device_name: str | None = None,
        driver_version: str | None = None
) -> int:
    """
    Save performance data to DuckDB.

    Args:
        test_name: Full test name (e.g., "test_graphics4.TestClass.test_method")
        file_path: Absolute path to test file
        backend: Backend name (ocl, metal, numbax, cuda)
        timing_data: Dictionary of checkpoint names to elapsed times
        metadata: Optional additional metadata
        test_start_time: Unix timestamp when test started (optional, defaults to now)
        device_name: Optional device name (e.g., "Apple M1 Pro")
        driver_version: Optional driver version string

    Returns:
        Run ID of the inserted record
    """
    git_info = get_git_info(file_path)

    # Use provided test start time if available, otherwise use current time
    if test_start_time is not None:
        timestamp = datetime.fromtimestamp(test_start_time)
        timestamp_unix = test_start_time
    else:
        timestamp = datetime.now()
        timestamp_unix = time.time()

    # Calculate statistics
    total_time = sum(timing_data.values()) if timing_data else 0
    times = list(timing_data.values()) if timing_data else []

    stats = {}
    if times:
        stats = {
            "min_time":          min(times),
            "max_time":          max(times),
            "mean_time":         sum(times) / len(times),
            "slowest_operation": max(timing_data.items(), key=lambda x: x[1])[0],
            "fastest_operation": min(timing_data.items(), key=lambda x: x[1])[0],
        }

    conn = _get_connection()

    # Insert run
    result = conn.execute("""
                          INSERT INTO performance_runs (test_name, test_file, backend, device_name, driver_version,
                                                        branch, commit, file_status,
                                                        timestamp, timestamp_unix, total_time, min_time, max_time,
                                                        mean_time, slowest_operation, fastest_operation)
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
                          """, [
                              test_name,
                              os.path.basename(file_path),
                              backend,
                              device_name,
                              driver_version,
                              git_info["branch"],
                              git_info["commit"],
                              git_info["file_status"],
                              timestamp,
                              timestamp_unix,
                              total_time,
                              stats.get("min_time"),
                              stats.get("max_time"),
                              stats.get("mean_time"),
                              stats.get("slowest_operation"),
                              stats.get("fastest_operation"),
                          ]).fetchone()

    run_id = result[0]

    # Insert checkpoints
    for checkpoint_name, elapsed_time in timing_data.items():
        conn.execute("""
                     INSERT INTO performance_checkpoints (run_id, checkpoint_name, elapsed_time)
                     VALUES (?, ?, ?)
                     """, [run_id, checkpoint_name, elapsed_time])

    conn.close()
    return run_id


def load_performance_data(
        backend: str | None = None,
        test_name: str | None = None,
        branch: str | None = None,
        file_status: str | None = None,
        limit: int | None = None,
        exact_match: bool = False
) -> list[dict]:
    """
    Load and filter performance data from DuckDB.

    Args:
        backend: Filter by backend (ocl, metal, numbax, cuda)
        test_name: Filter by test name (supports partial match unless exact_match=True)
        branch: Filter by git branch
        file_status: Filter by file status (clean, modified)
        limit: Maximum number of results to return (most recent first)
        exact_match: If True, test_name must match exactly (default: False for partial match)

    Returns:
        list of performance data dictionaries, sorted by timestamp (newest first)
    """
    conn = _get_connection()

    # Build query
    query = """
            SELECT r.id,
                   r.test_name,
                   r.test_file,
                   r.backend,
                   r.device_name,
                   r.driver_version,
                   r.branch,
                   r.commit,
                   r.file_status,
                   r.timestamp,
                   r.timestamp_unix,
                   r.total_time,
                   r.min_time,
                   r.max_time,
                   r.mean_time,
                   r.slowest_operation,
                   r.fastest_operation
            FROM performance_runs r
            WHERE 1 = 1 \
            """
    params = []

    if backend:
        query += " AND r.backend = ?"
        params.append(backend)

    if test_name:
        if exact_match:
            query += " AND r.test_name = ?"
            params.append(test_name)
        else:
            query += " AND r.test_name LIKE ?"
            params.append(f"%{test_name}%")

    if branch:
        query += " AND r.branch = ?"
        params.append(branch)

    if file_status:
        query += " AND r.file_status = ?"
        params.append(file_status)

    query += " ORDER BY r.timestamp_unix DESC"

    if limit:
        query += f" LIMIT {limit}"

    results = conn.execute(query, params).fetchall()

    # Load checkpoint data for each run
    data = []
    for row in results:
        run_id = row[0]

        # Get checkpoints for this run
        checkpoints_result = conn.execute("""
                                          SELECT checkpoint_name, elapsed_time
                                          FROM performance_checkpoints
                                          WHERE run_id = ?
                                          """, [run_id]).fetchall()

        checkpoints = {name: time for name, time in checkpoints_result}

        # Build data structure
        entry = {
            "metadata": {
                "test_name":      row[1],
                "test_file":      row[2],
                "backend":        row[3],
                "device_name":    row[4],
                "driver_version": row[5],
                "branch":         row[6],
                "commit":         row[7],
                "file_status":    row[8],
                "timestamp":      row[9].isoformat(),
                "timestamp_unix": row[10],
            },
            "timing":   {
                "checkpoints": checkpoints,
                "total":       row[11],
            },
            "stats":    {
                "min":               row[12],
                "max":               row[13],
                "mean":              row[14],
                "slowest_operation": row[15],
                "fastest_operation": row[16],
            }
        }

        data.append(entry)

    conn.close()
    return data


def purge_performance_data(
        older_than_days: int | None = None,
        backend: str | None = None,
        test_name: str | None = None,
        dry_run: bool = False
) -> tuple[int, list[str]]:
    """
    Delete old performance data from DuckDB.

    Args:
        older_than_days: Delete data older than N days
        backend: Only delete data for specific backend
        test_name: Only delete data for specific test (partial match)
        dry_run: If True, don't actually delete, just return what would be deleted

    Returns:
        tuple of (count_deleted, list_of_descriptions)
    """
    conn = _get_connection()

    # Build query
    query = "SELECT id, test_name, backend, timestamp FROM performance_runs WHERE 1=1"
    params = []

    if older_than_days:
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        query += " AND timestamp_unix < ?"
        params.append(cutoff_time)

    if backend:
        query += " AND backend = ?"
        params.append(backend)

    if test_name:
        query += " AND test_name LIKE ?"
        params.append(f"%{test_name}%")

    results = conn.execute(query, params).fetchall()

    deleted_descriptions = []
    for row in results:
        run_id, test, backend, timestamp = row
        deleted_descriptions.append(f"{test} ({backend}) - {timestamp}")

    if not dry_run and results:
        run_ids = [row[0] for row in results]

        # Delete checkpoints first (foreign key constraint)
        conn.execute(f"""
            DELETE FROM performance_checkpoints
            WHERE run_id IN ({','.join('?' * len(run_ids))})
        """, run_ids)

        # Delete runs
        conn.execute(f"""
            DELETE FROM performance_runs
            WHERE id IN ({','.join('?' * len(run_ids))})
        """, run_ids)

    conn.close()
    return len(results), deleted_descriptions


def generate_summary_table(data: list[dict]) -> str:
    """
    Generate a summary table of performance data using tabulate.

    Args:
        data: list of performance data dictionaries

    Returns:
        Formatted table string
    """
    try:
        from tabulate import tabulate
    except ImportError:
        return "Error: tabulate not installed. Run: pip install tabulate"

    if not data:
        return "No performance data found."

    # Build table rows with unique combinations only
    headers = ["Test Name", "Backend", "Device", "Branch", "Commit", "Status", "Total (ms)", "Timestamp"]
    rows = []
    seen = set()

    for entry in data:
        meta = entry["metadata"]
        timing = entry["timing"]

        # Create a unique key for deduplication
        key = (
            meta["test_name"],
            meta["backend"],
            meta.get("device_name", "N/A"),
            meta["timestamp"]
        )

        if key in seen:
            continue
        seen.add(key)

        rows.append([
            meta["test_name"],
            meta["backend"],
            meta.get("device_name", "N/A"),
            meta["branch"],
            meta["commit"][:7],  # Short commit hash
            meta["file_status"],
            f"{timing['total'] * 1000:.2f}",  # Convert to milliseconds
            datetime.fromisoformat(meta["timestamp"]).strftime("%Y-%m-%d %H:%M")
        ])

    return tabulate(rows, headers=headers, tablefmt="github")


def generate_checkpoint_comparison_table(data: list[dict]) -> str:
    """
    Generate a table comparing checkpoints across multiple runs.

    Args:
        data: list of performance data dictionaries (should be same test, different runs)

    Returns:
        Formatted table string
    """
    try:
        from tabulate import tabulate
    except ImportError:
        return "Error: tabulate not installed. Run: pip install tabulate"

    if not data:
        return "No performance data found."

    # Get all checkpoint names
    checkpoint_names = set()
    for entry in data:
        checkpoint_names.update(entry["timing"]["checkpoints"].keys())

    checkpoint_names = sorted(checkpoint_names)

    # Build table
    headers = ["Checkpoint"] + [
        f"{entry['metadata']['backend']} ({entry['metadata']['commit'][:7]})"
        for entry in data
    ]
    rows = []

    for checkpoint in checkpoint_names:
        row = [checkpoint]
        for entry in data:
            time_val = entry["timing"]["checkpoints"].get(checkpoint, 0)
            row.append(f"{time_val:.3f}s")
        rows.append(row)

    # Add total row
    total_row = ["TOTAL"]
    for entry in data:
        total_row.append(f"{entry['timing']['total']:.3f}s")
    rows.append(total_row)

    return tabulate(rows, headers=headers, tablefmt="github")


def plot_backend_comparison(
        data: list[dict],
        output_path: Path | None = None,
        checkpoint_name: str | None = None
) -> Path | None:
    """
    Generate a bar chart comparing performance across backends.

    Args:
        data: list of performance data dictionaries
        output_path: Path to save the plot (default: REPORTS_DIR/backend_comparison.png)
        checkpoint_name: Specific checkpoint to compare (default: total time)

    Returns:
        Path to saved plot, or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Error: matplotlib and seaborn required. Run: pip install matplotlib seaborn")
        return None

    if not data:
        print("No data to plot")
        return None

    # set style
    sns.set_style("whitegrid")

    # Group data by backend
    backend_times = {}
    for entry in data:
        backend = entry["metadata"]["backend"]
        if checkpoint_name:
            time_val = entry["timing"]["checkpoints"].get(checkpoint_name, 0)
        else:
            time_val = entry["timing"]["total"]

        if backend not in backend_times:
            backend_times[backend] = []
        backend_times[backend].append(time_val)

    # Calculate mean and std for each backend
    backends = sorted(backend_times.keys())
    means = [sum(backend_times[b]) / len(backend_times[b]) for b in backends]
    stds = []
    for b in backends:
        times = backend_times[b]
        if len(times) > 1:
            mean = sum(times) / len(times)
            variance = sum((t - mean) ** 2 for t in times) / len(times)
            stds.append(variance ** 0.5)
        else:
            stds.append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = range(len(backends))

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8)

    # Color bars
    colors = sns.color_palette("husl", len(backends))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel("Backend", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    title = f"Backend Comparison"
    if checkpoint_name:
        title += f" - {checkpoint_name}"
    else:
        title += " - Total Time"
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(backends)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        label = f"{mean:.3f}s"
        if std > 0:
            label += f"\n±{std:.3f}s"
        ax.text(i, mean, label, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = REPORTS_DIR / "backend_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_longitudinal_trend(
        data: list[dict],
        output_path: Path | None = None,
        checkpoint_name: str | None = None
) -> Path | None:
    """
    Generate a line graph showing performance trends over time.

    Args:
        data: list of performance data dictionaries (sorted by time)
        output_path: Path to save the plot (default: REPORTS_DIR/longitudinal_trend.png)
        checkpoint_name: Specific checkpoint to track (default: total time)

    Returns:
        Path to saved plot, or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import seaborn as sns
    except ImportError:
        print("Error: matplotlib and seaborn required. Run: pip install matplotlib seaborn")
        return None

    if not data:
        print("No data to plot")
        return None

    # set style
    sns.set_style("whitegrid")

    # Group data by backend and sort by timestamp
    backend_data = {}
    for entry in data:
        backend = entry["metadata"]["backend"]
        timestamp = datetime.fromisoformat(entry["metadata"]["timestamp"])
        if checkpoint_name:
            time_val = entry["timing"]["checkpoints"].get(checkpoint_name, 0)
        else:
            time_val = entry["timing"]["total"]

        if backend not in backend_data:
            backend_data[backend] = {"timestamps": [], "times": []}

        backend_data[backend]["timestamps"].append(timestamp)
        backend_data[backend]["times"].append(time_val)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("husl", len(backend_data))

    for (backend, bd), color in zip(sorted(backend_data.items()), colors):
        ax.plot(bd["timestamps"], bd["times"], marker='o', label=backend,
                color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Date/Time", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    title = "Performance Trend Over Time"
    if checkpoint_name:
        title += f" - {checkpoint_name}"
    else:
        title += " - Total Time"
    ax.set_title(title, fontsize=14)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = REPORTS_DIR / "longitudinal_trend.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def plot_checkpoint_breakdown(
        data: dict,
        output_path: Path | None = None,
        plot_type: str = "bar"
) -> Path | None:
    """
    Generate a breakdown chart of checkpoint times for a single test run.

    Args:
        data: Single performance data dictionary
        output_path: Path to save the plot (default: REPORTS_DIR/checkpoint_breakdown.png)
        plot_type: Type of plot ("bar" or "pie")

    Returns:
        Path to saved plot, or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Error: matplotlib and seaborn required. Run: pip install matplotlib seaborn")
        return None

    checkpoints = data["timing"]["checkpoints"]
    if not checkpoints:
        print("No checkpoint data to plot")
        return None

    # set style
    sns.set_style("whitegrid")

    # Sort checkpoints by time
    sorted_checkpoints = sorted(checkpoints.items(), key=lambda x: x[1], reverse=True)
    names = [name for name, _ in sorted_checkpoints]
    times = [time for _, time in sorted_checkpoints]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == "pie":
        colors = sns.color_palette("husl", len(names))
        wedges, texts, autotexts = ax.pie(
            times,
            labels=names,
            autopct=lambda pct: f"{pct:.1f}%\n({pct / 100 * sum(times):.3f}s)",
            colors=colors,
            startangle=90
        )
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_weight('bold')
        ax.set_title(
            f"Checkpoint Breakdown - {data['metadata']['test_name']}\n"
            f"{data['metadata']['backend']} - Total: {sum(times):.3f}s",
            fontsize=12
        )
    else:  # bar chart
        colors = sns.color_palette("husl", len(names))
        bars = ax.barh(range(len(names)), times, color=colors, alpha=0.8)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Time (seconds)", fontsize=12)
        ax.set_title(
            f"Checkpoint Breakdown - {data['metadata']['test_name']}\n"
            f"{data['metadata']['backend']} - Total: {sum(times):.3f}s",
            fontsize=12
        )

        # Add value labels
        for i, (time, bar) in enumerate(zip(times, bars)):
            percentage = (time / sum(times) * 100)
            ax.text(time, i, f" {time:.3f}s ({percentage:.1f}%)",
                    va='center', fontsize=9)

        ax.invert_yaxis()  # Highest value on top

    plt.tight_layout()

    # Save plot
    if output_path is None:
        output_path = REPORTS_DIR / "checkpoint_breakdown.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def generate_backend_comparison_report(
        test_name: str,
        output_path: Path | None = None
) -> Path:
    """
    Generate a comprehensive markdown report comparing backend performance.

    Args:
        test_name: Test name to filter by (partial match)
        output_path: Path to save the report (default: REPORTS_DIR/backend_comparison_report.md)

    Returns:
        Path to saved report
    """
    # Load data for this test (exact match to avoid partial matches like "anisodiff.I" matching "anisodiff.IS")
    data = load_performance_data(test_name=test_name, limit=100, exact_match=True)

    if not data:
        raise ValueError(f"No performance data found for test: {test_name}")

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"backend_comparison_{timestamp}.md"

    # Generate visualizations
    plot_path = REPORTS_DIR / f"backend_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_backend_comparison(data, output_path=plot_path)

    # Generate markdown report
    lines = [
        f"# Backend Comparison Report: {test_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Test: `{test_name}`",
        f"- Total runs analyzed: {len(data)}",
        f"- Backends compared: {', '.join(sorted(set(d['metadata']['backend'] for d in data)))}",
        "",
        "## Performance Comparison",
        "",
        f"![Backend Comparison]({plot_path.name})",
        "",
        "## Detailed Results",
        "",
        f"{generate_summary_table(data)}",
        "",
        "## Checkpoint Breakdown",
        "",
        f"{generate_checkpoint_comparison_table(data[:5])}",  # Top 5 most recent runs
        "",
        "## Analysis",
        "",
        f"This report compares performance across different backends for the test `{test_name}`.",
        "The bar chart shows mean execution times with standard deviation error bars.",
        "Lower execution time indicates better performance.",
    ]

    # Add performance statistics
    backend_stats = {}
    for entry in data:
        backend = entry["metadata"]["backend"]
        total_time = entry["timing"]["total"]
        if backend not in backend_stats:
            backend_stats[backend] = []
        backend_stats[backend].append(total_time)

    if backend_stats:
        lines.append("")
        lines.append("### Backend Statistics")
        lines.append("")
        for backend in sorted(backend_stats.keys()):
            times = backend_stats[backend]
            mean = sum(times) / len(times)
            if len(times) > 1:
                variance = sum((t - mean) ** 2 for t in times) / len(times)
                std = variance ** 0.5
                lines.append(f"- **{backend}**: {mean:.3f}s ± {std:.3f}s ({len(times)} runs)")
            else:
                lines.append(f"- **{backend}**: {mean:.3f}s ({len(times)} run)")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


def generate_longitudinal_report(
        test_name: str,
        backend: str | None = None,
        output_path: Path | None = None
) -> Path:
    """
    Generate a markdown report showing performance trends over time.

    Args:
        test_name: Test name to filter by (partial match)
        backend: Optional backend to filter by
        output_path: Path to save the report (default: REPORTS_DIR/longitudinal_report.md)

    Returns:
        Path to saved report
    """
    # Load data (exact match)
    data = load_performance_data(test_name=test_name, backend=backend, limit=100, exact_match=True)

    if not data:
        raise ValueError(f"No performance data found for test: {test_name}")

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"longitudinal_{timestamp}.md"

    # Generate visualizations
    plot_path = REPORTS_DIR / f"longitudinal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_longitudinal_trend(data, output_path=plot_path)

    # Build report
    lines = [
        f"# Longitudinal Performance Report: {test_name}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Test: `{test_name}`",
    ]

    if backend:
        lines.append(f"- Backend: {backend}")

    lines.extend([
        f"- Total runs analyzed: {len(data)}",
        f"- Time range: {datetime.fromisoformat(data[-1]['metadata']['timestamp']).strftime('%Y-%m-%d')} to {datetime.fromisoformat(data[0]['metadata']['timestamp']).strftime('%Y-%m-%d')}",
        "",
        "## Performance Trend",
        "",
        f"![Longitudinal Trend]({plot_path.name})",
        "",
        "## Recent Results",
        "",
        f"{generate_summary_table(data[:10])}",
        "",
    ])

    # Regression detection
    lines.append("## Regression Analysis")
    lines.append("")

    # Group by backend
    backend_data = {}
    for entry in data:
        b = entry["metadata"]["backend"]
        if b not in backend_data:
            backend_data[b] = []
        backend_data[b].append((
            datetime.fromisoformat(entry["metadata"]["timestamp"]),
            entry["timing"]["total"]
        ))

    for b in sorted(backend_data.keys()):
        runs = sorted(backend_data[b], key=lambda x: x[0])  # Sort by timestamp
        if len(runs) >= 3:
            # Compare recent vs historical
            recent_times = [t for _, t in runs[-3:]]
            historical_times = [t for _, t in runs[:-3]] if len(runs) > 3 else recent_times

            recent_mean = sum(recent_times) / len(recent_times)
            historical_mean = sum(historical_times) / len(historical_times)

            change_pct = ((recent_mean - historical_mean) / historical_mean * 100) if historical_mean > 0 else 0

            if abs(change_pct) > 10:
                direction = "slower" if change_pct > 0 else "faster"
                lines.append(
                    f"- **{b}**: Recent performance is {abs(change_pct):.1f}% {direction} "
                    f"({recent_mean:.3f}s vs {historical_mean:.3f}s)"
                )
            else:
                lines.append(f"- **{b}**: Performance stable ({recent_mean:.3f}s)")
        else:
            lines.append(f"- **{b}**: Insufficient data for regression analysis ({len(runs)} runs)")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


def generate_comprehensive_report(
        test_name: str | None = None,
        output_path: Path | None = None
) -> Path:
    """
    Generate a comprehensive performance report with all available data.

    Args:
        test_name: Optional test name to filter by
        output_path: Path to save the report (default: REPORTS_DIR/comprehensive_report.md)

    Returns:
        Path to saved report
    """
    try:
        from tabulate import tabulate
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib, seaborn, and tabulate required for comprehensive reports")

    # Load all data or filtered by test name
    data = load_performance_data(test_name=test_name, limit=1000)

    if not data:
        raise ValueError("No performance data found")

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"comprehensive_{timestamp}.md"

    # Aggregate data by (test_name, backend, branch, commit, status)
    aggregated = {}
    for entry in data:
        meta = entry["metadata"]
        key = (
            meta["test_name"],
            meta["backend"],
            meta["branch"],
            meta["commit"],
            meta["file_status"]
        )

        if key not in aggregated:
            aggregated[key] = {
                "times":       [],
                "metadata":    meta,
                "checkpoints": {}
            }

        aggregated[key]["times"].append(entry["timing"]["total"])

        # Aggregate checkpoint data
        for cp_name, cp_time in entry["timing"]["checkpoints"].items():
            if cp_name not in aggregated[key]["checkpoints"]:
                aggregated[key]["checkpoints"][cp_name] = []
            aggregated[key]["checkpoints"][cp_name].append(cp_time)

    # Calculate aggregated statistics
    agg_data = []
    for key, values in aggregated.items():
        test_name_key, backend, branch, commit, status = key
        times = values["times"]
        mean_time = sum(times) / len(times)

        if len(times) > 1:
            variance = sum((t - mean_time) ** 2 for t in times) / len(times)
            std_time = variance ** 0.5
        else:
            std_time = 0.0

        agg_data.append({
            "test_name":   test_name_key,
            "backend":     backend,
            "branch":      branch,
            "commit":      commit,
            "status":      status,
            "mean":        mean_time,
            "std":         std_time,
            "runs":        len(times),
            "checkpoints": {k: sum(v) / len(v) for k, v in values["checkpoints"].items()}
        })

    # Sort by test name, then backend
    agg_data.sort(key=lambda x: (x["test_name"], x["backend"]))

    # Build aggregated table
    table_rows = []
    for item in agg_data:
        if item["std"] > 0:
            time_str = f"{item['mean']:.3f} ± {item['std']:.3f}"
        else:
            time_str = f"{item['mean']:.3f}"

        table_rows.append([
            item["test_name"],
            item["backend"],
            item["branch"],
            item["commit"][:7],
            item["status"],
            time_str,
            item["runs"]
        ])

    aggregated_table = tabulate(
        table_rows,
        headers=["Test Name", "Backend", "Branch", "Commit", "Status", "Time (s)", "Runs"],
        tablefmt="github"
    )

    # Generate comparison graph across all tests
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Group by test for graphing
    test_backend_data = {}
    for item in agg_data:
        tn = item["test_name"]
        if tn not in test_backend_data:
            test_backend_data[tn] = {}
        test_backend_data[tn][item["backend"]] = item["mean"]

    # Create grouped bar chart
    sns.set_style("whitegrid")

    # Only plot tests that have multiple backends for comparison
    multi_backend_tests = {k: v for k, v in test_backend_data.items() if len(v) > 1}

    if multi_backend_tests:
        # Limit to first 20 tests for readability
        plot_tests = dict(list(multi_backend_tests.items())[:20])

        backends = sorted(set(b for test_data in plot_tests.values() for b in test_data.keys()))
        test_names = list(plot_tests.keys())

        fig, ax = plt.subplots(figsize=(max(12, len(test_names) * 0.5), 8))

        x = range(len(test_names))
        width = 0.8 / len(backends)
        colors = sns.color_palette("husl", len(backends))

        for i, backend in enumerate(backends):
            times = [plot_tests[tn].get(backend, 0) for tn in test_names]
            offset = (i - len(backends) / 2) * width + width / 2
            ax.bar([pos + offset for pos in x], times, width, label=backend, color=colors[i], alpha=0.8)

        ax.set_xlabel("Test", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title("Performance Comparison Across Tests and Backends", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([tn.split('.')[-1] if '.' in tn else tn for tn in test_names],
                           rotation=45, ha='right')
        ax.legend(title="Backend")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        graph_path = REPORTS_DIR / f"comprehensive_comparison_{timestamp_str}.png"
        plt.savefig(graph_path, dpi=150)
        plt.close()
    else:
        graph_path = None

    # Build report
    lines = [
        "# Comprehensive Performance Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- Total runs analyzed: {len(data)}",
        f"- Aggregated results: {len(agg_data)}",
        f"- Backends: {', '.join(sorted(set(d['backend'] for d in agg_data)))}",
        f"- Tests: {len(set(d['test_name'] for d in agg_data))}",
    ]

    if test_name:
        lines.append(f"- Filtered by test: `{test_name}`")

    if graph_path:
        lines.extend([
            "",
            "## Performance Comparison",
            "",
            f"![Performance Comparison]({graph_path.name})",
        ])

    lines.extend([
        "",
        "## Aggregated Results",
        "",
        "Results are aggregated by (Test Name, Backend, Branch, Commit, Status).",
        "Multiple runs with the same key are averaged together.",
        "",
        aggregated_table,
        "",
        "## Performance by Test",
        "",
    ])

    # Group by test name for detailed breakdown
    test_groups = {}
    for item in agg_data:
        tn = item["test_name"]
        if tn not in test_groups:
            test_groups[tn] = []
        test_groups[tn].append(item)

    for tn in sorted(test_groups.keys()):
        test_items = test_groups[tn]
        lines.append(f"### {tn}")
        lines.append("")

        for item in test_items:
            if item["std"] > 0:
                lines.append(
                    f"- **{item['backend']}** ({item['status']}): {item['mean']:.3f}s ± {item['std']:.3f}s ({item['runs']} runs)")
            else:
                lines.append(f"- **{item['backend']}** ({item['status']}): {item['mean']:.3f}s ({item['runs']} run)")
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "- Performance data is tracked automatically when tests use `StopWatcher(save_perf=True)`",
        "- Results are aggregated by test name, backend, branch, commit, and file status",
        "- Times shown are mean ± standard deviation across multiple runs",
        "- Use `scripts/performance_report.py compare <test>` for detailed backend comparisons",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


def generate_backend_cross_comparison_report(
        test_pattern: str | None = None,
        branch: str | None = None,
        component: str | None = None,
        limit: int = 100,
        output_path: Path | None = None
) -> Path:
    """
    Generate a comprehensive cross-backend comparison report.

    This report compares performance across all available backends (ocl, metal, cuda, numbax)
    for tests matching the pattern, grouped by test name.

    Args:
        test_pattern: Optional test name pattern to filter by (partial match)
        branch: Optional git branch to filter by
        component: Optional component name (e.g., 'graphics4' matches all tests with graphics4 in name/path)
        limit: Maximum number of results per backend (default: 100)
        output_path: Path to save the report (default: REPORTS_DIR/backend_cross_comparison_<timestamp>.md)

    Returns:
        Path to saved report
    """
    try:
        from tabulate import tabulate
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib, seaborn, and tabulate required for backend comparison reports")

    # Load data from database
    conn = _get_connection()

    # Build query
    query = """
            SELECT test_name,
                   backend,
                   device_name,
                   branch, commit, file_status, total_time, timestamp
            FROM performance_runs
            WHERE 1=1 \
            """

    params = []
    if test_pattern:
        query += " AND test_name LIKE ?"
        params.append(f"%{test_pattern}%")

    if component:
        # Component matches anywhere in test_name or test_file
        query += " AND (test_name LIKE ? OR test_file LIKE ?)"
        params.append(f"%{component}%")
        params.append(f"%{component}%")

    if branch:
        query += " AND branch = ?"
        params.append(branch)

    query += " ORDER BY test_name, backend, timestamp DESC"

    result = conn.execute(query, params).fetchall()
    conn.close()

    if not result:
        raise ValueError(
            f"No performance data found (test_pattern={test_pattern}, component={component}, branch={branch})")

    # Convert to dict format
    data = []
    for row in result:
        data.append({
            "metadata": {
                "test_name":   row[0],
                "backend":     row[1],
                "device_name": row[2] or "N/A",
                "branch":      row[3],
                "commit":      row[4],
                "file_status": row[5],
                "timestamp":   row[7].isoformat() if hasattr(row[7], 'isoformat') else str(row[7])
            },
            "timing":   {
                "total": row[6]
            }
        })

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"backend_cross_comparison_{timestamp}.md"

    # Group by test name and backend
    test_backend_data = {}
    for entry in data:
        test_name = entry["metadata"]["test_name"]
        backend = entry["metadata"]["backend"]

        if test_name not in test_backend_data:
            test_backend_data[test_name] = {}

        if backend not in test_backend_data[test_name]:
            test_backend_data[test_name][backend] = []

        test_backend_data[test_name][backend].append(entry["timing"]["total"])

    # Calculate statistics
    test_stats = {}
    for test_name, backends in test_backend_data.items():
        test_stats[test_name] = {}
        for backend, times in backends.items():
            mean = sum(times) / len(times)
            if len(times) > 1:
                variance = sum((t - mean) ** 2 for t in times) / len(times)
                std = variance ** 0.5
            else:
                std = 0.0

            test_stats[test_name][backend] = {
                "mean": mean,
                "std":  std,
                "min":  min(times),
                "max":  max(times),
                "runs": len(times)
            }

    # Create comparison table
    all_backends = sorted(set(b for backends in test_backend_data.values() for b in backends.keys()))

    table_rows = []
    for test_name in sorted(test_stats.keys()):
        backend_stats = test_stats[test_name]

        # Find baseline (fastest backend)
        baseline_backend = min(backend_stats.keys(), key=lambda b: backend_stats[b]["mean"])
        baseline_mean = backend_stats[baseline_backend]["mean"]

        row = [test_name]
        for backend in all_backends:
            if backend in backend_stats:
                stats = backend_stats[backend]
                try:
                    speedup = baseline_mean / stats["mean"]

                    # Format: mean±std (speedup) [runs]
                    if stats["std"] > 0:
                        cell = f"{stats['mean'] * 1000:.2f}±{stats['std'] * 1000:.2f}ms"
                    else:
                        cell = f"{stats['mean'] * 1000:.2f}ms"

                    if speedup != 1.0:
                        cell += f" ({speedup:.2f}x)"
                    else:
                        cell += " (baseline)"

                    cell += f" [{stats['runs']}]"
                    row.append(cell)
                except (KeyError, ZeroDivisionError):
                    row.append("N/A")
            else:
                row.append("N/A")

        table_rows.append(row)

    comparison_table = tabulate(
        table_rows,
        headers=["Test"] + all_backends,
        tablefmt="github"
    )

    # Create visualization
    sns.set_style("whitegrid")

    # Limit to first 15 tests for readability
    plot_tests = dict(list(test_stats.items())[:15])

    if len(plot_tests) > 0:
        test_names = list(plot_tests.keys())
        fig, ax = plt.subplots(figsize=(max(12, len(test_names) * 0.6), 8))

        x = range(len(test_names))
        width = 0.8 / len(all_backends)
        colors = sns.color_palette("husl", len(all_backends))

        for i, backend in enumerate(all_backends):
            means = []
            stds = []
            for tn in test_names:
                if backend in plot_tests[tn]:
                    means.append(plot_tests[tn][backend]["mean"] * 1000)  # Convert to ms
                    stds.append(plot_tests[tn][backend]["std"] * 1000)
                else:
                    means.append(0)
                    stds.append(0)

            offset = (i - len(all_backends) / 2) * width + width / 2
            ax.bar(
                [pos + offset for pos in x],
                means,
                width,
                yerr=stds,
                label=backend,
                color=colors[i],
                alpha=0.8,
                capsize=3
            )

        ax.set_xlabel("Test", fontsize=12)
        ax.set_ylabel("Time (milliseconds)", fontsize=12)
        ax.set_title("Backend Performance Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [tn.split('.')[-1] if '.' in tn else tn for tn in test_names],
            rotation=45,
            ha='right'
        )
        ax.legend(title="Backend", loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = REPORTS_DIR / f"backend_cross_comparison_{timestamp_str}.png"
        plt.savefig(graph_path, dpi=150)
        plt.close()
    else:
        graph_path = None

    # Build markdown report
    lines = [
        "# Backend Cross-Comparison Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Tests analyzed: {len(test_stats)}",
        f"- Backends: {', '.join(all_backends)}",
        f"- Total runs: {sum(sum(len(times) for times in backends.values()) for backends in test_backend_data.values())}",
    ]

    if test_pattern:
        lines.append(f"- Filtered by pattern: `{test_pattern}`")
    if component:
        lines.append(f"- Filtered by component: `{component}`")
    if branch:
        lines.append(f"- Filtered by branch: `{branch}`")

    lines.extend([
        "",
        "## Performance Comparison",
        "",
        "Times shown as: mean±std (speedup vs baseline) [number of runs]",
        "",
    ])

    if graph_path:
        lines.append(f"![Backend Comparison]({graph_path.name})")
        lines.append("")

    lines.extend([
        "## Results Table",
        "",
        comparison_table,
        "",
        "## Backend Summary",
        "",
    ])

    # Overall backend statistics
    backend_totals = {}
    for test_name, backends in test_stats.items():
        for backend, stats in backends.items():
            if backend not in backend_totals:
                backend_totals[backend] = []
            backend_totals[backend].append(stats["mean"])

    for backend in sorted(backend_totals.keys()):
        times = backend_totals[backend]
        mean = sum(times) / len(times)
        lines.append(f"- **{backend}**: Average {mean * 1000:.2f}ms across {len(times)} tests")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Speedup**: Ratio of baseline (fastest) backend time to current backend time",
        "- **Baseline**: The fastest backend for each test (shown as speedup 1.0x)",
        "- **N/A**: Backend has no performance data for this test",
        "- Lower times and higher speedups indicate better performance",
        "",
        "## Notes",
        "",
        "- Results are averaged across multiple runs when available",
        "- Standard deviation shows consistency (lower is more consistent)",
        "- Compare backends on same hardware for fairest results",
        "- Different tests may favor different backends based on workload characteristics",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


def generate_branch_comparison_report(
        test_pattern: str | None = None,
        backend: str | None = None,
        component: str | None = None,
        branches: list[str] | None = None,
        limit: int = 100,
        output_path: Path | None = None
) -> Path:
    """
    Generate a comprehensive cross-branch comparison report.

    This report compares performance across different git branches for the same tests,
    allowing you to see how performance changes across branches (e.g., main vs feature branches).

    Args:
        test_pattern: Optional test name pattern to filter by (partial match)
        backend: Optional backend to filter by (if None, compares all backends)
        component: Optional component name (e.g., 'graphics4' matches all tests with graphics4 in name/path)
        branches: Optional list of specific branches to compare (if None, uses all branches)
        limit: Maximum number of results per branch (default: 100)
        output_path: Path to save the report (default: REPORTS_DIR/branch_comparison_<timestamp>.md)

    Returns:
        Path to saved report
    """
    try:
        from tabulate import tabulate
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("matplotlib, seaborn, and tabulate required for branch comparison reports")

    # Load data from database
    conn = _get_connection()

    # Build query
    query = """
            SELECT test_name,
                   backend,
                   branch, commit, file_status, total_time, timestamp
            FROM performance_runs
            WHERE 1=1 \
            """

    params = []
    if test_pattern:
        query += " AND test_name LIKE ?"
        params.append(f"%{test_pattern}%")

    if component:
        query += " AND (test_name LIKE ? OR test_file LIKE ?)"
        params.append(f"%{component}%")
        params.append(f"%{component}%")

    if backend:
        query += " AND backend = ?"
        params.append(backend)

    if branches:
        placeholders = ','.join(['?' for _ in branches])
        query += f" AND branch IN ({placeholders})"
        params.extend(branches)

    query += " ORDER BY test_name, branch, backend, timestamp DESC"

    result = conn.execute(query, params).fetchall()
    conn.close()

    if not result:
        raise ValueError(
            f"No performance data found (test_pattern={test_pattern}, component={component}, backend={backend}, branches={branches})")

    # Convert to dict format
    data = []
    for row in result:
        data.append({
            "metadata": {
                "test_name":   row[0],
                "backend":     row[1],
                "branch":      row[2],
                "commit":      row[3],
                "file_status": row[4],
                "timestamp":   row[6].isoformat() if hasattr(row[6], 'isoformat') else str(row[6])
            },
            "timing":   {
                "total": row[5]
            }
        })

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"branch_comparison_{timestamp}.md"

    # Group by test name, branch, and backend
    test_branch_backend_data = {}
    for entry in data:
        test_name = entry["metadata"]["test_name"]
        branch = entry["metadata"]["branch"]
        backend_name = entry["metadata"]["backend"]

        if test_name not in test_branch_backend_data:
            test_branch_backend_data[test_name] = {}

        if branch not in test_branch_backend_data[test_name]:
            test_branch_backend_data[test_name][branch] = {}

        if backend_name not in test_branch_backend_data[test_name][branch]:
            test_branch_backend_data[test_name][branch][backend_name] = []

        test_branch_backend_data[test_name][branch][backend_name].append(entry["timing"]["total"])

    # Calculate statistics
    test_stats = {}
    for test_name, branches_data in test_branch_backend_data.items():
        test_stats[test_name] = {}
        for branch_name, backends_data in branches_data.items():
            test_stats[test_name][branch_name] = {}
            for backend_name, times in backends_data.items():
                mean = sum(times) / len(times)
                if len(times) > 1:
                    variance = sum((t - mean) ** 2 for t in times) / len(times)
                    std = variance ** 0.5
                else:
                    std = 0.0

                test_stats[test_name][branch_name][backend_name] = {
                    "mean": mean,
                    "std":  std,
                    "min":  min(times),
                    "max":  max(times),
                    "runs": len(times)
                }

    # Get all branches and backends
    all_branches = sorted(set(b for test_data in test_branch_backend_data.values() for b in test_data.keys()))
    all_backends = sorted(set(
        be for test_data in test_branch_backend_data.values()
        for branch_data in test_data.values()
        for be in branch_data.keys()
    ))

    # Create comparison table
    # If single backend, show branches as columns; if multiple backends, show backend:branch combinations
    if len(all_backends) == 1:
        # Simple: columns are branches
        table_headers = ["Test"] + all_branches
        table_rows = []

        for test_name in sorted(test_stats.keys()):
            branch_stats = test_stats[test_name]
            backend_name = all_backends[0]

            # Find baseline (fastest branch)
            valid_branches = [b for b in all_branches if b in branch_stats and backend_name in branch_stats[b]]
            if not valid_branches:
                continue

            baseline_branch = min(valid_branches, key=lambda b: branch_stats[b][backend_name]["mean"])
            baseline_mean = branch_stats[baseline_branch][backend_name]["mean"]

            row = [test_name]
            for branch_name in all_branches:
                if branch_name in branch_stats and backend_name in branch_stats[branch_name]:
                    stats = branch_stats[branch_name][backend_name]
                    speedup = baseline_mean / stats["mean"]

                    if stats["std"] > 0:
                        cell = f"{stats['mean'] * 1000:.2f}±{stats['std'] * 1000:.2f}ms"
                    else:
                        cell = f"{stats['mean'] * 1000:.2f}ms"

                    if speedup != 1.0:
                        cell += f" ({speedup:.2f}x)"
                    else:
                        cell += " (baseline)"

                    cell += f" [{stats['runs']}]"
                    row.append(cell)
                else:
                    row.append("N/A")

            table_rows.append(row)

    else:
        # Complex: columns are backend:branch combinations
        table_headers = ["Test"]
        for backend_name in all_backends:
            for branch_name in all_branches:
                table_headers.append(f"{backend_name}:{branch_name}")

        table_rows = []

        for test_name in sorted(test_stats.keys()):
            branch_stats = test_stats[test_name]

            # Find global baseline (fastest backend:branch combination)
            all_times = []
            for branch_name in all_branches:
                if branch_name in branch_stats:
                    for backend_name in all_backends:
                        if backend_name in branch_stats[branch_name]:
                            all_times.append(branch_stats[branch_name][backend_name]["mean"])

            if not all_times:
                continue

            baseline_mean = min(all_times)

            row = [test_name]
            for backend_name in all_backends:
                for branch_name in all_branches:
                    if branch_name in branch_stats and backend_name in branch_stats[branch_name]:
                        stats = branch_stats[branch_name][backend_name]
                        speedup = baseline_mean / stats["mean"]

                        if stats["std"] > 0:
                            cell = f"{stats['mean'] * 1000:.2f}±{stats['std'] * 1000:.2f}ms"
                        else:
                            cell = f"{stats['mean'] * 1000:.2f}ms"

                        if speedup != 1.0:
                            cell += f" ({speedup:.2f}x)"
                        else:
                            cell += " (baseline)"

                        cell += f" [{stats['runs']}]"
                        row.append(cell)
                    else:
                        row.append("N/A")

            table_rows.append(row)

    comparison_table = tabulate(
        table_rows,
        headers=table_headers,
        tablefmt="github"
    )

    # Create visualization
    sns.set_style("whitegrid")

    # Limit to first 12 tests for readability
    plot_tests = dict(list(test_stats.items())[:12])

    graph_path = None
    if len(plot_tests) > 0:
        test_names = list(plot_tests.keys())
        fig, ax = plt.subplots(figsize=(max(14, len(test_names) * 0.7), 8))

        x = range(len(test_names))

        if len(all_backends) == 1:
            # Simple: group by branch
            width = 0.8 / len(all_branches)
            colors = sns.color_palette("husl", len(all_branches))
            backend_name = all_backends[0]

            for i, branch_name in enumerate(all_branches):
                means = []
                stds = []
                for tn in test_names:
                    if branch_name in plot_tests[tn] and backend_name in plot_tests[tn][branch_name]:
                        means.append(plot_tests[tn][branch_name][backend_name]["mean"] * 1000)
                        stds.append(plot_tests[tn][branch_name][backend_name]["std"] * 1000)
                    else:
                        means.append(0)
                        stds.append(0)

                offset = (i - len(all_branches) / 2) * width + width / 2
                ax.bar(
                    [pos + offset for pos in x],
                    means,
                    width,
                    yerr=stds,
                    label=branch_name,
                    color=colors[i],
                    alpha=0.8,
                    capsize=3
                )

            ax.set_title(f"Branch Performance Comparison ({backend_name})", fontsize=14)

        else:
            # Complex: group by backend:branch
            combinations = [(be, br) for be in all_backends for br in all_branches]
            width = 0.8 / len(combinations)
            colors = sns.color_palette("husl", len(combinations))

            for i, (backend_name, branch_name) in enumerate(combinations):
                means = []
                stds = []
                for tn in test_names:
                    if branch_name in plot_tests[tn] and backend_name in plot_tests[tn][branch_name]:
                        means.append(plot_tests[tn][branch_name][backend_name]["mean"] * 1000)
                        stds.append(plot_tests[tn][branch_name][backend_name]["std"] * 1000)
                    else:
                        means.append(0)
                        stds.append(0)

                offset = (i - len(combinations) / 2) * width + width / 2
                ax.bar(
                    [pos + offset for pos in x],
                    means,
                    width,
                    yerr=stds,
                    label=f"{backend_name}:{branch_name}",
                    color=colors[i],
                    alpha=0.8,
                    capsize=3
                )

            ax.set_title("Branch Performance Comparison (All Backends)", fontsize=14)

        ax.set_xlabel("Test", fontsize=12)
        ax.set_ylabel("Time (milliseconds)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [tn.split('.')[-1] if '.' in tn else tn for tn in test_names],
            rotation=45,
            ha='right'
        )
        ax.legend(title="Branch" if len(all_backends) == 1 else "Backend:Branch", loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = REPORTS_DIR / f"branch_comparison_{timestamp_str}.png"
        plt.savefig(graph_path, dpi=150)
        plt.close()

    # Build markdown report
    lines = [
        "# Branch Comparison Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Tests analyzed: {len(test_stats)}",
        f"- Branches: {', '.join(all_branches)}",
        f"- Backends: {', '.join(all_backends)}",
        f"- Total runs: {sum(sum(sum(len(times) for times in backends.values()) for backends in branches.values()) for branches in test_branch_backend_data.values())}",
    ]

    if test_pattern:
        lines.append(f"- Filtered by pattern: `{test_pattern}`")
    if component:
        lines.append(f"- Filtered by component: `{component}`")
    if backend:
        lines.append(f"- Filtered by backend: `{backend}`")

    lines.extend([
        "",
        "## Performance Comparison",
        "",
        "Times shown as: mean±std (speedup vs baseline) [number of runs]",
        "",
    ])

    if graph_path:
        lines.append(f"![Branch Comparison]({graph_path.name})")
        lines.append("")

    lines.extend([
        "## Results Table",
        "",
        comparison_table,
        "",
        "## Branch Summary",
        "",
    ])

    # Overall branch statistics
    branch_totals = {}
    for test_name, branches_data in test_stats.items():
        for branch_name, backends_data in branches_data.items():
            if branch_name not in branch_totals:
                branch_totals[branch_name] = []
            for backend_name, stats in backends_data.items():
                branch_totals[branch_name].append(stats["mean"])

    for branch_name in sorted(branch_totals.keys()):
        times = branch_totals[branch_name]
        mean = sum(times) / len(times)
        lines.append(f"- **{branch_name}**: Average {mean * 1000:.2f}ms across {len(times)} test-backend combinations")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Speedup**: Ratio of baseline (fastest) time to current time",
        "- **Baseline**: The fastest branch (or backend:branch combination) for each test",
        "- **N/A**: Branch has no performance data for this test",
        "- Lower times and higher speedups indicate better performance",
        "",
        "## Use Cases",
        "",
        "- **Feature branch comparison**: See if your feature branch introduced performance regressions",
        "- **Before/after comparison**: Compare performance before and after a specific change",
        "- **Release validation**: Ensure release branches maintain or improve performance",
        "",
        "## Notes",
        "",
        "- Results are averaged across multiple runs when available",
        "- Standard deviation shows consistency (lower is more consistent)",
        "- Compare same backend across branches for fairest comparison",
        "- File status (clean/modified) may affect results - check individual runs if needed",
    ])

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path
