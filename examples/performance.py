#!/usr/bin/env python3
"""
Streamlit dashboard to visualize performance metrics from perf_results.

Features
- Load JSONL results from `perf_results/perf_results.jsonl`.
- Filter by metric, interface, backend, kernel, commit.
- Views:
  - Latest commit: cross-backend bar charts per interface.
  - Trends: line charts over time per (interface, backend).
  - Heatmap: latest commit interface × backend grid.
  - Raw table preview.
- Rendering: choose Matplotlib Agg or yapocis backend for figure rendering.

Run
- streamlit run examples/performance.py
- Optional args: `--results perf_results/perf_results.jsonl`
"""

from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Matplotlib/Seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------
# Data loading and preparation
# ------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as inf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
    return out


def normalize_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "commit",
                "backend",
                "interface",
                "kernel",
                "metric",
                "value",
                "iters",
                "source_md5",
                "num_kernels",
            ]
        )
    df = pd.json_normalize(records)
    # Ensure expected columns exist
    for col in (
        "timestamp",
        "commit",
        "backend",
        "interface",
        "kernel",
        "metric",
        "value",
        "iters",
        "source_md5",
        "num_kernels",
    ):
        if col not in df.columns:
            df[col] = np.nan
    # Parse timestamp
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception:
            pass
    return df


def latest_commit(df: pd.DataFrame) -> Optional[str]:
    if df.empty or "commit" not in df:
        return None
    # Use timestamp order if available; otherwise, keep appearance order
    if "timestamp" in df and df["timestamp"].notna().any():
        s = df.loc[df["timestamp"].notna()].drop_duplicates(["commit"]).sort_values("timestamp")["commit"]
        return s.iloc[-1] if len(s) else None
    commits = df["commit"].dropna().astype(str).tolist()
    return commits[-1] if commits else None


def summarize_latest_median(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return per (interface, backend) median for the latest commit."""
    if df.empty:
        return df
    lat = latest_commit(df)
    if lat is None:
        return pd.DataFrame(columns=["interface", "backend", "value"])
    sub = df[(df["commit"] == lat) & (df["metric"] == metric)].copy()
    # Ensure numeric values for aggregation/plotting
    if "value" in sub.columns:
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        sub = sub.dropna(subset=["value"])  # drop non-numeric
    if sub.empty:
        return pd.DataFrame(columns=["interface", "backend", "value"])
    g = sub.groupby(["interface", "backend"])  # median across repeats
    res = g["value"].median().reset_index()
    return res


def trend_series(df: pd.DataFrame, metric: str, interface: str, backend: str) -> pd.DataFrame:
    sub = df[(df["metric"] == metric) & (df["interface"] == interface) & (df["backend"] == backend)]
    cols = [c for c in ("timestamp", "commit", "value") if c in sub.columns]
    out = sub[cols].dropna(subset=["value"]).copy()
    if "timestamp" in out:
        out = out.sort_values("timestamp")
    return out


# ------------------------------
# Rendering helpers
# ------------------------------


@dataclass
class RenderBackend:
    name: str  # "Agg" or "yapocis"


def render_matplotlib(fig: plt.Figure, backend: RenderBackend):
    """Render a Matplotlib figure via Agg or the yapocis backend and display in Streamlit."""
    if backend.name == "Agg":
        st.pyplot(fig)
        return
    # Render using yapocis FigureCanvas -> PNG path -> st.image
    try:
        from yapocis.lib.mplbackend import FigureCanvas as FigureCanvasYapo
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            canvas = FigureCanvasYapo(fig)
            canvas.print_png(tmp_path)
            with open(tmp_path, "rb") as inf:
                st.image(inf.read())
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception as e:
        st.warning(f"Falling back to Agg renderer (yapocis backend unavailable): {e}")
        st.pyplot(fig)


# ------------------------------
# Streamlit app
# ------------------------------


def main(args: Optional[argparse.Namespace] = None):
    st.set_page_config(page_title="Yapocis Performance Dashboard", layout="wide")
    st.title("Yapocis Performance Dashboard")

    # Input path
    default_path = Path("perf_results/perf_results.jsonl")
    cli_path = Path(args.results) if args and getattr(args, "results", None) else default_path
    results_path = st.sidebar.text_input("Results JSONL path", str(cli_path))
    path = Path(results_path)

    records = load_jsonl(path)
    df = normalize_records(records)

    if df.empty:
        st.info("No results found. Run performance tests to populate perf_results.")
        st.stop()

    # Sidebar filters
    metrics = sorted(df["metric"].dropna().astype(str).unique().tolist())
    metric = st.sidebar.selectbox("Metric", options=metrics, index=metrics.index("exec_ms") if "exec_ms" in metrics else 0)

    # Optional categorical filters
    all_interfaces = sorted(df["interface"].dropna().astype(str).unique().tolist())
    sel_interfaces = st.sidebar.multiselect("Interfaces", options=["<all>"] + all_interfaces, default=["<all>"])
    all_backends = sorted(df["backend"].dropna().astype(str).unique().tolist())
    sel_backends = st.sidebar.multiselect("Backends", options=["<all>"] + all_backends, default=["<all>"])
    all_kernels = sorted(df["kernel"].dropna().astype(str).unique().tolist())
    sel_kernels = st.sidebar.multiselect("Kernels", options=["<all>"] + all_kernels, default=["<all>"])

    # Rendering backend selector
    rb = st.sidebar.radio("Matplotlib renderer", options=["Agg", "yapocis"], index=0, help="Use yapocis GPU renderer when available.")
    rbackend = RenderBackend(name=rb)

    # Optional speedup vs baseline backend
    st.sidebar.markdown("---")
    show_speedup = st.sidebar.checkbox("Show speedup vs baseline backend", value=False)
    baseline_backend = st.sidebar.selectbox(
        "Baseline backend",
        options=all_backends or [""],
        index=0 if all_backends else 0,
        help="Used when speedup is enabled. Speedup = baseline/value",
    )

    # Apply filters
    fdf = df[df["metric"] == metric].copy()
    if sel_interfaces and "<all>" not in sel_interfaces:
        fdf = fdf[fdf["interface"].isin(sel_interfaces)]
    if sel_backends and "<all>" not in sel_backends:
        fdf = fdf[fdf["backend"].isin(sel_backends)]
    if sel_kernels and "<all>" not in sel_kernels:
        fdf = fdf[fdf["kernel"].isin(sel_kernels)]

    # Tabs for different views
    tabs = st.tabs(["Latest Bars", "Trend", "Delta", "Kernel Drilldown", "Facets", "Heatmap", "Table"])

    # Latest commit bar charts per interface
    with tabs[0]:
        st.subheader("Latest commit: backend comparison per interface")
        lat = latest_commit(fdf)
        if not lat:
            st.info("No commit information available.")
        latest = summarize_latest_median(fdf, metric)
        if latest.empty:
            st.info("No data for selected metric/filters.")
        else:
            # CSV export for latest summary
            csv_latest = latest.assign(metric=metric, commit=lat).to_csv(index=False)
            st.download_button(
                label="Download latest summary CSV",
                data=csv_latest,
                file_name=f"perf_latest_{metric}.csv",
                mime="text/csv",
            )
            # Optionally compute speedup vs baseline per interface
            latest_plot = latest.copy()
            if show_speedup and baseline_backend:
                base = latest_plot[latest_plot["backend"] == baseline_backend][["interface", "value"]].rename(columns={"value": "baseline"})
                latest_plot = latest_plot.merge(base, on="interface", how="left")
                latest_plot["speedup"] = latest_plot.apply(lambda r: (r["baseline"] / r["value"]) if pd.notna(r.get("baseline")) and pd.notna(r.get("value")) and r["value"] != 0 else np.nan, axis=1)
            # For each interface having multiple backends, plot bars
            for iname, sub in latest_plot.groupby("interface"):
                if len(sub["backend"].unique()) < 1:
                    continue
                fig, ax = plt.subplots(figsize=(6, 3))
                if show_speedup and "speedup" in sub.columns:
                    sns.barplot(data=sub, x="backend", y="speedup", ax=ax, color="#4477aa")
                    ax.set_title(f"{iname} – speedup vs {baseline_backend} (median) – commit {lat}")
                    ax.set_xlabel("backend"); ax.set_ylabel("speedup")
                else:
                    sns.barplot(data=sub, x="backend", y="value", ax=ax, color="#4477aa")
                    ax.set_title(f"{iname} – {metric} (median) – commit {lat}")
                    ax.set_xlabel("backend"); ax.set_ylabel(metric)
                for p in ax.patches:
                    h = p.get_height();
                    ax.annotate(f"{h:.2f}", (p.get_x()+p.get_width()/2, h), ha="center", va="bottom", fontsize=8, rotation=0)
                fig.tight_layout()
                render_matplotlib(fig, rbackend)
                plt.close(fig)

    # Trend view: choose one interface/backend
    with tabs[1]:
        st.subheader("Trend over time")
        col1, col2 = st.columns(2)
        pick_iface = col1.selectbox("Interface", options=all_interfaces, index=0 if all_interfaces else None)
        pick_be = col2.selectbox("Backend", options=all_backends, index=0 if all_backends else None)
        if pick_iface and pick_be:
            ts = trend_series(df, metric, pick_iface, pick_be)
            if ts.empty:
                st.info("No trend data for this selection.")
            else:
                fig, ax = plt.subplots(figsize=(7, 3))
                # Prefer timestamp when available; otherwise order by commit appearance
                if "timestamp" in ts.columns and ts["timestamp"].notna().any():
                    sns.lineplot(data=ts, x="timestamp", y="value", marker="o", ax=ax)
                    ax.set_xlabel("timestamp")
                else:
                    # fallback: order by commit string
                    ts2 = ts.copy(); ts2["idx"] = np.arange(len(ts2))
                    sns.lineplot(data=ts2, x="idx", y="value", marker="o", ax=ax)
                    ax.set_xlabel("commit index")
                ax.set_ylabel(metric)
                ax.set_title(f"{pick_iface}:{pick_be} – {metric} over time")
                fig.tight_layout()
                render_matplotlib(fig, rbackend)
                plt.close(fig)

    # Delta vs pinned commit pair
    with tabs[2]:
        st.subheader("Delta between two commits (median)")
        # Determine ordered commits by timestamp if present
        commits_series = None
        if "timestamp" in fdf.columns and fdf["timestamp"].notna().any():
            commits_series = (
                fdf.loc[fdf["timestamp"].notna(), ["commit", "timestamp"]]
                .drop_duplicates(["commit"]).sort_values("timestamp")["commit"]
            )
        else:
            commits_series = fdf["commit"].dropna().astype(str)
        commits_list = commits_series.tolist() if commits_series is not None else []
        commits_unique = list(dict.fromkeys(commits_list))  # preserve order
        if len(commits_unique) < 2:
            st.info("Need at least two commits to compute deltas.")
        else:
            colc1, colc2 = st.columns(2)
            prev_c = colc1.selectbox("Older commit", options=commits_unique, index=max(0, len(commits_unique)-2))
            latest_c = colc2.selectbox("Newer commit", options=commits_unique, index=len(commits_unique)-1)
            # Compute median per (interface, backend) for both commits
            lat_df = fdf[(fdf["commit"] == latest_c)]
            prev_df = fdf[(fdf["commit"] == prev_c)]
            lat_sum = lat_df.groupby(["interface", "backend"])['value'].median().rename('latest').reset_index()
            prev_sum = prev_df.groupby(["interface", "backend"])['value'].median().rename('previous').reset_index()
            merged = pd.merge(lat_sum, prev_sum, on=["interface", "backend"], how="inner")
            if merged.empty:
                st.info("No overlapping (interface, backend) entries between last two commits.")
            else:
                merged["delta_ms"] = merged["latest"] - merged["previous"]
                merged["pct"] = np.where(merged["previous"] != 0, merged["delta_ms"] / merged["previous"] * 100.0, np.nan)
                st.caption(f"Comparing {prev_c} → {latest_c}")
                # CSV export for deltas
                st.download_button(
                    label="Download delta CSV",
                    data=merged.assign(metric=metric, prev_commit=prev_c, latest_commit=latest_c).to_csv(index=False),
                    file_name=f"perf_delta_{metric}.csv",
                    mime="text/csv",
                )
                # Bar chart: delta per interface with backend hue
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=merged, x="interface", y="delta_ms", hue="backend", ax=ax)
                ax.axhline(0, color="#888", linewidth=1)
                ax.set_ylabel(f"Δ {metric} (ms)")
                ax.set_xlabel("interface")
                ax.set_title("Delta between two commits (median)")
                fig.tight_layout()
                render_matplotlib(fig, rbackend)
                plt.close(fig)
                # Show table with percentage deltas
                st.dataframe(merged.sort_values(["interface", "backend"]))

    # Kernel drilldown for latest commit
    with tabs[3]:
        st.subheader("Per-kernel drilldown (latest commit)")
        lat = latest_commit(fdf)
        if not lat:
            st.info("No commit information available.")
        else:
            k_ifaces = sorted(fdf["interface"].dropna().astype(str).unique().tolist())
            pick_iface = st.selectbox("Interface", options=k_ifaces, index=0 if k_ifaces else None, key="kernel_iface")
            if pick_iface:
                sub = fdf[(fdf["commit"] == lat) & (fdf["interface"] == pick_iface)]
                if sub.empty or sub["kernel"].isna().all():
                    st.info("No kernel-level data for this selection.")
                else:
                    agg = (
                        sub.dropna(subset=["kernel"]).groupby(["kernel", "backend"])['value']
                        .median().reset_index()
                    )
                    # Optionally compute speedup vs baseline by kernel
                    if show_speedup and baseline_backend:
                        base_k = agg[agg["backend"] == baseline_backend][["kernel", "value"]].rename(columns={"value": "baseline"})
                        agg = agg.merge(base_k, on="kernel", how="left")
                        agg["speedup"] = agg.apply(lambda r: (r["baseline"] / r["value"]) if pd.notna(r.get("baseline")) and pd.notna(r.get("value")) and r["value"] != 0 else np.nan, axis=1)
                    # CSV export
                    st.download_button(
                        label="Download kernel summary CSV",
                        data=agg.assign(metric=metric, commit=lat, interface=pick_iface).to_csv(index=False),
                        file_name=f"perf_kernels_{pick_iface}_{metric}.csv",
                        mime="text/csv",
                    )
                    # Bar chart: x=kernel, hue=backend
                    fig, ax = plt.subplots(figsize=(max(6, 0.3 * max(1, agg['kernel'].nunique()) + 2), 4))
                    ycol = "speedup" if (show_speedup and "speedup" in agg.columns) else "value"
                    sns.barplot(data=agg, x="kernel", y=ycol, hue="backend", ax=ax)
                    ttl = f"speedup vs {baseline_backend}" if ycol == "speedup" else metric
                    ax.set_title(f"{pick_iface} – {ttl} (median) by kernel – commit {lat}")
                    ax.set_xlabel("kernel"); ax.set_ylabel(ttl)
                    ax.tick_params(axis='x', rotation=45, labelsize=8)
                    fig.tight_layout()
                    render_matplotlib(fig, rbackend)
                    plt.close(fig)

    # Faceted comparison across interfaces for latest commit
    with tabs[4]:
        st.subheader("Faceted backend comparison (latest commit)")
        lat = latest_commit(fdf)
        latest = summarize_latest_median(fdf, metric)
        if latest.empty:
            st.info("No data for selected metric/filters.")
        else:
            # Select multiple interfaces
            iface_opts = sorted(latest["interface"].unique().tolist())
            sel = st.multiselect("Interfaces to compare", options=iface_opts, default=iface_opts[:4], key="facet_ifaces")
            data = latest[latest["interface"].isin(sel)] if sel else latest
            # Optional speedup vs baseline across interfaces
            if show_speedup and baseline_backend:
                base = data[data["backend"] == baseline_backend][["interface", "value"]].rename(columns={"value": "baseline"})
                data = data.merge(base, on="interface", how="left")
                data["speedup"] = data.apply(lambda r: (r["baseline"] / r["value"]) if pd.notna(r.get("baseline")) and pd.notna(r.get("value")) and r["value"] != 0 else np.nan, axis=1)
            normalize = st.checkbox("Normalize per-interface (value / min)", value=False)
            if normalize and show_speedup:
                st.caption("Note: normalization disabled when speedup is shown.")
                normalize = False
            if normalize:
                data = data.copy()
                data["norm"] = data.groupby("interface")["value"].transform(lambda s: s / s.min())
                ycol = "norm"; ylab = f"{metric} / min"
            else:
                if show_speedup and "speedup" in data.columns:
                    ycol = "speedup"; ylab = f"speedup vs {baseline_backend}"
                else:
                    ycol = "value"; ylab = metric
            if data.empty:
                st.info("No data after selection.")
            else:
                # Facet by interface
                g = sns.catplot(data=data, x="backend", y=ycol, col="interface", kind="bar", col_wrap=3, height=3, sharey=normalize)
                g.set_titles("{col_name}")
                g.set_xlabels("backend"); g.set_ylabels(ylab)
                fig = g.fig
                fig.suptitle(f"Latest commit {lat} – {ylab}", y=1.02)
                fig.tight_layout()
                render_matplotlib(fig, rbackend)
                plt.close(fig)

    # Heatmap for latest commit
    with tabs[5]:
        st.subheader("Latest commit heatmap")
        latest = summarize_latest_median(fdf, metric)
        if latest.empty:
            st.info("No data for heatmap.")
        else:
            # Pivot to interface × backend
            pivot = latest.pivot(index="interface", columns="backend", values="value")
            # Coerce to numeric to satisfy matplotlib pcolormesh
            pivot = pivot.apply(pd.to_numeric, errors="coerce")
            fig, ax = plt.subplots(figsize=(min(10, 2 + 0.6 * max(1, len(pivot.columns))),
                                           min(10, 2 + 0.4 * max(1, len(pivot.index)))))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
            ax.set_title(f"{metric} (median) – latest commit")
            fig.tight_layout()
            render_matplotlib(fig, rbackend)
            plt.close(fig)

    # Raw table
    with tabs[6]:
        st.subheader("Raw records (filtered)")
        st.dataframe(fdf.sort_values([c for c in ("timestamp", "interface", "backend", "kernel") if c in fdf.columns]))


if __name__ == "__main__":
    # Allow optional CLI override of results path to help local runs
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="perf_results/perf_results.jsonl", help="Path to results JSONL")
    ns = ap.parse_args()
    # Streamlit ignores __main__; this path helps when running as script
    main(ns)
