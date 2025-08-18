import re
import os
import statistics
from rich.console import Console
from rich.table import Table

# === CONFIG ===
LOG_DIR = "logs"
BATCH_SIZE = 128  # adjust if different
time_pattern = re.compile(r"([\d.]+)s/batch")

# Group log files by variant
variant_files = {
    "CLaRE (CoCoOp) with Cross Attention": [
        "cocoop_ca_41k.log",
        "cocoop_ca_82k.log",
        "cocoop_ca_164k.log",
    ],
    "CLaRE (CoCoOp) with ESA": [
        "cocoop_esa_41k.log",
        "cocoop_esa_82k.log",
        "cocoop_esa_164k.log",
    ],
    "CLaRE (CoOp) with Cross Attention": [
        "coop_ca_41k.log",
        "coop_ca_82k.log",
        "coop_ca_164k.log",
        "coop_ca_410k.log",
    ],
    "CLaRE (CoOp) with ESA": [
        "coop_esa_41k.log",
        "coop_esa_82k.log",
        "coop_esa_164k.log",
    ],
    "CoCoOp": [
        "cocoop_41k.log",
        "cocoop_82k.log",
        "cocoop_164k.log",
        "cocoop_410k.log",
    ],
    "CoOp": [
        "coop_41k.log",
        "coop_82k.log",
        "coop_164k.log",
        "coop_410k.log",
    ],
}

console = Console()

def extract_times(filepath):
    """Extract all s/batch values from a log file."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return [float(m.group(1)) for m in time_pattern.finditer(text)]

def summarize_variant(variant, files):
    """Summarize per-log and aggregate statistics for a variant."""
    variant_times = []

    # Per-log table
    table = Table(title=f"{variant} – Per Log Stats", show_lines=True)
    table.add_column("Log File")
    table.add_column("Count", justify="right")
    table.add_column("Avg (s/batch)", justify="right")
    table.add_column("StdDev", justify="right")

    for fname in files:
        fpath = os.path.join(LOG_DIR, fname)
        if not os.path.exists(fpath):
            table.add_row(fname, "-", "-", "-")
            continue

        times = extract_times(fpath)
        if times:
            avg = statistics.mean(times)
            std = statistics.pstdev(times) if len(times) > 1 else 0.0
            table.add_row(fname, str(len(times)), f"{avg:.3f}", f"{std:.3f}")
            variant_times.extend(times)
        else:
            table.add_row(fname, "0", "-", "-")

    console.print(table)

    # Aggregate across logs
    if variant_times:
        avg_all = statistics.mean(variant_times)
        std_all = statistics.pstdev(variant_times)
        time_per_img = avg_all / BATCH_SIZE
        console.print(
            f"[bold]{variant} – Aggregate:[/bold] "
            f"{avg_all:.3f} s/batch (±{std_all:.3f}, n={len(variant_times)})"
        )
        console.print(
            f"    ≈ {time_per_img:.4f} s/image (batch size {BATCH_SIZE})\n"
        )
    else:
        console.print(f"[bold]{variant} – Aggregate:[/bold] No data\n")

def main():
    for variant, files in variant_files.items():
        summarize_variant(variant, files)

if __name__ == "__main__":
    main()
