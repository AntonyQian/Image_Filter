"""Turn the bench CSV into Markdown tables for the README, one per sigma.

Usage:
    python scripts/csv_to_table.py results/bench.csv
    python scripts/csv_to_table.py results/bench.csv --threads 1,2,4,8

Every number comes straight from the CSV, so the README cannot disagree with it.
The serial row and the CUDA rows are always kept. --threads picks which OpenMP
thread counts to show (default: all of them).

Empty fields mean "does not apply" and are shown as "-": threads and efficiency
in a CUDA row, and the GPU times in a CPU row. The GPU time columns are only
shown if the CSV has CUDA rows; an older CSV without those columns works too.
"""
import argparse
import csv


def number(text, digits):
    """A CSV field as a rounded number, or "-" if it is empty or missing."""
    if not text:
        return "-"
    return f"{float(text):.{digits}f}"


def milliseconds(text):
    """A median time: whole ms, or one decimal below 10 ms (GPU Runs are short)."""
    return number(text, 1 if text and float(text) < 10 else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--threads", default="",
                        help="comma-separated OpenMP thread counts to show (default: all)")
    args = parser.parse_args()

    wanted = {int(t) for t in args.threads.split(",") if t}

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    show_gpu = any(r.get("kernel_ms") for r in rows)

    header = "| Size | Backend | Threads | Median (ms) | Speedup | Efficiency |"
    rule = "|---|---|---|---|---|---|"
    if show_gpu:
        header += " H2D (ms) | Kernel (ms) | D2H (ms) |"
        rule += "---|---|---|"

    sigmas = sorted({float(r["sigma"]) for r in rows})
    for sigma in sigmas:
        group = [r for r in rows if float(r["sigma"]) == sigma]
        radius = group[0]["radius"]
        print(f"### Sigma {sigma:g} (radius {radius})\n")
        print(header)
        print(rule)
        for r in group:
            if r["backend"] == "openmp" and wanted and int(r["threads"]) not in wanted:
                continue
            line = (f"| {r['width']}x{r['height']} | {r['backend']} | {r['threads'] or '-'} "
                    f"| {milliseconds(r['median_ms'])} | {number(r['speedup'], 2)} "
                    f"| {number(r['efficiency'], 2)} |")
            if show_gpu:
                line += (f" {number(r.get('h2d_ms'), 2)} | {number(r.get('kernel_ms'), 2)} "
                         f"| {number(r.get('d2h_ms'), 2)} |")
            print(line)
        print()
    print("Median of the timed runs after one warm-up; speedup is relative to the "
          f"serial row of the same size and sigma. Full data: `{args.csv_path}`.")


if __name__ == "__main__":
    main()
