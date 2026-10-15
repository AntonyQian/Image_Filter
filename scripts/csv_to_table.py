"""Turn the bench CSV into Markdown tables for the README, one per sigma.

Usage:
    python scripts/csv_to_table.py results/bench.csv
    python scripts/csv_to_table.py results/bench.csv --threads 1,2,4,8

Every number comes straight from the CSV, so the README cannot disagree with it.
The serial row is always kept. --threads picks which OpenMP thread counts to show
(default: all of them).
"""
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--threads", default="",
                        help="comma-separated OpenMP thread counts to show (default: all)")
    args = parser.parse_args()

    wanted = {int(t) for t in args.threads.split(",") if t}

    with open(args.csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    sigmas = sorted({float(r["sigma"]) for r in rows})
    for sigma in sigmas:
        group = [r for r in rows if float(r["sigma"]) == sigma]
        radius = group[0]["radius"]
        print(f"### Sigma {sigma:g} (radius {radius})\n")
        print("| Size | Backend | Threads | Median (ms) | Speedup | Efficiency |")
        print("|---|---|---|---|---|---|")
        for r in group:
            if r["backend"] != "serial" and wanted and int(r["threads"]) not in wanted:
                continue
            print(f"| {r['width']}x{r['height']} | {r['backend']} | {r['threads']} "
                  f"| {float(r['median_ms']):.0f} | {float(r['speedup']):.2f} "
                  f"| {float(r['efficiency']):.2f} |")
        print()
    print("Median of the timed runs after one warm-up; speedup is relative to the "
          "serial row of the same size and sigma. Full data: `results/bench.csv`.")


if __name__ == "__main__":
    main()
