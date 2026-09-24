#!/usr/bin/env python3
"""Plot the 1–5 core mean speedup from a completed Problem 1 summary CSV."""

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='mpl_problem1_'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.summary.open(encoding='utf-8-sig')))
    if len(rows) != 5 or any(int(row['cases_ok']) != int(row['cases_expected']) for row in rows):
        parser.error('Summary must contain complete results for cores 1–5')
    x = [int(row['cores']) for row in rows]
    y = [float(row['average_speedup']) for row in rows]
    if x != [1, 2, 3, 4, 5]:
        parser.error('Summary must list cores 1–5 in order')
    output = args.output or args.summary.with_name('average_speedup.png')
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    ax.plot(x, y, marker='o', linewidth=2.4, markersize=6, color='#176AA3')
    ax.plot([1, 5], [1, 5], linestyle='--', linewidth=1, color='#999999',
            label='Ideal linear speedup')
    for xi, yi in zip(x, y):
        ax.annotate(f'{yi:.2f}', (xi, yi), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.set(xlabel='Number of cores', ylabel='Mean speedup',
           title=f'Problem 1: {rows[0]["cases_expected"]} selected cases',
           xticks=x, xlim=(0.7, 5.3), ylim=(0, max(5.5, max(y) + 0.5)))
    ax.grid(axis='y', alpha=.25)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
