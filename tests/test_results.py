"""The headline numbers in the README are the numbers in results/*.json."""

import subprocess
import sys

from a1slip.paths import ROOT

README = (ROOT / "README.md").read_text()


def tour():
    return subprocess.run([sys.executable, "-m", "a1slip.tour"], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def test_tour_runs_and_matches_readme():
    out = tour()
    lines = {line.strip().split("  ")[0]: line for line in out.splitlines() if line.startswith("  ")}
    expected = {  # tour row -> numbers that must appear in that row and in the README
        "fixed 1.0": ("31.2 s", "20.5 s"),
        "fixed 0.3": ("49.0 s", "8.3 s"),
        "XGBoost governor (risk < 0.65)": ("39.1 s", "8.8 s"),
        "TabPFN governor (risk < 0.65)": ("32.0 s", "8.9 s"),
        "no map": ("14.4 s",),
        "TabPFN map": ("4.6 s",),
        "obstacles only": ("23.0 s",),
        "felt + seen (TabPFN)": ("4.5 s", "28.7 s"),
    }
    for row, numbers in expected.items():
        assert row in lines, f"tour has no row '{row}'"
        for n in numbers:
            assert n in lines[row], f"{row}: tour shows {lines[row]!r}, README expects {n}"
            assert f"**{n}**" in README or n in README, f"README is missing {n}"
