"""Compare accuracy conditions as a paired statistical test, not by eyeballing totals.

Every condition answers the SAME probes, so the right test is McNemar's exact test on the
discordant pairs: of the probes where two conditions disagree, how lopsided is the split?
Also prints a Wilson 95% interval per condition, which is honest about small samples.

    python bench/analyze.py bench/results/accuracy-paired.json [more.json ...]
"""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (0.0, 0.0)
    p = correct / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def binom_two_sided(b: int, c: int) -> float:
    """Exact McNemar p-value: a fair coin over b+c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main(paths: list[str]) -> int:
    probes: list[dict] = []
    conditions: list[str] = []
    for path in paths:
        d = json.loads(open(path).read())
        for c in d["conditions"]:
            if c not in conditions:
                conditions.append(c)
        for region in d["per_region"]:
            probes.extend(region.get("paired", []))
    if not probes:
        print("no paired per-probe records found; re-run accuracy.py to produce them")
        return 1

    print(f"{len(probes)} paired probes from {len(paths)} run(s)\n")
    print(f"{'condition':12} {'correct':>9} {'accuracy':>9}  95% interval")
    for c in conditions:
        vals = [p[c] for p in probes if c in p]
        correct, total = sum(vals), len(vals)
        lo, hi = wilson(correct, total)
        print(f"{c:12} {correct:4d}/{total:<4d} {correct / total:9.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\npaired comparisons (McNemar exact):")
    for a, b in combinations(conditions, 2):
        both = [p for p in probes if a in p and b in p]
        a_only = sum(1 for p in both if p[a] and not p[b])
        b_only = sum(1 for p in both if p[b] and not p[a])
        pv = binom_two_sided(a_only, b_only)
        verdict = "no detectable difference" if pv > 0.05 else f"{a if a_only > b_only else b} better"
        print(f"  {a:11} vs {b:11}  {a}-only={a_only:3d}  {b}-only={b_only:3d}  p={pv:.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["bench/results/accuracy-paired.json"]))
