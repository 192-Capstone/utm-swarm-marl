"""Small statistics helpers for evaluation results (no simulator imports).

wilson         : 95% confidence interval for a success rate
mcnemar_exact  : paired test for two policies flown on the same episodes
holm           : Holm-Bonferroni correction for a family of p-values
spread         : mean, standard deviation, min and max over training seeds
"""
import math


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n. Behaves sensibly
    at k=0 and k=n, unlike the normal approximation."""
    if n == 0:
        return (float('nan'), float('nan'))
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test. b = episodes only policy A solved,
    c = episodes only policy B solved (episodes both or neither solved carry
    no information about which policy is better). Exact binomial, so it is
    valid for small b + c, where the chi-square version is not."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def spread(values):
    """(mean, sample std, min, max) of a list of numbers (std 0 for one value)."""
    v = [float(x) for x in values]
    if not v:
        return (float('nan'),) * 4
    mean = sum(v) / len(v)
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return (mean, sd, min(v), max(v))
