"""gta6-trophy-rarity-forecast-01 advanced forecaster in Python 3.13 (standard library only).

Models trophy earn rates on Exophase at the target time using:
1. Logit-trend fit against log-hours since origin (2026-11-19T00:00:00Z).
2. Earned-vs-tracked dilution model (E = a + b * N with N extrapolated via log-linear fit).
3. Flat-history refresh prior (dilutes unrefreshed flat histories due to player growth).
4. Dynamic spread sigma incorporating horizon, trend residual variance, and jump/refresh variance.
5. Exact half-up rounding to 4 decimals, quantile clamping [0.01, 99.99], and non-decreasing enforcement.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp(x: float, low: float = 0.01, high: float = 99.99) -> float:
    return min(high, max(low, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    p = 100.0 / (1.0 + math.exp(-lg))
    return clamp(p)


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_iso(dt_str: str) -> datetime:
    if dt_str.endswith("Z"):
        dt_str = dt_str[:-1] + "+00:00"
    return datetime.fromisoformat(dt_str)


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Fits y = a + b * x. Returns (a, b, residual_variance)."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if abs(den) < 1e-12:
        return mean_y, 0.0, 0.0
    b = num / den
    a = mean_y - b * mean_x
    res_sq = sum((ys[i] - (a + b * xs[i])) ** 2 for i in range(n))
    res_var = res_sq / (n - 2) if n > 2 else 0.0
    return a, b, max(0.0, res_var)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    history = case["history"]
    target_dt = parse_iso(case["target"])
    horizon_minutes = case["horizon_minutes"]
    horizon_days = horizon_minutes / 1440.0

    # Parse history reads
    parsed_reads = []
    for read in history:
        dt = parse_iso(read["at"])
        t_hours = (dt - ORIGIN).total_seconds() / 3600.0
        x = math.log(1.0 + max(0.0, t_hours))
        N = float(read["players_tracked"])
        rates = read["rates"]
        parsed_reads.append({"dt": dt, "t_hours": t_hours, "x": x, "N": N, "rates": rates})

    last_read = parsed_reads[-1]
    last_dt = last_read["dt"]
    last_N = last_read["N"]
    last_rates = last_read["rates"]

    # Filter window: last 24 hours of reads (or all if < 3 distinct reads)
    cutoff_t = last_read["t_hours"] - 24.0
    window_reads = [r for r in parsed_reads if r["t_hours"] >= cutoff_t]
    if len(set(r["t_hours"] for r in window_reads)) < 3:
        window_reads = parsed_reads

    # Forecast N_target via least-squares fit of ln(N) against t (in hours) over window
    N_times = [r["t_hours"] for r in window_reads]
    N_vals = [math.log(max(1.0, r["N"])) for r in window_reads]
    na, nb, _ = linear_fit(N_times, N_vals)
    target_t_hours = (target_dt - ORIGIN).total_seconds() / 3600.0
    forecast_ln_N = na + nb * target_t_hours
    forecast_N = math.exp(forecast_ln_N)
    forecast_N = max(last_N, min(4.0 * last_N, forecast_N))

    # Check if history is perfectly flat across all reads
    all_rates_first = parsed_reads[0]["rates"]
    all_N_first = parsed_reads[0]["N"]
    is_perfectly_flat = True
    for r in parsed_reads:
        if r["N"] != all_N_first or r["rates"] != all_rates_first:
            is_perfectly_flat = False
            break

    # Flat-history refresh prior condition
    apply_flat_prior = is_perfectly_flat and (horizon_minutes > 60)

    rows = []
    num_trophies = len(case["trophies"])

    for i in range(num_trophies):
        trophy = case["trophies"][i]
        last_rate = last_rates[i]
        if last_rate is None:
            last_rate = 0.0

        last_lg = logit(last_rate)

        # Collect trophy history in window
        t_list = []
        lg_list = []
        E_list = []
        N_list = []
        x_list = []

        for r in window_reads:
            rt = r["rates"][i]
            if rt is not None and r["N"] > 0:
                t_list.append(r["t_hours"])
                x_list.append(r["x"])
                lg_list.append(logit(rt))
                E = (rt * r["N"]) / 100.0
                E_list.append(E)
                N_list.append(r["N"])

        # 1. Logit-trend fit
        trend_lg = last_lg
        trend_res_var = 0.0
        unique_x = set(x_list)
        if len(unique_x) >= 3:
            target_x = math.log(1.0 + max(0.0, target_t_hours))
            ta, tb, trend_res_var = linear_fit(x_list, lg_list)
            trend_lg = ta + tb * target_x

        # 2. Earned-vs-tracked fit
        en_lg = last_lg
        unique_N = set(N_list)
        if len(unique_N) >= 3:
            ea, eb, _ = linear_fit(N_list, E_list)
            forecast_E = ea + eb * forecast_N
            forecast_r = 100.0 * forecast_E / forecast_N
            forecast_r = clamp(forecast_r)
            en_lg = logit(forecast_r)

        # 3. Median combination
        has_trend = len(unique_x) >= 3
        has_en = len(unique_N) >= 3

        if has_trend and has_en:
            median_lg = 0.5 * trend_lg + 0.5 * en_lg
        elif has_trend:
            median_lg = trend_lg
        elif has_en:
            median_lg = en_lg
        else:
            median_lg = last_lg

        # Clamp median logit move from last read's logit to at most +/- 1.5
        median_lg = max(last_lg - 1.5, min(last_lg + 1.5, median_lg))

        # 4. Flat-history refresh prior
        # Shift median logit down by 0.16 (odds x 0.85) for every trophy except one whose last rate is 100.
        if apply_flat_prior:
            if abs(last_rate - 100.0) > 1e-5:
                median_lg -= 0.16

        median_rate = inv_logit(median_lg)

        # 5. Spread sigma (logit units)
        # sigma^2 = (0.5^2 * horizon_days) + residual_var + jump_var
        # jump_var = (largest |delta logit(r)| between consecutive reads)^2, or 0.35^2 when flat
        max_jump = 0.0
        if is_perfectly_flat:
            max_jump = 0.35
        else:
            # compute max jump across consecutive reads in history for this trophy
            prev_lg = None
            for r in parsed_reads:
                rt = r["rates"][i]
                if rt is not None:
                    curr_lg = logit(rt)
                    if prev_lg is not None:
                        jump = abs(curr_lg - prev_lg)
                        if jump > max_jump:
                            max_jump = jump
                    prev_lg = curr_lg

        jump_var = max_jump ** 2
        var_total = (0.25 * horizon_days) + trend_res_var + jump_var
        sigma = math.sqrt(max(0.0, var_total))
        sigma = max(0.25, min(2.0, sigma))
        if is_perfectly_flat:
            # One unseen Exophase refresh over horizon should widen the 90% band
            # to roughly a factor of e^1 up/down in odds, so the band honestly
            # covers the refresh the practice answers show.
            sigma = max(sigma, 0.6)

        # 6. Quantiles
        raw_qs = []
        for z in Z:
            q_val = inv_logit(median_lg + z * sigma)
            # If last rate is 100, keep median near 100 (clamped 99.99)
            if abs(last_rate - 100.0) <= 1e-5:
                q_val = 99.99
            raw_qs.append(q_val)

        # Enforce non-decreasing (cumulative maximum)
        enforced_qs = []
        running_max = 0.01
        for q in raw_qs:
            running_max = max(running_max, q)
            enforced_qs.append(round4(running_max))

        rows.append({"id": trophy["id"], "q": enforced_qs})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
