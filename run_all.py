"""
run_all.py -- Yield Curve Construction & Forecasting: end-to-end pipeline
=========================================================================
Runs the full project on real U.S. Treasury data (bundled offline sample by
default; pass --online to download the latest history) plus a calibrated
simulator for the data-hungry pieces, and writes all figures + a RESULTS.md
into ./outputs.

Stages
------
1.  Load & preprocess yields
2.  Static curve construction  -- fit NS & NSS (scipy.optimize)
3.  Bootstrap zero curve + bond & derivative pricing
4.  Dynamic Nelson-Siegel      -- factor extraction (two-step + Kalman MLE)
5.  Out-of-sample forecast horse-race (RW / ARIMA / VAR / ML / DNS)
6.  Forecast the curve forward with uncertainty bands
7.  From-scratch LSTM on simulated history
8.  Forecast-driven trading-strategy backtest
9.  Write figures + RESULTS.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from yclib import (data_loader, fitting, bonds, dns as dns_mod, plotting,
                   simulate, backtest)
from yclib.arima import ARIMA
from yclib.var_model import VAR
from yclib.ml_forecast import MLForecaster
from yclib.lstm import LSTMForecaster

ROOT = os.path.dirname(__file__)
OUT = os.path.join(ROOT, "outputs")


def banner(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70)


# ---------------------------------------------------------------------------
def stage_data(online: bool):
    banner("STAGE 1  Load & preprocess Treasury yields")
    src = "auto" if online else "bundled"
    df = data_loader.get_yields(source=src)
    df = data_loader.preprocess(df)
    print(f"Loaded {df.shape[0]} curves x {df.shape[1]} maturities "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
    print("Latest curve (%):")
    print("  " + "  ".join(f"{t:g}y={df.iloc[-1][t]:.2f}" for t in df.columns))
    return df


def stage_static(df, figs):
    banner("STAGE 2  Static curve construction (NS / NSS)")
    tau = df.columns.values.astype(float)
    y = df.iloc[-1].values.astype(float)
    ns = fitting.fit_ns(tau, y)
    nss = fitting.fit_nss(tau, y)
    print(f"NS : RMSE={ns.rmse*100:.2f} bps  max|err|={ns.max_abs_err*100:.2f} bps  "
          f"beta={np.round(ns.beta,3)}  lambda={np.round(ns.lams,3)}")
    print(f"NSS: RMSE={nss.rmse*100:.2f} bps  max|err|={nss.max_abs_err*100:.2f} bps  "
          f"beta={np.round(nss.beta,3)}  lambda={np.round(nss.lams,3)}")
    date = df.index[-1].date()
    figs["fit"] = plotting.plot_fit_vs_market(
        tau, y, nss.curve, OUT, title=f"NSS yield-curve fit  ({date})")
    figs["ns_vs_nss"] = plotting.plot_ns_vs_nss(tau, y, ns.curve, nss.curve, OUT)
    return ns, nss


def stage_pricing(df):
    banner("STAGE 3  Bootstrap zero curve + bond/derivative pricing")
    tau = df.columns.values.astype(float)
    y = df.iloc[-1].values.astype(float)
    zc = bonds.ZeroCurve.from_par(tau, y, freq=2, max_t=30,
                                  ref_date=str(df.index[-1].date()))
    print("Bootstrapped zero (spot) rates, cont-comp %:")
    for t in (0.5, 1, 2, 5, 10, 30):
        print(f"  {t:>4}y: zero={zc.zero_rate(t):.3f}  DF={zc.discount(t):.4f}")

    rows = []
    for cpn, mat in [(4.0, 2), (4.0, 5), (4.0, 10), (5.0, 30)]:
        px = bonds.price_bond(zc, mat, cpn, 2, 100)
        r = bonds.bond_risk(px, mat, cpn, 2, 100)
        rows.append((f"{cpn:.0f}% {mat}y", px, r["ytm_pct"], r["modified_dur"],
                     r["convexity"], r["dv01"]))
    print("\nBond pricing & risk (per 100 face):")
    print(f"  {'bond':<9}{'price':>9}{'ytm%':>8}{'modDur':>9}{'convex':>9}{'DV01':>8}")
    for nm, px, yt, md, cv, dv in rows:
        print(f"  {nm:<9}{px:>9.3f}{yt:>8.3f}{md:>9.3f}{cv:>9.2f}{dv:>8.4f}")

    print("\nInterest-rate derivatives:")
    for T in (2, 5, 10, 30):
        print(f"  par swap {T:>2}y fixed rate: {bonds.par_swap_rate(zc, T, 2):.3f}%")
    sv = bonds.swap_value(zc, 4.0, 5, 2, 10_000_000, payer=True)
    print(f"  5y payer swap @4% on $10mm: PV = ${sv:,.0f}")
    fra = bonds.fra_rate(zc, 1.0, 1.5)
    print(f"  FRA 1y x 1.5y (simple fwd): {fra:.3f}%")
    cap = bonds.black_caplet(zc, 1.0, 1.5, 4.0, 20.0, 1_000_000)
    print(f"  Caplet 1y->1.5y K=4% vol=20% on $1mm: ${cap:,.0f}")
    return zc


def stage_dns(df, figs):
    banner("STAGE 4  Dynamic Nelson-Siegel (factor model)")
    m2 = dns_mod.DynamicNelsonSiegel().fit(df, method="two_step")
    r = m2.result_
    print(f"Fixed lambda = {r.lam:.3f}   cross-sectional fit RMSE = {r.fit_rmse_bps:.2f} bps")
    print(f"Long-run factor means (L,S,C): {np.round(r.mu,3)}")
    print(f"VAR(1) persistence diag       : {np.round(np.diag(r.Phi),3)}")
    print(f"Implied short rate L+S        : {r.mu[0]+r.mu[1]:.3f}%   long rate L: {r.mu[0]:.3f}%")
    figs["factors"] = plotting.plot_factors(r.factors, OUT)
    figs["surface"] = plotting.plot_surface(df, OUT)
    figs["heatmap"] = plotting.plot_heatmap(df, OUT)

    try:
        mk = dns_mod.DynamicNelsonSiegel().fit(df, method="kalman")
        print(f"Kalman-MLE refinement RMSE    : {mk.result_.fit_rmse_bps:.2f} bps   "
              f"persistence diag {np.round(np.diag(mk.result_.Phi),3)}")
    except Exception as e:
        print("  (Kalman MLE skipped:", str(e)[:50], ")")
    return m2


def _oos_eval(df, h=5, warmup=45):
    """Rolling h-step-ahead forecast RMSE (bps) for each model on the 10y."""
    tenors = [c for c in df.columns if c in (2.0, 5.0, 10.0, 30.0)]
    target = 10.0
    errs = {k: [] for k in ["RW", "ARIMA", "VAR", "ML", "DNS"]}
    for t in range(warmup, len(df) - h):
        train = df.iloc[:t + 1]
        actual = df.iloc[t + h][target]
        # Random walk
        errs["RW"].append(train.iloc[-1][target] - actual)
        # ARIMA on 10y level
        try:
            a = ARIMA((1, 1, 1)).fit(train[target].values)
            errs["ARIMA"].append(a.forecast(h)[-1] - actual)
        except Exception:
            errs["ARIMA"].append(np.nan)
        # VAR on tenor block
        try:
            v = VAR(1).fit(train[tenors])
            fc = v.forecast(h)[-1]
            errs["VAR"].append(fc[tenors.index(target)] - actual)
        except Exception:
            errs["VAR"].append(np.nan)
        # ML
        try:
            ml = MLForecaster(model="ridge", n_lags=5).fit(train[tenors])
            errs["ML"].append(ml.forecast(h).iloc[-1][target] - actual)
        except Exception:
            errs["ML"].append(np.nan)
        # DNS
        try:
            d = dns_mod.DynamicNelsonSiegel().fit(train, method="two_step")
            fc = d.forecast_curve(h, tau=np.array([target]))
            errs["DNS"].append(fc.iloc[-1][target] - actual)
        except Exception:
            errs["DNS"].append(np.nan)
    rmse = {k: float(np.sqrt(np.nanmean(np.array(v) ** 2)) * 100) for k, v in errs.items()}
    return rmse


def stage_forecast(df, dns_model, figs):
    banner("STAGE 5  Out-of-sample forecast horse-race (10y, 5-day ahead)")
    rmse = _oos_eval(df, h=5, warmup=45)
    order = sorted(rmse, key=rmse.get)
    print("Rolling 5-day-ahead RMSE (bps):")
    for k in order:
        flag = "  <- best" if k == order[0] else ""
        print(f"  {k:<6} {rmse[k]:6.2f}{flag}")
    figs["rmse"] = plotting.plot_model_comparison(
        rmse, OUT, title="OOS 5-day-ahead 10y forecast RMSE")

    banner("STAGE 6  Forecast the curve forward (20 business days)")
    fc, lo, hi = dns_model.forecast_curve(20, return_band=True)
    print("DNS 20-day-ahead forecast (selected tenors, %):")
    for t in [0.25, 2, 10, 30]:
        col = min(fc.columns, key=lambda c: abs(c - t))
        print(f"  {col:>5g}y: {fc.iloc[-1][col]:.3f}  "
              f"[{lo.iloc[-1][col]:.3f}, {hi.iloc[-1][col]:.3f}]")
    figs["forecast"] = plotting.plot_forecast(
        df.iloc[-40:], fc, lo, hi, tenor=10.0, outdir=OUT)
    return rmse


def stage_lstm(real_df, figs):
    banner("STAGE 7  From-scratch NumPy LSTM (trained on simulated history)")
    sim = simulate.calibrate_from_real(real_df, n_days=1200, seed=11)
    print(f"Calibrated synthetic panel: {sim.shape} (SYNTHETIC, for training only)")
    tenors = [c for c in sim.columns if c in (2.0, 10.0, 30.0)]
    series = sim[tenors]
    t0 = time.time()
    lstm = LSTMForecaster(window=12, n_hidden=20, lr=0.02, epochs=45, seed=0)
    lstm.fit(series.values, verbose=False)
    lh = lstm.loss_history_
    print(f"Trained {len(lh)} epochs in {time.time()-t0:.1f}s   "
          f"loss {lh[0]:.4f} -> {lh[-1]:.4f}  ({lh[0]/max(lh[-1],1e-9):.1f}x lower)")
    fc = lstm.forecast(10)
    print(f"LSTM 10-step forecast (10y): "
          f"{', '.join(f'{v:.2f}' for v in fc[:, tenors.index(10.0)][:5])} ...")
    figs["lstm_loss"] = plotting.plot_loss(lh, OUT)
    return sim


def stage_backtest(sim, figs):
    banner("STAGE 8  Forecast-driven trading-strategy backtest")
    # Use a manageable slice of the simulated panel for speed
    panel = sim.iloc[-500:]
    results = {}
    print("Duration-timing strategy (trade 10y on predicted direction):")
    tbl = backtest.compare_models(panel, strategy="duration",
                                  models=("rw", "var", "ml", "dns"),
                                  warmup=60, refit_every=5)
    print(tbl.to_string(index=False))
    for m in ("rw", "var", "ml", "dns"):
        try:
            results[m.upper()] = backtest.backtest(
                panel, strategy="duration", model=m, warmup=60, refit_every=5)
        except Exception:
            pass
    figs["equity"] = plotting.plot_equity(
        results, OUT, title="Duration-timing equity curve (simulated)")

    print("\n2s10s slope strategy (steepener/flattener on predicted slope):")
    tbl2 = backtest.compare_models(panel, strategy="slope",
                                   models=("rw", "var", "ml", "dns"),
                                   warmup=60, refit_every=5)
    print(tbl2.to_string(index=False))
    return tbl, tbl2


def write_results(df, ns, nss, rmse, bt_tbl, figs):
    banner("STAGE 9  Writing RESULTS.md")
    os.makedirs(OUT, exist_ok=True)
    lines = []
    A = lines.append
    A("# Results — Yield Curve Construction & Forecasting\n")
    A(f"_Generated by `run_all.py`. Real data: {df.index[0].date()} to "
      f"{df.index[-1].date()} ({len(df)} curves, {df.shape[1]} tenors)._\n")
    A("## 1. Static curve fit (latest date)\n")
    A(f"- **NS** : RMSE **{ns.rmse*100:.2f} bps**, params "
      f"β={np.round(ns.beta,3).tolist()}, λ={np.round(ns.lams,3).tolist()}")
    A(f"- **NSS**: RMSE **{nss.rmse*100:.2f} bps**, params "
      f"β={np.round(nss.beta,3).tolist()}, λ={np.round(nss.lams,3).tolist()}\n")
    A("![curve fit](outputs/01_curve_fit.png)\n")
    A("![ns vs nss](outputs/02_ns_vs_nss.png)\n")
    A("## 2. Dynamic factors\n")
    A("![factors](outputs/03_factors.png)\n")
    A("## 3. Out-of-sample 5-day-ahead 10y forecast RMSE (bps)\n")
    A("| model | RMSE (bps) |")
    A("|---|---|")
    for k in sorted(rmse, key=rmse.get):
        A(f"| {k} | {rmse[k]:.2f} |")
    A("\n![rmse](outputs/07_model_rmse.png)\n")
    A("![forecast](outputs/06_forecast.png)\n")
    A("## 4. Trading strategy backtest (simulated panel)\n")
    A(bt_tbl.to_markdown(index=False) if hasattr(bt_tbl, "to_markdown")
      else bt_tbl.to_string(index=False))
    A("\n\n![equity](outputs/08_backtest_equity.png)\n")
    A("## 5. From-scratch LSTM training\n")
    A("![lstm](outputs/09_lstm_loss.png)\n")
    path = os.path.join(ROOT, "RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("Wrote", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true",
                    help="download latest Treasury data instead of bundled sample")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    figs = {}
    t0 = time.time()

    df = stage_data(args.online)
    ns, nss = stage_static(df, figs)
    stage_pricing(df)
    dmodel = stage_dns(df, figs)
    rmse = stage_forecast(df, dmodel, figs)
    sim = stage_lstm(df, figs)
    bt_tbl, _ = stage_backtest(sim, figs)
    results_path = write_results(df, ns, nss, rmse, bt_tbl, figs)

    banner("DONE")
    print(f"Total runtime: {time.time()-t0:.1f}s")
    print(f"Figures written to: {OUT}")
    for k, v in figs.items():
        print(f"  {k:<10} {os.path.basename(v)}")
    print(f"Summary: {results_path}")


if __name__ == "__main__":
    main()
