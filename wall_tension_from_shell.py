#!/usr/bin/env python3
"""
wall_tension_from_shell.py

Integrate KE+PE across a shell around the wall (using O(3) action-density CSV)
to obtain a wall "tension" proxy τ̃ in solver units.

Inputs:
  --profile   background_profile.csv (needs r, Phi, phi, and ideally w_FWHM, R_peak)
  --o3        o3_action_density.csv (columns: rho, rho2*(KE+PE) OR I3/action_density)
  --width-mult  shell width in units of FWHM (default 3.0 => ±1.5 FWHM)
  --output    JSON path (default phaseA_tau_summary.json)
"""
import argparse, json, math
import numpy as np, pandas as pd

def choose_col(df, names):
    for n in names:
        for c in df.columns:
            if c.strip().lower() == n.lower():
                return c
    return None

def estimate_wall_center_from_grad(df, r_col, Phi_col, phi_col):
    r = df[r_col].to_numpy()
    Phi = df[Phi_col].to_numpy()
    phi = df[phi_col].to_numpy()
    dPhi = np.gradient(Phi, r, edge_order=2)
    dphi = np.gradient(phi, r, edge_order=2)
    score = np.abs(dPhi) + np.abs(dphi)
    idx = int(np.nanargmax(score))
    return float(r[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--o3", required=True)
    ap.add_argument("--width-mult", type=float, default=3.0)
    ap.add_argument("--output", default="phaseA_tau_summary.json")
    args = ap.parse_args()

    prof = pd.read_csv(args.profile)
    o3 = pd.read_csv(args.o3)

    r_col = choose_col(prof, ["r"])
    Phi_col = choose_col(prof, ["Phi","phi1","Phi1"])
    phi_col = choose_col(prof, ["phi","phi2","Phi2"])
    fwhm_col = choose_col(prof, ["w_FWHM","fwhm","width"])
    rpeak_col = choose_col(prof, ["R_peak","r_peak","rwall","r_wall"])

    if r_col is None or Phi_col is None or phi_col is None:
        raise RuntimeError("profile CSV must contain columns for r, Phi, phi.")

    Delta_rS = float(np.nanmean(prof[fwhm_col])) if fwhm_col in prof.columns else None
    R_peak = float(np.nanmean(prof[rpeak_col])) if rpeak_col in prof.columns else None

    R0 = R_peak if R_peak is not None else estimate_wall_center_from_grad(prof, r_col, Phi_col, phi_col)
    width = Delta_rS if Delta_rS is not None else (prof[r_col].iloc[-1] - prof[r_col].iloc[0]) * 0.1

    r_o3_col = choose_col(o3, ["rho","r","radius"])
    if r_o3_col is None:
        raise RuntimeError("o3 CSV needs a radius column (rho/r/radius).")

    cand_A = choose_col(o3, ["rho2*(KE+PE)","r2*(KE+PE)","rho2_kepe"])
    cand_B = choose_col(o3, ["I3","action_density","integrand","integrand_o3"])

    r = o3[r_o3_col].to_numpy().astype(float)
    if cand_A is not None:
        val = o3[cand_A].to_numpy().astype(float)
        eps = val / np.maximum(r,1e-30)**2
    elif cand_B is not None:
        val = o3[cand_B].to_numpy().astype(float)
        eps = val / (4.0*math.pi*np.maximum(r,1e-30)**2)
    else:
        numeric_cols = [c for c in o3.columns if np.issubdtype(o3[c].dtype, np.number) and c != r_o3_col]
        if not numeric_cols:
            raise RuntimeError("No usable energy-density column found in o3 CSV.")
        val = o3[numeric_cols[-1]].to_numpy().astype(float)
        eps = val / (4.0*math.pi*np.maximum(r,1e-30)**2)

    half = 0.5*args.width_mult*width
    rmin, rmax = R0 - half, R0 + half
    mask = (r >= rmin) & (r <= rmax)
    if mask.sum() < 5:
        nearest = np.argsort(np.abs(r - R0))[:200]
        mask = np.zeros_like(r, dtype=bool); mask[nearest] = True

    tau_tilde = float(np.trapz(eps[mask], r[mask]))
    out = {
        "files": {"profile": args.profile, "o3": args.o3},
        "background_profile_consistency": {
            "R_peak": R0, "Delta_rS_solver_units": Delta_rS,
            "shell_window_used": [float(rmin), float(rmax)]
        },
        "tau_tilde_solver_units": tau_tilde,
        "points_integrated": int(mask.sum())
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output} with tau_tilde={tau_tilde:.8g}")

if __name__=="__main__":
    main()
