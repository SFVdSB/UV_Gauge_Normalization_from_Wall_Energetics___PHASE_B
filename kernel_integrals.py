#!/usr/bin/env python3
"""
kernel_integrals.py

Compute wall-window basis integrals for microscopic gauge-kinetic kernels
using the O(4) background profile (moderate-thick walls OK).

Outputs the basis moments:
  B0      = ∫_window dz * 1
  BPhi    = ∫_window dz * (|Phi|^2 - vPhi_baseline^2)
  Bphi    = ∫_window dz * (phi^2    - vphi_baseline^2)

Here z is taken proportional to (r - R_peak) in solver units; any constant
scaling cancels in ratios of group weights w_i.

Inputs:
  --profile    background_profile.csv  (needs columns: r, Phi, phi, w_FWHM, R_peak)
  --shell-mult window size in FWHM units (default 3.0 = ±1.5 FWHM)
  --baseline   FV | TV | avg   (default FV) to define the baseline VEVs
  --output     JSON path (default kernel_basis.json)

Usage:
  python kernel_integrals.py --profile background_profile.csv --output kernel_basis.json
"""
import argparse, json
import numpy as np
import pandas as pd

def compute_basis(df, shell_mult=3.0, baseline='FV'):
    # Required columns
    for col in ["r","Phi","phi","R_peak","w_FWHM"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing required column '{col}' in profile CSV")
    r   = df['r'      ].to_numpy().astype(float)
    Phi = df['Phi'    ].to_numpy().astype(float)
    phi = df['phi'    ].to_numpy().astype(float)
    Rpk = float(np.nanmean(df['R_peak']))
    FWHM= float(np.nanmean(df['w_FWHM']))

    # window in solver units (z ∝ r - Rpk)
    half = 0.5*shell_mult*FWHM
    rmin, rmax = Rpk - half, Rpk + half
    m = (r>=rmin)&(r<=rmax)

    # baselines from ends (robust to moderate thickness)
    n = max(1, int(0.15*len(r)))
    Phi_FV  = float(np.nanmean(Phi[-n:]))
    phi_FV  = float(np.nanmean(phi[-n:]))
    Phi_TV  = float(np.nanmean(Phi[:n]))
    phi_TV  = float(np.nanmean(phi[:n]))

    if baseline.lower()=='fv':
        vPhi2, vphi2 = Phi_FV**2, phi_FV**2
    elif baseline.lower()=='tv':
        vPhi2, vphi2 = Phi_TV**2, phi_TV**2
    else:
        vPhi2, vphi2 = 0.5*(Phi_FV**2+Phi_TV**2), 0.5*(phi_FV**2+phi_TV**2)

    z = r - Rpk
    one = np.ones_like(z)
    sPhi = (Phi**2 - vPhi2)
    sphi = (phi**2 - vphi2)

    # basis moments over the wall window
    B0   = float(np.trapz(one[m], z[m]))
    BPhi = float(np.trapz(sPhi[m], z[m]))
    Bphi = float(np.trapz(sphi[m], z[m]))

    out = {
        "window": {"R_peak":Rpk, "FWHM":FWHM, "shell_mult":shell_mult, "rmin":float(rmin), "rmax":float(rmax)},
        "baselines": {
            "Phi_FV":Phi_FV, "phi_FV":phi_FV, "Phi_TV":Phi_TV, "phi_TV":phi_TV,
            "used": baseline, "vPhi2":vPhi2, "vphi2":vphi2
        },
        "integrals": {"B0":B0, "BPhi":BPhi, "Bphi":Bphi}
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--shell-mult", type=float, default=3.0)
    ap.add_argument("--baseline", choices=["FV","TV","avg"], default="FV")
    ap.add_argument("--output", default="kernel_basis.json")
    args = ap.parse_args()

    df = pd.read_csv(args.profile)
    out = compute_basis(df, shell_mult=args.shell_mult, baseline=args.baseline)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")

if __name__=="__main__":
    main()
