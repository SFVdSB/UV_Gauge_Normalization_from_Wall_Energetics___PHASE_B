#!/usr/bin/env python3
"""predict_qcd_from_w3.py

End-to-end *consistency check* that combines:
- Phase A wall normalization (tau_summary -> A_xi)
- Phase B kernel-derived weights (derived_wi_*.json)

It fits (C, w1) at two-loop while holding w3 fixed to the kernel-derived value,
so that alpha1(MZ) and alpha2(MZ) match the SM inputs. Then it reports
alpha3(MZ) and Lambda5 (2-loop MSbar formula).

Usage:
  python predict_qcd_from_w3.py \
    --tau-summary tau_summary.json \
    --derived-w derived_wi_modelA_phi.json \
    --m-sfv 3.94 \
    --uv 2.41e14

Notes:
- This avoids the unphysical branch that can appear if you only fit alpha2.
- It uses the same two-loop beta matrices used in your Phase A scripts.
"""

import argparse, json, math
import numpy as np

MZ = 91.1876
XI_CONST = 1.3953e-16  # m*GeV

BETA_1L = np.array([41.0/10.0, -19.0/6.0, -7.0])
B2MAT = np.array([
    [199.0/50.0, 27.0/10.0, 44.0/5.0],
    [9.0/10.0,   35.0/6.0,  12.0     ],
    [11.0/10.0,  9.0/2.0,   -26.0    ]
])


def read_tau_and_delta(path: str) -> tuple[float, float]:
    """Read Phase-A tau summary JSON and return (tau_tilde_solver_units, Delta_rS_solver_units).

    The Phase-A pipeline has used a few key names over time; we accept all known variants:
      - tau: tau_tilde_solver_units / tau_tilde_o4_solver_units / tau_tilde_wall_only_solver_units
      - Delta: Delta_rS_solver_units / Delta_rS_csv / background_profile_consistency.Delta_rS_csv / background_profile.Delta_rS
    """
    import os, json
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)

    tau = js.get("tau_tilde_solver_units")
    if tau is None:
        tau = js.get("tau_tilde_o4_solver_units") or js.get("tau_tilde_wall_only_solver_units")
    if tau is None:
        raise RuntimeError(
            f"tau_tilde_solver_units not found in tau summary: {os.path.abspath(path)}; keys={list(js.keys())}"
        )

    Delta = (
        js.get("Delta_rS_solver_units")
        or js.get("Delta_rS_csv")
        or js.get("background_profile_consistency", {}).get("Delta_rS_solver_units")
        or js.get("background_profile_consistency", {}).get("Delta_rS_csv")
        or js.get("background_profile", {}).get("Delta_rS")
    )
    if Delta is None:
        bpc = js.get("background_profile_consistency", {})
        raise RuntimeError(
            "Delta_rS not found in tau summary. Looked for: "
            "Delta_rS_solver_units, Delta_rS_csv, background_profile_consistency.Delta_rS_csv, background_profile.Delta_rS. "
            f"File={os.path.abspath(path)}; top keys={list(js.keys())}; "
            f"background_profile_consistency keys={list(bpc.keys()) if isinstance(bpc, dict) else type(bpc)}"
        )

    return float(tau), float(Delta)
def alpha_inputs(alpha_em: float, sin2: float) -> tuple[float, float]:
    cos2 = 1.0 - sin2
    alpha2_MZ = alpha_em / sin2
    alpha1_MZ = (5.0/3.0) * alpha_em / cos2
    return alpha1_MZ, alpha2_MZ


def run_two_loop(alpha_init, mu_init, mu_final, steps=1200):
    al = np.array(alpha_init, dtype=float)
    t0, t1 = math.log(mu_init), math.log(mu_final)
    dt = (t1 - t0)/steps

    def beta(a):
        a = np.clip(a, 1e-12, 10.0)
        one = (BETA_1L/(2.0*math.pi)) * (a**2)
        sumterm = B2MAT @ a
        two = (a**2) * (sumterm) / (8.0*math.pi**2)
        return one + two

    for _ in range(steps):
        k1 = beta(al)
        k2 = beta(al + 0.5*dt*k1)
        k3 = beta(al + 0.5*dt*k2)
        k4 = beta(al + dt*k3)
        al = al + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return al


def lambda_qcd_two_loop(alpha_s: float, mu: float = MZ, nf: int = 5) -> float:
    if alpha_s <= 0 or not math.isfinite(alpha_s):
        return float("nan")
    beta0 = 11.0 - 2.0/3.0*nf
    beta1 = 102.0 - 38.0/3.0*nf
    X = (4.0*math.pi)/(beta0*alpha_s)
    if X <= 0 or not math.isfinite(X):
        return float("nan")
    L = X - (beta1/(beta0**2))*math.log(max(X, 1e-30))
    return mu * math.exp(-0.5*L)


def fit_C_w1_fixed_w3(A_xi: float, MU: float, alpha1_t: float, alpha2_t: float, w3: float,
                      w2: float = 1.0, steps: int = 1200, iters: int = 8):
    # 1-loop seed
    b1, b2, _ = BETA_1L
    lnU = math.log(MU/MZ)
    lhs1 = 1.0/alpha1_t - (b1/(2.0*math.pi))*lnU
    lhs2 = 1.0/alpha2_t - (b2/(2.0*math.pi))*lnU
    C = lhs2/(4.0*math.pi*A_xi*w2)
    w1 = lhs1/lhs2

    def shoot(Cv, w1v):
        aUV = np.array([
            1.0/(4.0*math.pi*Cv*w1v*A_xi),
            1.0/(4.0*math.pi*Cv*w2 *A_xi),
            1.0/(4.0*math.pi*Cv*w3 *A_xi),
        ])
        return run_two_loop(aUV, MU, MZ, steps=steps)

    for _ in range(iters):
        al = shoot(C, w1)
        f = np.array([al[0]-alpha1_t, al[1]-alpha2_t])

        dC = C*1e-4 + 1e-16
        dw = w1*1e-4 + 1e-16
        al_C = shoot(C+dC, w1)
        al_w = shoot(C, w1+dw)
        J = np.array([
            [(al_C[0]-al[0])/dC, (al_w[0]-al[0])/dw],
            [(al_C[1]-al[1])/dC, (al_w[1]-al[1])/dw],
        ])
        delta = np.linalg.lstsq(J, -f, rcond=None)[0]
        C += float(delta[0])
        w1 += float(delta[1])

    al = shoot(C, w1)
    return C, w1, al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-summary", required=True)
    ap.add_argument("--derived-w", required=True, help="JSON that contains derived_weights.w3_over_w2")
    ap.add_argument("--m-sfv", type=float, default=3.94)
    ap.add_argument("--uv", type=float, default=2.41e14)
    ap.add_argument("--alpha-em", type=float, default=1/127.955)
    ap.add_argument("--sin2", type=float, default=0.23122)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--output", default="qcd_prediction_from_w3.json")
    args = ap.parse_args()

    tau, Delta = read_tau_and_delta(args.tau_summary)
    xi = XI_CONST / args.m_sfv
    A_xi = (xi*xi) * (tau/Delta)

    dw = json.load(open(args.derived_w, "r"))
    w1_kernel = float(dw["derived_weights"]["w1_over_w2"])
    w3 = float(dw["derived_weights"]["w3_over_w2"])

    a1_t, a2_t = alpha_inputs(args.alpha_em, args.sin2)

    C, w1_fit, al = fit_C_w1_fixed_w3(A_xi, args.uv, a1_t, a2_t, w3, steps=args.steps)

    a1, a2, a3 = [float(x) for x in al]
    lam5 = float(lambda_qcd_two_loop(a3, mu=MZ, nf=5))

    out = {
        "inputs": {
            "tau_summary": args.tau_summary,
            "derived_w": args.derived_w,
            "m_sfv_GeV": args.m_sfv,
            "MU_UV_GeV": args.uv,
            "alpha_em_MZ": args.alpha_em,
            "sin2thetaW_MZ": args.sin2,
            "steps": args.steps,
        },
        "A_xi_m2": A_xi,
        "derived_weights": {"w1_over_w2_kernel": w1_kernel, "w3_over_w2": w3},
        "two_loop_fit": {
            "C_units_inv_m2": C,
            "w1_over_w2_fit": w1_fit,
            "alpha_MZ": {"alpha1": a1, "alpha2": a2, "alpha3": a3},
            "Lambda5_two_loop_GeV": lam5,
        }
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print("[QCD prediction]")
    print(f"  w3/w2 (kernel) = {w3:.9f}")
    print(f"  alpha_s(MZ)    = {a3:.9f}")
    print(f"  Lambda5 (2L)   = {lam5:.6f} GeV")
    print(f"  wrote          = {args.output}")


if __name__ == "__main__":
    main()
