#!/usr/bin/env python3
"""
gauge_normalizer_from_wall.py  —  Phase A (Route A) gauge normalization

Compute gauge-kinetic normalization constants from wall energetics and run
SM gauge couplings to MZ using either 1-loop closed forms or a 2-loop
pure-gauge integrator.

Inputs
------
  --tau-summary   JSON produced by wall_tension_from_shell.py or O(4) variant
                  Must contain:
                    - tau_tilde_solver_units  (or tau_tilde_o4_solver_units,
                                               or tau_tilde_wall_only_solver_units)
                    - background_profile_consistency.Delta_rS_solver_units
                      (or background_profile.Delta_rS)

  --msfv          One or more SFV mass values in GeV (default: 3.94)
  --uv            UV matching scale in GeV (default: 2.41e14)
  --alpha3-target Target alpha_s(MZ) (default: 0.1181)
  --alpha-em      alpha_em(MZ) (default: 1/127.955)
  --sin2          sin^2(theta_W)(MZ) (default: 0.23122)
  --w2            Reference weight for SU(2) (default: 1.0)
  --two-loop      Use 2-loop pure-gauge RGE (default off => 1-loop analytic)
  --steps         (two-loop) RK4 steps from UV->MZ (default 1200)
  --lambda-report Also compute Lambda_QCD^{MSbar} at two-loop from alpha_s(MZ)
  --output        Output JSON path (default phaseA_routeA_report.json)
"""
import argparse, json, math
import numpy as np

XI_CONST = 1.3953e-16  # m*GeV
MZ = 91.1876

BETA_1L = np.array([41.0/10.0, -19.0/6.0, -7.0])  # (b1, b2, b3)
B2MAT = np.array([
    [199.0/50.0, 27.0/10.0, 44.0/5.0],
    [9.0/10.0,   35.0/6.0,  12.0     ],
    [11.0/10.0,  9.0/2.0,   -26.0    ]
])

def read_tau_and_delta(path):
    with open(path, "r") as f:
        js = json.load(f)
    tau = None
    for k in ["tau_tilde_solver_units","tau_tilde_o4_solver_units","tau_tilde_wall_only_solver_units"]:
        if k in js:
            tau = float(js[k]); break
    if tau is None and "inputs" in js and "tau_tilde_o4_solver_units" in js["inputs"]:
        tau = float(js["inputs"]["tau_tilde_o4_solver_units"])
    Delta = None
    if "background_profile_consistency" in js and "Delta_rS_solver_units" in js["background_profile_consistency"]:
        Delta = float(js["background_profile_consistency"]["Delta_rS_solver_units"])
    if "background_profile" in js and "Delta_rS" in js["background_profile"]:
        Delta = float(js["background_profile"]["Delta_rS"])
    if tau is None or Delta is None:
        raise RuntimeError("Could not read tau_tilde and/or Delta_rS from the tau-summary JSON.")
    return tau, Delta

def alpha_inputs(alpha_em, sin2):
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
        k1 = beta(al); k2 = beta(al + 0.5*dt*k1)
        k3 = beta(al + 0.5*dt*k2); k4 = beta(al + dt*k3)
        al = al + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return al

def solve_one_loop(A_xi, MU, alpha1_MZ, alpha2_MZ, alpha3_target, w2=1.0):
    b1, b2, b3 = BETA_1L
    lnU = math.log(MU/MZ)
    lhs1 = 1.0/alpha1_MZ - (b1/(2.0*math.pi))*lnU
    lhs2 = 1.0/alpha2_MZ - (b2/(2.0*math.pi))*lnU
    C = lhs2/(4.0*math.pi*A_xi*w2)
    w1 = lhs1/lhs2
    alpha3_inv_MZ = 4.0*math.pi*C*1.0*A_xi + (b3/(2.0*math.pi))*lnU
    alpha3_pred = 1.0/alpha3_inv_MZ
    target_inv = 1.0/alpha3_target
    w3_req = (target_inv - (b3/(2.0*math.pi))*lnU) / (4.0*math.pi*C*A_xi)
    return C, w1, alpha3_pred, w3_req

def solve_two_loop(A_xi, MU, alpha1_MZ, alpha2_MZ, alpha3_target, w2=1.0, steps=1200, newton_iters=6):
    b1, b2, b3 = BETA_1L
    lnU = math.log(MU/MZ)
    lhs1 = 1.0/alpha1_MZ - (b1/(2.0*math.pi))*lnU
    lhs2 = 1.0/alpha2_MZ - (b2/(2.0*math.pi))*lnU
    C = lhs2/(4.0*math.pi*A_xi*w2)
    w1 = lhs1/lhs2
    def shoot(C, w1, w3=1.0):
        aUV = np.array([1.0/(4.0*math.pi*C*w1*A_xi),
                        1.0/(4.0*math.pi*C*w2*A_xi),
                        1.0/(4.0*math.pi*C*w3*A_xi)])
        return run_two_loop(aUV, MU, MZ, steps=steps)
    for _ in range(newton_iters):
        al = shoot(C, w1)
        f1, f2 = al[1] - alpha2_MZ, al[0] - alpha1_MZ
        dC = C*1e-4 + 1e-16; dw = w1*1e-4 + 1e-16
        al_C = shoot(C+dC, w1); al_w = shoot(C, w1+dw)
        J = np.array([[ (al_C[1]-al[1])/dC, (al_w[1]-al[1])/dw ],
                      [ (al_C[0]-al[0])/dC, (al_w[0]-al[0])/dw ]])
        delta = np.linalg.lstsq(J, -np.array([f1,f2]), rcond=None)[0]
        C += delta[0]; w1 += delta[1]
    alpha3_pred_w3eq1 = shoot(C, w1, 1.0)[2]
    def f(w3): return shoot(C, w1, w3)[2] - alpha3_target
    lo, hi = 0.5, 1.5
    flo, fhi = f(lo), f(hi)
    for _ in range(20):
        if flo*fhi < 0: break
        lo *= 0.8; hi *= 1.2; flo, fhi = f(lo), f(hi)
    for _ in range(32):
        mid = 0.5*(lo+hi); fm = f(mid)
        if flo*fm <= 0: hi, fhi = mid, fm
        else: lo, flo = mid, fm
    w3 = 0.5*(lo+hi)
    return C, w1, alpha3_pred_w3eq1, w3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-summary", required=True)
    ap.add_argument("--msfv", type=float, nargs="+", default=[3.94])
    ap.add_argument("--uv", type=float, default=2.41e14)
    ap.add_argument("--alpha3-target", type=float, default=0.1181)
    ap.add_argument("--alpha-em", type=float, default=1/127.955)
    ap.add_argument("--sin2", type=float, default=0.23122)
    ap.add_argument("--w2", type=float, default=1.0)
    ap.add_argument("--two-loop", action="store_true")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lambda-report", action="store_true")
    ap.add_argument("--output", default="phaseA_routeA_report.json")
    args = ap.parse_args()

    tau_tilde, Delta_rS = read_tau_and_delta(args.tau_summary)
    alpha1_MZ, alpha2_MZ = alpha_inputs(args.alpha_em, args.sin2)

    results = {}
    for m in args.msfv:
        xi = 1.3953e-16 / m
        A_xi = xi*xi * (tau_tilde/Delta_rS)
        if args.two_loop:
            C, w1, a3_pred, w3_req = solve_two_loop(A_xi, args.uv, alpha1_MZ, alpha2_MZ,
                                                    args.alpha3_target, w2=args.w2,
                                                    steps=args.steps, newton_iters=6)
        else:
            C, w1, a3_pred, w3_req = solve_one_loop(A_xi, args.uv, alpha1_MZ, alpha2_MZ,
                                                    args.alpha3_target, w2=args.w2)
        entry = {
            "inputs": {"m_sfv_GeV": m, "MU_UV_GeV": args.uv, "two_loop": bool(args.two_loop), "steps": (args.steps if args.two_loop else None)},
            "calibration": {"xi_m": xi, "A_xi_m2": A_xi},
            "fit": {"C_units_inv_m2": C, "w1_over_w2": w1},
            "predictions": {"alpha1_MZ_match": alpha1_MZ, "alpha2_MZ_match": alpha2_MZ,
                            "alpha3_MZ_pred_w3eq1": a3_pred,
                            "required_w3_over_w2_for_target_alpha3": w3_req}
        }
        if args.lambda_report:
            def LQCD(a): 
                beta0 = 11.0 - 2.0/3.0*5
                beta1 = 102.0 - 38.0/3.0*5
                X = (4.0*math.pi)/(beta0*a)
                L = X - (beta1/(beta0**2))*math.log(X)
                return MZ * math.exp(-0.5*L)
            entry["Lambda5_two_loop_GeV"] = {
                "from_predicted_alpha3": LQCD(a3_pred),
                "from_target_alpha3":    LQCD(args.alpha3_target)
            }
        results[str(m)] = entry

    out = {"tau_summary": args.tau_summary, "Delta_rS_solver_units": Delta_rS,
           "tau_tilde_solver_units": tau_tilde, "results": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")

if __name__=="__main__":
    main()
