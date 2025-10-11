#!/usr/bin/env python3
"""
derive_wi_from_kernel.py  — v2 with --fit-w minimal-norm solver

What's new
----------
- --fit-w W1 W3 : solve for Model-A coefficients (c's) that *exactly* hit
  target weights w1/w2=W1 and w3/w2=W3 using the minimal-norm solution.
  By default SU(2) coefficients (c2Phi,c2phi) are allowed; use --fix-su2
  to set them to zero (baseline group).

- Robust two-loop check: never crashes on unphysical αs; Λ5 becomes NaN.

Model A kernel
--------------
K_i(z) = k0 + c_{iPhi} * SPhi(z) + c_{iphi} * Sphi(z),
with basis integrals B0 = ∫dz, BPhi = ∫SPhi, Bphi = ∫Sphi over the wall window.
I_i = ∫ K_i = k0*B0 + c_{iPhi}*BPhi + c_{iphi}*Bphi,  w_i = I_i/I_2.

Minimal-norm fit
----------------
We solve the linear constraints
  w1 I2 - I1 = 0,   w3 I2 - I3 = 0
for the vector x = (c1Phi,c1phi,c2Phi,c2phi,c3Phi,c3phi) with the solution
  x* = A^T (A A^T)^(-1) b
which minimizes ||x||_2 subject to A x = b.  If --fix-su2, we set c2Phi=c2phi=0
and solve two independent 2×1 problems with their minimal-norm solutions.

"""
import argparse, json, math
import numpy as np

MZ = 91.1876

BETA_1L = np.array([41.0/10.0, -19.0/6.0, -7.0])
B2MAT = np.array([
    [199.0/50.0, 27.0/10.0, 44.0/5.0],
    [9.0/10.0,   35.0/6.0,  12.0     ],
    [11.0/10.0,  9.0/2.0,   -26.0    ]
])

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
    return np.maximum(al, 1e-12)

def lambda_qcd_two_loop(alpha_s, mu, nf):
    if alpha_s is None or alpha_s <= 0 or not math.isfinite(alpha_s):
        return float("nan")
    beta0 = 11.0 - 2.0/3.0*nf
    beta1 = 102.0 - 38.0/3.0*nf
    X = (4.0*math.pi)/(beta0*alpha_s)
    if X <= 0 or not math.isfinite(X):
        return float("nan")
    L = X - (beta1/(beta0**2))*math.log(max(X, 1e-30))
    return mu * math.exp(-0.5*L)

def load_basis(path):
    with open(path,"r") as f:
        js = json.load(f)
    B0   = js["integrals"]["B0"]
    BPhi = js["integrals"]["BPhi"]
    Bphi = js["integrals"]["Bphi"]
    return B0, BPhi, Bphi, js

def modelA_integrals(B0,BPhi,Bphi, k0, c1Phi,c1phi,c2Phi,c2phi,c3Phi,c3phi):
    I1 = k0*B0 + c1Phi*BPhi + c1phi*Bphi
    I2 = k0*B0 + c2Phi*BPhi + c2phi*Bphi
    I3 = k0*B0 + c3Phi*BPhi + c3phi*Bphi
    return I1, I2, I3

def minimal_norm_fit_full(B0,BPhi,Bphi,k0,w1,w3):
    # Solve A x = b with x = (c1Phi,c1phi,c2Phi,c2phi,c3Phi,c3phi)
    # Constraints: w1*I2 - I1 = (w1-1)*k0*B0,  w3*I2 - I3 = (w3-1)*k0*B0
    # Build A (2x6) and b (2)
    A = np.array([
        [-BPhi, -Bphi,  w1*BPhi,  w1*Bphi,   0.0,   0.0],
        [ 0.0,   0.0,   w3*BPhi,  w3*Bphi,  -BPhi, -Bphi]
    ], dtype=float)
    b = np.array([(w1 - 1.0)*k0*B0, (w3 - 1.0)*k0*B0], dtype=float)
    # Minimal-norm solution: x = A^T (A A^T)^-1 b
    AA = A @ A.T
    x = A.T @ np.linalg.solve(AA, b)
    return x  # length-6

def minimal_norm_fit_fix_su2(B0,BPhi,Bphi,k0,w1,w3):
    # c2Phi=c2phi=0; solve two independent 2-var problems
    # For group 1: BPhi*c1Phi + Bphi*c1phi = (w1-1)*k0*B0
    denom = (BPhi*BPhi + Bphi*Bphi) if (BPhi*BPhi + Bphi*Bphi)!=0 else 1e-30
    rhs1 = (w1 - 1.0)*k0*B0
    c1Phi = rhs1 * BPhi / denom
    c1phi = rhs1 * Bphi / denom
    rhs3 = (w3 - 1.0)*k0*B0
    c3Phi = rhs3 * BPhi / denom
    c3phi = rhs3 * Bphi / denom
    return (c1Phi, c1phi, 0.0, 0.0, c3Phi, c3phi)

def rge_with_fixed_wi(w1, w2, w3, MU, alpha_em=1/127.955, sin2=0.23122, steps=1200):
    cos2 = 1.0 - sin2
    a2_MZ = alpha_em / sin2
    def shoot(c_log):
        C = math.exp(c_log)
        aUV = np.array([1.0/(4.0*math.pi*C*w1),
                        1.0/(4.0*math.pi*C*w2),
                        1.0/(4.0*math.pi*C*w3)])
        return run_two_loop(aUV, MU, MZ, steps=steps), C
    c_log = 0.0
    for _ in range(10):
        al, C = shoot(c_log)
        f = al[1] - a2_MZ
        al2, _ = shoot(c_log + 1e-4)
        J = (al2[1]-al[1]) / 1e-4
        if not math.isfinite(J) or abs(J) < 1e-14:
            break
        c_log -= f / J
    al, C = shoot(c_log)
    a1_MZ_pred, a2_MZ_match, a3_MZ_pred = [max(x, 1e-12) for x in al]
    return {"C": math.exp(c_log),
            "alpha1_MZ_pred": a1_MZ_pred,
            "alpha2_MZ_match": a2_MZ_match,
            "alpha3_MZ_pred": a3_MZ_pred}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", required=True)
    ap.add_argument("--model", choices=["A","B"], required=True)
    # Model A params
    ap.add_argument("--k0", type=float, default=1.0)
    ap.add_argument("--c1Phi", type=float, default=0.0)
    ap.add_argument("--c1phi", type=float, default=0.0)
    ap.add_argument("--c2Phi", type=float, default=0.0)
    ap.add_argument("--c2phi", type=float, default=0.0)
    ap.add_argument("--c3Phi", type=float, default=0.0)
    ap.add_argument("--c3phi", type=float, default=0.0)
    # Fit targets
    ap.add_argument("--fit-w", type=float, nargs=2, metavar=("W1","W3"),
                    help="Target weights w1/w2 and w3/w2 (Model A only)")
    ap.add_argument("--fix-su2", action="store_true",
                    help="Fix SU(2) coefficients to zero in the fit (Model A)")
    # Model B params
    ap.add_argument("--b1", type=float, default=0.2)
    ap.add_argument("--b2", type=float, default=0.1)
    ap.add_argument("--b3", type=float, default=0.3)
    ap.add_argument("--m0", type=float, default=1.0)
    ap.add_argument("--yPhi", type=float, default=0.1)
    ap.add_argument("--yphi", type=float, default=0.1)
    ap.add_argument("--mu", type=float, default=1.0)
    # RGE
    ap.add_argument("--two-loop-rge", action="store_true")
    ap.add_argument("--uv", type=float, default=2.41e14)
    ap.add_argument("--alpha-em", type=float, default=1/127.955)
    ap.add_argument("--sin2", type=float, default=0.23122)
    ap.add_argument("--alpha3-target", type=float, default=0.1181)
    ap.add_argument("--output", default="derived_wi_report.json")
    args = ap.parse_args()

    B0,BPhi,Bphi,js = load_basis(args.basis)

    if args.model=="A" and args.fit_w is not None:
        W1, W3 = args.fit_w
        if args.fix_su2:
            x = minimal_norm_fit_fix_su2(B0,BPhi,Bphi,args.k0,W1,W3)
        else:
            x = minimal_norm_fit_full(B0,BPhi,Bphi,args.k0,W1,W3)
        (c1Phi,c1phi,c2Phi,c2phi,c3Phi,c3phi) = x
    else:
        c1Phi,c1phi = args.c1Phi, args.c1phi
        c2Phi,c2phi = args.c2Phi, args.c2phi
        c3Phi,c3phi = args.c3Phi, args.c3phi

    if args.model=="A":
        I1,I2,I3 = modelA_integrals(B0,BPhi,Bphi, args.k0, c1Phi,c1phi,c2Phi,c2phi,c3Phi,c3phi)
        w1 = I1/I2
        w2 = 1.0
        w3 = I3/I2
    else:
        # Model B (unchanged): near-unity unless you amplify logs/group factors
        m0, yPhi, yphi, mu = args.m0, args.yPhi, args.yphi, args.mu
        sPhi_avg = BPhi / B0
        sphi_avg = Bphi / B0
        log_term = math.log(max(mu*mu/(m0*m0), 1e-30)) - 2.0*(yPhi*sPhi_avg + yphi*sphi_avg)/max(m0, 1e-12)
        pref = 1.0/(8.0*math.pi**2)
        I1 = B0*(1.0 + pref*args.b1*log_term)
        I2 = B0*(1.0 + pref*args.b2*log_term)
        I3 = B0*(1.0 + pref*args.b3*log_term)
        w1 = I1/I2; w2 = 1.0; w3 = I3/I2

    out = {
        "basis_file": args.basis,
        "integrals": {"B0":B0,"BPhi":BPhi,"Bphi":Bphi},
        "model": args.model,
        "params": {
            "k0": args.k0,
            "c1Phi": c1Phi, "c1phi": c1phi,
            "c2Phi": c2Phi, "c2phi": c2phi,
            "c3Phi": c3Phi, "c3phi": c3phi,
            "fit_w": args.fit_w, "fix_su2": bool(args.fix_su2),
            "b1": args.b1, "b2": args.b2, "b3": args.b3,
            "m0": args.m0, "yPhi": args.yPhi, "yphi": args.yphi, "mu": args.mu
        },
        "derived_weights": {"w1_over_w2": float(w1), "w2":1.0, "w3_over_w2": float(w3)}
    }

    print(f"[derive_wi] w1/w2={w1:.6f}, w3/w2={w3:.6f}")
    if args.two_loop_rge:
        rge = rge_with_fixed_wi(w1, 1.0, w3, args.uv, alpha_em=args.alpha_em, sin2=args.sin2)
        a3 = rge.get("alpha3_MZ_pred", None)
        lam5_pred = lambda_qcd_two_loop(a3, MZ, 5)
        lam5_tgt  = lambda_qcd_two_loop(args.alpha3_target, MZ, 5)
        out["two_loop_rge"] = {
            "alpha1_MZ_pred": rge["alpha1_MZ_pred"],
            "alpha2_MZ_match": rge["alpha2_MZ_match"],
            "alpha3_MZ_pred": a3,
            "Lambda5_two_loop_GeV": {
                "from_predicted_alpha3": lam5_pred,
                "from_target_alpha3":   lam5_tgt
            }
        }

    with open(args.output,"w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")

if __name__=="__main__":
    main()
