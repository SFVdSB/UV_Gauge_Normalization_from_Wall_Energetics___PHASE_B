#!/usr/bin/env python3
"""
qcd_rge.py

Convert α_s(MZ) to Λ_QCD (MSbar) at one- and two-loop level.
"""
import argparse, json, math

MZ = 91.1876

def lambda_qcd_one_loop(alpha_s, mu, nf):
    beta0 = 11.0 - 2.0/3.0*nf
    return mu * math.exp(-(2.0*math.pi)/(beta0*alpha_s))

def lambda_qcd_two_loop(alpha_s, mu, nf):
    beta0 = 11.0 - 2.0/3.0*nf
    beta1 = 102.0 - 38.0/3.0*nf
    X = (4.0*math.pi)/(beta0*alpha_s)
    L = X - (beta1/(beta0**2))*math.log(X)
    return mu * math.exp(-0.5*L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha3", type=float, default=0.1181, help="alpha_s(MZ)")
    ap.add_argument("--mu", type=float, default=MZ, help="Scale for alpha_s (GeV)")
    ap.add_argument("--nf", type=int, default=5, help="Active flavors for Λ extraction")
    ap.add_argument("--output", default="qcd_lambda_report.json", help="Output JSON")
    args = ap.parse_args()

    lam1 = lambda_qcd_one_loop(args.alpha3, args.mu, args.nf)
    lam2 = lambda_qcd_two_loop(args.alpha3, args.mu, args.nf)

    out = {
        "inputs": {"alpha3": args.alpha3, "mu_GeV": args.mu, "nf": args.nf},
        "Lambda_QCD_one_loop_GeV": lam1,
        "Lambda_QCD_two_loop_GeV": lam2
    }
    with open(args.output,"w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")

if __name__=="__main__":
    main()
