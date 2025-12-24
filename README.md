# UPDATE - Phase B
Run final calculation for Phase B in a single run calculation with predict_qcd_from_w3_patched.py
Requires phaseA_tau_summary.json and derived_wi_modelA_phi.json (generated from the other Phase B scripts)

# SFV/dSB — Phase A (Route A) Toolkit (v2)

This bundle includes:
- `kernel_integrals.py` — Build wall-window basis integrals **directly** from your O(4) profile (moderate-thick walls OK).
- `derive_wi_from_kernel.py` — Two micro-models for the gauge-kinetic kernel 𝒦_i(z) and a two-loop RGE check to produce **derived** (w₁,w₂,w₃).
- `gauge_normalizer_from_wall.py` — Route-A fitter with `--two-loop` flag using τ̃ and Δr̃_S to match α₁, α₂ and predict α₃ (or the w₃ required).
- `wall_tension_from_shell.py` — O(3)-based wall strip integral tool (τ̃ proxy).
- `qcd_rge.py` — Convert αₛ(M_Z) to Λ_QCD at one/two loops.

## Minimal install
```
pip install numpy pandas
```

## A) Derive (w₁,w₂,w₃) from a microscopic kernel (moderate-thick wall handled)
1) Build basis from your profile (uses ±1.5×FWHM by default):
```
python kernel_integrals.py \
  --profile background_profile.csv \
  --shell-mult 3.0 \
  --baseline FV \
  --output kernel_basis.json
```

2) Choose a kernel model and derive wᵢ (plus an RGE check):

**Model A** (dimension-6 EFT; simple small coefficients):
```
python derive_wi_from_kernel.py \
  --basis kernel_basis.json \
  --model A --k0 1.0 \
  --c1Phi 0.12 --c2Phi 0.00 --c3Phi 0.10 \
  --c1phi 0.00 --c2phi 0.00 --c3phi 0.00 \
  --two-loop-rge --uv 2.41e14 \
  --output derived_wi_modelA.json
```

**Model B** (one smooth threshold from vectorlike X; pick your group factors):
```
python derive_wi_from_kernel.py \
  --basis kernel_basis.json \
  --model B --b1 0.2 --b2 0.1 --b3 0.3 \
  --m0 1.0 --yPhi 0.1 --yphi 0.1 --mu 1.0 \
  --two-loop-rge --uv 2.41e14 \
  --output derived_wi_modelB.json
```

Both write: `derived_weights.w1_over_w2`, `derived_weights.w3_over_w2`, and (if RGE on) `alpha3_MZ_pred` and Λ₅.

## B) Route-A fit with τ̃ and Δr̃_S (O(3) or O(4) source)
If you already have `phaseA_routeA_two_loop_O4_wallonly.json` from earlier, run:
```
python gauge_normalizer_from_wall.py \
  --tau-summary phaseA_routeA_two_loop_O4_wallonly.json \
  --msfv 1.94 3.94 5.94 \
  --uv 2.41e14 \
  --alpha3-target 0.1181 \
  --two-loop --steps 1200 --lambda-report \
  --output phaseA_routeA_report_two_loop.json
```

If using O(3) action density instead, first get τ̃:
```
python wall_tension_from_shell.py \
  --profile background_profile.csv \
  --o3 o3_action_density.csv \
  --output phaseA_tau_summary.json
```
then feed that JSON to `gauge_normalizer_from_wall.py` as `--tau-summary`.

---

Moderate-thick walls are fully supported: all integrals are done over your measured window, and only **ratios** are used to form wᵢ, so absolute z-scaling cancels.
