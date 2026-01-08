# ml-residual-pendulum

# Pendulum-v1: Energy-Shaping + PD Baseline + Residual RL (Task Plan)

Here’s a concrete, doable plan for **Pendulum-v1 + (Energy-shaping + PD) baseline + Residual RL**, written as a task breakdown with **checks/tests** so you’ll know you’re on track.

Assumptions: **Colab**, **Python**, **Gymnasium**. For RL, the path of least resistance is **Stable-Baselines3 (SAC)** (works well for continuous actions).

---

## 0) Repo / notebook skeleton (½ day)

### Tasks
- Create a GitHub repo with:
  - `notebooks/pendulum_residual_rl.ipynb`
  - `src/` (controllers, wrappers, eval)
  - `README.md` with goals + roadmap
- In Colab:
  - install deps: `gymnasium[classic-control]`, `numpy`, `matplotlib`, `stable-baselines3`, `torch`

### Checks
- You can `import gymnasium as gym` and `env = gym.make("Pendulum-v1")`.
- `env.reset()` returns obs shape `(3,)` `(cosθ, sinθ, θ_dot)`.

---

## 1) Understand the environment & define metrics (½ day)

### Tasks
- Log what Pendulum-v1 provides:
  - Observation: `[cos θ, sin θ, θ_dot]`
  - Action: torque `u ∈ [-2, 2]`
  - Reward (actually “negative cost”): penalizes angle error, angular velocity, and torque.
- Define evaluation metrics you’ll track consistently:
  - mean episode return (higher is better)
  - mean angle error over time
  - time-to-stabilize (optional)
  - torque usage (optional)

### Checks
- Run 10 random episodes, plot `θ(t)` and return distribution.
- Confirm you can reconstruct angle: `θ = atan2(sinθ, cosθ)`.

---

## 2) Build the baseline controller (Energy shaping + PD) (1–2 days)

This is the core **“strict math module”**.

### 2.1 Implement helpers

#### Tasks
- Write utilities:
  - `angle = atan2(sin, cos)`
  - `wrap_to_pi(angle)` (keep in `[-π, π]`)
- Make a function: `baseline_action(obs) -> u`

#### Checks
- Given a batch of obs, your `baseline_action` returns shape `(1,)` or scalar torque.

### 2.2 Energy shaping (swing-up)

#### Tasks
- Implement an “energy error” controller:
  - compute current energy `E(θ, θ_dot)` (up to a constant is fine)
  - desired energy corresponds to upright
  - apply torque to increase/decrease energy depending on `E - E*`
- Keep it simple and robust:
  - clip torque to env bounds
  - optionally add a deadzone when close enough

#### Checks
- From the downward start, the baseline should reliably “pump” and reach upright sometimes.
- Plot `E(t)` — it should move toward the target energy.

### 2.3 PD stabilization near upright

#### Tasks
- Define “near upright” region (e.g., `|θ| < θ_switch`)
- In that region use PD:
  - `u = -Kp * θ - Kd * θ_dot`
- Add a smooth switch:
  - either hard switch on `θ_switch`
  - or blend (recommended): `u = (1-w)*u_energy + w*u_pd`, where `w` grows near upright

#### Checks
- Starting near upright, it should settle and hold.
- Plot `θ(t)` from an initial small perturbation: should damp quickly.

### 2.4 Tune parameters

#### Tasks
- Tune `{Kp, Kd, energy_gain, switch_threshold}` manually.
- Keep a small table in the notebook documenting each trial.

#### Checks
- Run 50 eval episodes:
  - baseline return should be meaningfully better than random
  - behavior should look stable (no endless spinning unless you want that)
- Portfolio tip: include one short GIF or plot showing baseline swing-up + stabilization.

---

## 3) Wrap the environment for residual RL (½–1 day)

Residual RL: final action is **baseline + correction**.

### Tasks
- Create a wrapper (or just a policy function) where:
  - `u_total = clip(u_base(obs) + α * u_residual(obs), -2, 2)`
- Decide what RL sees:
  - Option A: RL sees original obs only
  - Option B (often better): RL also sees `u_base` and/or `θ` explicitly
- Start with a small residual scale `α` (e.g., `0.1–0.3`)

### Checks
- With `u_residual = 0`, your wrapper behaves exactly like baseline.
- With random residual, it doesn’t explode because of clipping and `α`.

---

## 4) Train a pure RL benchmark (SAC) (½–1 day)

This gives you a baseline you can compare to later.

### Tasks
- Train SAC on raw env (no baseline controller).
- Save model + training curves.

### Checks
- Learning curve improves over time.
- Evaluate 50 episodes, record mean return.

---

## 5) Train Residual RL on top of baseline (1–2 days)

### Tasks
- Train SAC where the action output is residual torque, and the env receives `u_total`.
- Use a smaller network than the pure RL benchmark (that’s part of your thesis).
- Keep everything else as similar as possible (timesteps, seed, eval procedure).

### Checks
- Early training should already be decent because baseline “works”.
- Residual should improve return vs baseline-only.
- Compare:
  - baseline-only
  - pure RL SAC
  - baseline + residual SAC (small net)

---

## 6) Evaluation & ablations (1–2 days)

This is where you turn it into a strong portfolio piece.

### 6.1 Main comparison

#### Tasks
- For each method, run `N=100` eval episodes with fixed seeds.
- Report mean ± std return.

#### Checks
- Residual should be:
  - better than baseline (ideally)
  - and often more sample-efficient / smaller than pure RL

### 6.2 Ablations (simple but impressive)

Pick 2–3:
- Vary residual scale `α` (`0.1 / 0.3 / 1.0`)
- Vary residual network size (tiny vs normal)
- Try model mismatch: add observation noise, or change gravity if you wrap a custom pendulum

#### Checks
- Residual RL should degrade more gracefully than pure RL under mismatch (often true).

---

## 7) Documentation & portfolio packaging (½–1 day)

### Tasks
- README structure:
  - Goal and hypothesis
  - Environment details
  - Baseline controller derivation (energy + PD)
  - Residual RL formulation
  - Results table + plots + gif
- Keep code clean:
  - `controllers.py`
  - `train.py` (optional)
  - `eval.py`
  - `plots.py`

### Checks
- A new person can run:
  - `pip install -r requirements.txt`
  - `python eval.py --method residual`
  - and reproduce results.

---

## What you might be forgetting (common gotchas)

- Seeding & reproducibility (set seeds for env + torch + numpy).
- Evaluation protocol separate from training (no exploration noise).
- Angle convention: Pendulum uses `[cosθ, sinθ, θ_dot]`, so always compute `θ` via `atan2`.
- Clipping and scaling: residual action should be scaled so it can’t overpower baseline early.
- Logging: store time series for a few episodes (`θ`, `θ_dot`, `u_base`, `u_res`, `u_total`).

---

## “Are things going right?” quick checklist

You’re on track if:
- Baseline-only can swing up and stabilize in many episodes.
- Pure SAC can learn (given enough steps).
- Residual SAC starts with baseline performance and then improves.
- Residual SAC achieves similar/better results with a smaller network or fewer steps.

