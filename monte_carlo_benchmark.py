"""
Monte Carlo Integration & High-Dimensional Curse of Dimensionality Benchmark
============================================================================
1. Evaluates the 1D integral int_0^4 exp(x) dx using Crude Monte Carlo vs.
   Composite Simpson's Rule across matched function evaluation budgets N,
   repeated over 50 independent random trials.
2. Demonstrates the Curse of Dimensionality on [0, 1]^d (d = 1, 2, 4, 10) for
   f(x) = prod_{k=1}^d (pi/2) sin(pi x_k), comparing O(N^{-4/d}) tensor-product
   Simpson quadrature against dimension-independent O(N^{-1/2}) Monte Carlo.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def composite_simpson_1d(f, a: float, b: float, n_subintervals: int):
    """Composite Simpson's rule on [a, b] with even n_subintervals."""
    if n_subintervals % 2 != 0:
        n_subintervals += 1
    x = np.linspace(a, b, n_subintervals + 1)
    y = f(x)
    h = (b - a) / n_subintervals
    weights = np.ones(n_subintervals + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0
    return float((h / 3.0) * np.sum(weights * y)), n_subintervals + 1


def monte_carlo_1d(f, a: float, b: float, n_samples: int, rng: np.random.Generator) -> float:
    """Crude Monte Carlo estimator Q_N on [a, b]."""
    x = rng.uniform(a, b, size=n_samples)
    return float((b - a) * np.mean(f(x)))


def composite_simpson_nd_separable(f_1d, d: int, n_per_dim: int):
    """Tensor-product Composite Simpson's rule on [0, 1]^d for separable integrands."""
    if n_per_dim % 2 != 0:
        n_per_dim += 1
    val_1d, evals_1d = composite_simpson_1d(f_1d, 0.0, 1.0, n_per_dim)
    return float(val_1d ** d), int(evals_1d ** d)


def monte_carlo_nd(f_nd, d: int, n_samples: int, rng: np.random.Generator) -> float:
    """Crude Monte Carlo estimator on [0, 1]^d."""
    pts = rng.uniform(0.0, 1.0, size=(n_samples, d))
    return float(np.mean(f_nd(pts)))


def main():
    a, b = 0.0, 4.0
    exact_1d = np.exp(4.0) - 1.0
    mean_f = (np.exp(4.0) - 1.0) / 4.0
    var_f = (np.exp(8.0) - 1.0) / 8.0 - mean_f**2
    theo_rmse_coeff = (b - a) * np.sqrt(var_f)

    subintervals_list = [10, 40, 160, 640, 2560, 10240]
    n_trials = 50
    rng = np.random.default_rng(42)

    eval_counts_1d, simp_errs_1d, mc_rmses_1d, mc_theo_1d, mc_trial_errs = [], [], [], [], []
    print(f"=== Part 1: 1D Integral int_0^4 exp(x) dx (Exact = {exact_1d:.6f}) ===")
    print(f"{'Evals N':<10} | {'Simpson Err':<13} | {'MC Emp RMSE':<13} | {'MC Theo RMSE':<13}")
    print("-" * 58)

    for m_sub in subintervals_list:
        s_val, n_eval = composite_simpson_1d(np.exp, a, b, m_sub)
        s_err = abs(s_val - exact_1d)
        trials = np.array([monte_carlo_1d(np.exp, a, b, n_eval, rng) for _ in range(n_trials)])
        emp_rmse = float(np.sqrt(np.mean((trials - exact_1d) ** 2)))
        theo_rmse = float(theo_rmse_coeff / np.sqrt(n_eval))

        eval_counts_1d.append(n_eval)
        simp_errs_1d.append(s_err)
        mc_rmses_1d.append(emp_rmse)
        mc_theo_1d.append(theo_rmse)
        mc_trial_errs.append(np.abs(trials - exact_1d))
        print(f"{n_eval:<10d} | {s_err:<13.3e} | {emp_rmse:<13.3e} | {theo_rmse:<13.3e}")

    f_1d_sin = lambda x: 0.5 * np.pi * np.sin(np.pi * x)
    f_nd_sin = lambda pts: np.prod(0.5 * np.pi * np.sin(np.pi * pts), axis=-1)
    hd_configs = {
        1: [4, 16, 64, 256, 1024],
        2: [2, 6, 16, 40, 100],
        4: [2, 4, 8, 14, 22],
        10: [2, 4],
    }

    hd_results = {}
    print("\n=== Part 2: High-Dimensional Comparison on [0, 1]^d ===")
    for d, n_grid in hd_configs.items():
        ev_l, s_l, mc_l = [], [], []
        for npd in n_grid:
            s_est, tot_ev = composite_simpson_nd_separable(f_1d_sin, d, npd)
            s_err = abs(s_est - 1.0)
            if tot_ev <= 200_000:
                mc_ests = np.array([monte_carlo_nd(f_nd_sin, d, tot_ev, rng) for _ in range(30)])
                mc_rmse = float(np.sqrt(np.mean((mc_ests - 1.0) ** 2)))
            else:
                mc_rmse = float(np.sqrt((np.pi**2 / 8.0) ** d - 1.0) / np.sqrt(tot_ev))
            ev_l.append(tot_ev)
            s_l.append(s_err)
            mc_l.append(mc_rmse)
            print(f"d={d:<2d} | n_per_dim={npd:<4d} | N={tot_ev:<9d} | Simpson Err={s_err:.3e} | MC RMSE={mc_rmse:.3e}")
        hd_results[d] = (ev_l, s_l, mc_l)

    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=150)

    ev_arr = np.array(eval_counts_1d)
    for t_idx in range(10):
        t_curve = [mc_trial_errs[k][t_idx] for k in range(len(ev_arr))]
        axes[0].loglog(ev_arr, t_curve, color="#3498db", alpha=0.22, linewidth=1.0)
    axes[0].loglog(ev_arr, mc_rmses_1d, "o-", color="#1f618d", linewidth=2.2, label="MC Empirical RMSE (50 trials)")
    axes[0].loglog(ev_arr, mc_theo_1d, "--", color="#e67e22", linewidth=2.0, label=r"MC Theory $(b-a)\sigma_f N^{-1/2}$")
    axes[0].loglog(ev_arr, simp_errs_1d, "s-", color="#27ae60", linewidth=2.2, label=r"Composite Simpson $O(N^{-4})$")
    axes[0].set_title(r"1D Integral $\int_0^4 e^x\,dx$: Convergence Rate Comparison", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Function Evaluations $N$ (log scale)")
    axes[0].set_ylabel("Absolute Error / RMSE (log scale)")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8.5, loc="lower left")

    colors_d = {1: "#27ae60", 2: "#2980b9", 4: "#8e44ad", 10: "#c0392b"}
    for d in [1, 2, 4, 10]:
        ev_l, s_l, mc_l = hd_results[d]
        axes[1].loglog(ev_l, s_l, "s-", color=colors_d[d], linewidth=1.9, label=f"Simpson ($d={d}$, $O(N^{{-4/{d}}})$)")
        if d == 10:
            axes[1].loglog(ev_l, mc_l, "o--", color="#111111", linewidth=2.3, label=r"Monte Carlo ($d=10$, $O(N^{-1/2})$)")
    axes[1].set_title(r"Curse of Dimensionality on $[0, 1]^d$ ($d=1,2,4,10$)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Total Function Evaluations $N$ (log scale)")
    axes[1].set_ylabel("Relative Error / RMSE (log scale)")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=8.0, loc="lower left")

    plt.tight_layout()
    plt.savefig("results/mc_vs_simpson_convergence.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
