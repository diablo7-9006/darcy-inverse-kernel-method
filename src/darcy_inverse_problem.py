import os

# Keep CPU/JAX runs from over-allocating threads on smaller machines.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import contextlib
import io
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

jax.config.update("jax_enable_x64", True)

# Initial conditions

SEED = 42

N_grid = 60
f_const = 1.0

n_interior = 220
n_boundary = 80
n_data = 40
noise_std = 1e-3

n_center_1d = 9
ell = 0.18

lam_u = 1e-3
lam_a = 1e-3
n_gn_steps = 8
lm_damping = 1e-2
line_search_decay = 0.5

plot_grid = 120
N_TRIALS = 3

SAVE_FIGS = True
DISPLAY_FIGS = False
SAVED_FILES = []

def save_show(fig, filename):
    if SAVE_FIGS:
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        SAVED_FILES.append(filename)
    if DISPLAY_FIGS:
        plt.show()
    else:
        plt.close(fig)

# Synthetic inverse-Darcy true solution

def true_kappa(x, y):
    s = np.sin(2.0 * np.pi * x) + np.sin(2.0 * np.pi * y)
    return np.exp(s) + np.exp(-s)

def true_a(x, y):
    return np.log(true_kappa(x, y))

def solve_forward_darcy_variable_k(N):

    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    h = x[1] - x[0]

    X, Y = np.meshgrid(x, y, indexing="ij")
    K = true_kappa(X, Y)

    n_unknown = (N - 2) * (N - 2)
    A = lil_matrix((n_unknown, n_unknown))
    b = np.full(n_unknown, f_const)

    def idx(i, j):
        return (i - 1) * (N - 2) + (j - 1)

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            row = idx(i, j)

            # Face-centered permeability values.
            ke = 0.5 * (K[i, j] + K[i + 1, j])
            kw = 0.5 * (K[i, j] + K[i - 1, j])
            kn = 0.5 * (K[i, j] + K[i, j + 1])
            ks = 0.5 * (K[i, j] + K[i, j - 1])

            A[row, row] = (ke + kw + kn + ks) / h**2

            if i + 1 <= N - 2:
                A[row, idx(i + 1, j)] = -ke / h**2
            if i - 1 >= 1:
                A[row, idx(i - 1, j)] = -kw / h**2
            if j + 1 <= N - 2:
                A[row, idx(i, j + 1)] = -kn / h**2
            if j - 1 >= 1:
                A[row, idx(i, j - 1)] = -ks / h**2

    u_int = spsolve(csr_matrix(A), b)

    U = np.zeros((N, N))
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            U[i, j] = u_int[idx(i, j)]

    return x, y, U


def build_truth_cache():
    # Build once and reuse in baseline and robustness trials.
    xg, yg, U_true_grid = solve_forward_darcy_variable_k(N_grid)
    interp_u = RegularGridInterpolator((xg, yg), U_true_grid)

    xp = np.linspace(0.05, 0.95, plot_grid)
    yp = np.linspace(0.05, 0.95, plot_grid)
    XP, YP = np.meshgrid(xp, yp, indexing="ij")
    plot_pts = np.column_stack([XP.ravel(), YP.ravel()])

    return {
        "xg": xg,
        "yg": yg,
        "interp_u": interp_u,
        "XP": XP,
        "YP": YP,
        "plot_pts": plot_pts,
        "a_true": true_a(XP, YP),
        "u_true": interp_u(plot_pts).reshape(plot_grid, plot_grid),
        "avg_a": np.mean(true_a(xg[:, None], yg[None, :])),
    }

# Sampling and RBF basis

def sample_points(n_int, n_bnd, n_dat):
    interior = np.random.rand(n_int, 2)

    m = n_bnd // 4
    t1 = np.random.rand(m)
    t2 = np.random.rand(m)
    t3 = np.random.rand(m)
    t4 = np.random.rand(n_bnd - 3 * m)

    boundary = np.vstack([
        np.column_stack([t1, np.zeros_like(t1)]),
        np.column_stack([t2, np.ones_like(t2)]),
        np.column_stack([np.zeros_like(t3), t3]),
        np.column_stack([np.ones_like(t4), t4]),
    ])

    data_pts = np.random.rand(n_dat, 2)
    return interior, boundary, data_pts


def make_centers(n1d):
    gx = np.linspace(0.08, 0.92, n1d)
    gy = np.linspace(0.08, 0.92, n1d)
    CX, CY = np.meshgrid(gx, gy, indexing="ij")
    return np.column_stack([CX.ravel(), CY.ravel()])


def gaussian_basis_and_derivatives(points, centers, length_scale):
    # Analytic Gaussian RBF derivatives in JAX.
    px = points[:, 0][:, None]
    py = points[:, 1][:, None]
    cx = centers[:, 0][None, :]
    cy = centers[:, 1][None, :]

    dx = px - cx
    dy = py - cy
    r2 = dx**2 + dy**2
    ell2 = length_scale**2

    Phi = jnp.exp(-r2 / (2.0 * ell2))
    dPhi_dx = -(dx / ell2) * Phi
    dPhi_dy = -(dy / ell2) * Phi
    lapPhi = ((r2 - 2.0 * ell2) / (ell2**2)) * Phi
    return Phi, dPhi_dx, dPhi_dy, lapPhi


def eval_field(points, centers, length_scale, coeff):
    Phi, dPx, dPy, lapP = gaussian_basis_and_derivatives(points, centers, length_scale)
    return Phi @ coeff, dPx @ coeff, dPy @ coeff, lapP @ coeff


def eval_field_np(points, centers, length_scale, coeff):
    values = eval_field(
        jnp.asarray(points),
        jnp.asarray(centers),
        jnp.asarray(float(length_scale)),
        jnp.asarray(coeff),
    )
    return tuple(np.asarray(v) for v in values)

# JAX residual and Gauss-Newton solver

def residual_core(x, centers, length_scale, interior_pts, boundary_pts, data_pts,
                  data_obs, f_const_val, noise_std_val, lam_u_val, lam_a_val):
    M = centers.shape[0]
    alpha = x[:M]
    beta = x[M:]

    _, ux_i, uy_i, lap_u_i = eval_field(interior_pts, centers, length_scale, alpha)
    a_i, ax_i, ay_i, _ = eval_field(interior_pts, centers, length_scale, beta)

    pde_res = -jnp.exp(a_i) * (lap_u_i + ax_i * ux_i + ay_i * uy_i) - f_const_val
    u_b, _, _, _ = eval_field(boundary_pts, centers, length_scale, alpha)
    u_d, _, _, _ = eval_field(data_pts, centers, length_scale, alpha)

    data_res = (u_d - data_obs) / noise_std_val
    reg_u = jnp.sqrt(lam_u_val) * alpha
    reg_a = jnp.sqrt(lam_a_val) * beta

    return jnp.concatenate([pde_res, u_b, data_res, reg_u, reg_a])


residual_jit = jax.jit(residual_core)
jacobian_jit = jax.jit(jax.jacfwd(residual_core, argnums=0))


def gauss_newton_solve(centers, length_scale, x0, interior_pts, boundary_pts, data_pts,
                       data_obs, noise, lu, la, max_steps=8, damping=1e-2, verbose=True):
    x = x0.copy()
    loss_hist = []

    args = (
        jnp.asarray(centers),
        jnp.asarray(float(length_scale)),
        jnp.asarray(interior_pts),
        jnp.asarray(boundary_pts),
        jnp.asarray(data_pts),
        jnp.asarray(data_obs),
        jnp.asarray(float(f_const)),
        jnp.asarray(float(noise)),
        jnp.asarray(float(lu)),
        jnp.asarray(float(la)),
    )

    def residual_np(z):
        return np.asarray(residual_jit(jnp.asarray(z), *args))

    def loss_np(z):
        r = residual_np(z)
        return 0.5 * np.dot(r, r)

    for it in range(max_steps + 1):
        r = residual_np(x)
        loss = 0.5 * np.dot(r, r)
        loss_hist.append(loss)

        if verbose:
            print(f"Step {it:02d}: loss = {loss:.6e}")

        if it == max_steps:
            break

        # Gauss-Newton normal equations with Levenberg damping.
        J = np.asarray(jacobian_jit(jnp.asarray(x), *args))
        JTJ = J.T @ J
        g = J.T @ r
        dx = np.linalg.solve(JTJ + damping * np.eye(JTJ.shape[0]), -g)

        step_scale = 1.0
        x_new = x + step_scale * dx

        # Backtracking line search. The minimum step prevents infinite shrinking
        # in difficult parameter settings.
        while loss_np(x_new) >= loss and step_scale > 1e-8:
            step_scale *= line_search_decay
            x_new = x + step_scale * dx

        x = x_new

    return x, loss_hist


def initial_coefficients(centers, avg_a):
    M = centers.shape[0]
    alpha0 = np.zeros(M)
    beta0 = np.full(M, avg_a / M)
    return np.concatenate([alpha0, beta0])


# Plotting

def plot_collocation_points(interior_pts, boundary_pts, data_pts, filename):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.scatter(interior_pts[:, 0], interior_pts[:, 1], s=14, label="Interior nodes")
    ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], s=20, label="Boundary nodes")
    ax.scatter(data_pts[:, 0], data_pts[:, 1], s=20, label="Data nodes")
    ax.set_title("Collocation points")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")
    save_show(fig, filename)


def plot_field(XP, YP, field, title, filename):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.contourf(XP, YP, field, levels=30, cmap="coolwarm")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    save_show(fig, filename)


def plot_loss_history(loss_hist, filename):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.semilogy(np.arange(len(loss_hist)), loss_hist, "-o")
    ax.set_title("Loss function history")
    ax.set_xlabel("Gauss-Newton step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    save_show(fig, filename)


def plot_sweep(df, xkey, xlabel, filename, logx=True):
    grouped = df.groupby(xkey)
    xs = np.array(sorted(df[xkey].unique()))

    def stats(metric):
        mean = np.array([grouped.get_group(x)[metric].mean() for x in xs])
        std = np.array([grouped.get_group(x)[metric].std() for x in xs])
        return mean, np.nan_to_num(std, nan=0.0)

    ma, sa = stats("err_a_L2")
    mu, su = stats("err_u_L2")

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.errorbar(xs, ma, yerr=sa, marker="o", capsize=3, label=r"Rel. err. in $a$")
    ax.errorbar(xs, mu, yerr=su, marker="s", capsize=3, label=r"Rel. err. in $u$")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Relative $L^2$ error")
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    save_show(fig, filename)


# Baseline solution

def run_baseline(cache):
    np.random.seed(SEED)
    interior_pts, boundary_pts, data_pts = sample_points(n_interior, n_boundary, n_data)

    data_obs = cache["interp_u"](data_pts) + noise_std * np.random.randn(n_data)
    centers = make_centers(n_center_1d)
    x0 = initial_coefficients(centers, cache["avg_a"])

    x_opt, loss_hist = gauss_newton_solve(
        centers, ell, x0, interior_pts, boundary_pts, data_pts, data_obs,
        noise_std, lam_u, lam_a, max_steps=n_gn_steps, damping=lm_damping, verbose=True
    )

    M = centers.shape[0]
    alpha_opt = x_opt[:M]
    beta_opt = x_opt[M:]

    a_rec, _, _, _ = eval_field_np(cache["plot_pts"], centers, ell, beta_opt)
    u_rec, _, _, _ = eval_field_np(cache["plot_pts"], centers, ell, alpha_opt)
    a_rec = a_rec.reshape(plot_grid, plot_grid)
    u_rec = u_rec.reshape(plot_grid, plot_grid)

    plot_collocation_points(interior_pts, boundary_pts, data_pts, "darcy_01_collocation_points.png")
    plot_field(cache["XP"], cache["YP"], cache["a_true"], r"Truth $a(x)$", "darcy_02_truth_a.png")
    plot_field(cache["XP"], cache["YP"], cache["u_true"], r"Truth $u(x)$", "darcy_03_truth_u.png")
    plot_loss_history(loss_hist, "darcy_04_loss_history.png")
    plot_field(cache["XP"], cache["YP"], a_rec, r"Recovered $a(x)$", "darcy_05_recovered_a.png")
    plot_field(cache["XP"], cache["YP"], u_rec, r"Recovered $u(x)$", "darcy_06_recovered_u.png")

    err_a = np.linalg.norm(a_rec - cache["a_true"]) / np.linalg.norm(cache["a_true"])
    err_u = np.linalg.norm(u_rec - cache["u_true"]) / np.linalg.norm(cache["u_true"])
    print(f"Relative error in a: {err_a:.4e}")
    print(f"Relative error in u: {err_u:.4e}")


# Robustness experiments

def run_one_trial(cache, *, n_int=220, n_bnd=80, n_dat=40, noise=1e-3,
                  n_c=9, length_scale=0.18, lu=1e-3, la=1e-3,
                  max_steps=8, seed=0):
    # One stochastic realization for one parameter setting.
    np.random.seed(seed)
    interior_pts, boundary_pts, data_pts = sample_points(n_int, n_bnd, n_dat)

    data_obs = cache["interp_u"](data_pts) + noise * np.random.randn(n_dat)
    centers = make_centers(n_c)
    x0 = initial_coefficients(centers, cache["avg_a"])

    with contextlib.redirect_stdout(io.StringIO()):
        x_opt, loss_hist = gauss_newton_solve(
            centers, length_scale, x0, interior_pts, boundary_pts, data_pts, data_obs,
            noise, lu, la, max_steps=max_steps, damping=lm_damping, verbose=False
        )

    M = centers.shape[0]
    alpha_opt = x_opt[:M]
    beta_opt = x_opt[M:]

    a_rec, _, _, _ = eval_field_np(cache["plot_pts"], centers, length_scale, beta_opt)
    u_rec, _, _, _ = eval_field_np(cache["plot_pts"], centers, length_scale, alpha_opt)
    a_rec = a_rec.reshape(plot_grid, plot_grid)
    u_rec = u_rec.reshape(plot_grid, plot_grid)

    da = a_rec - cache["a_true"]
    du = u_rec - cache["u_true"]

    return {
        "err_a_L2": float(np.linalg.norm(da) / np.linalg.norm(cache["a_true"])),
        "err_u_L2": float(np.linalg.norm(du) / np.linalg.norm(cache["u_true"])),
        "err_a_Linf": float(np.max(np.abs(da)) / np.max(np.abs(cache["a_true"]))),
        "err_u_Linf": float(np.max(np.abs(du)) / np.max(np.abs(cache["u_true"]))),
        "final_loss": float(loss_hist[-1]),
    }


def sweep(label, settings, cache):
    # Repeat each setting over N_TRIALS independent random draws.
    rows = []
    print(f"\n=== {label} ===")

    for setting in settings:
        err_a_vals = []
        err_u_vals = []

        for trial in range(N_TRIALS):
            seed = 1000 * trial + 17
            result = run_one_trial(cache, **setting, seed=seed)
            result.update(setting)
            result["trial"] = trial
            rows.append(result)

            err_a_vals.append(result["err_a_L2"])
            err_u_vals.append(result["err_u_L2"])

        kstr = ", ".join(f"{k}={v}" for k, v in setting.items())
        print(f"  {kstr}: mean L2 err a={np.mean(err_a_vals):.3e}, u={np.mean(err_u_vals):.3e}")

    return pd.DataFrame(rows)


def summarize(df, key):
    grouped = df.groupby(key)
    return pd.DataFrame({
        "a_L2_mean": grouped["err_a_L2"].mean(),
        "a_L2_std": grouped["err_a_L2"].std(),
        "u_L2_mean": grouped["err_u_L2"].mean(),
        "u_L2_std": grouped["err_u_L2"].std(),
        "a_Linf_mean": grouped["err_a_Linf"].mean(),
        "u_Linf_mean": grouped["err_u_Linf"].mean(),
    })


def run_experiments(cache):
    sweep_obs = [{"n_dat": v} for v in [5, 10, 20, 40, 80, 160]]
    sweep_noise = [{"noise": v} for v in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]]
    sweep_coll = [{"n_int": v} for v in [50, 100, 220, 400, 600]]
    sweep_ell = [{"length_scale": v} for v in [0.08, 0.12, 0.18, 0.25, 0.35]]
    sweep_reg = [{"lu": v, "la": v} for v in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]]

    df_obs = sweep("E1: number of observation points", sweep_obs, cache)
    df_noise = sweep("E2: observation noise gamma", sweep_noise, cache)
    df_coll = sweep("E3: number of collocation points", sweep_coll, cache)
    df_ell = sweep("E4a: kernel length scale", sweep_ell, cache)
    df_reg = sweep("E4b: regularization lambda", sweep_reg, cache)

    plot_sweep(df_obs, "n_dat", r"Number of observations $I$", "darcy_exp_01_observation_count.png")
    plot_sweep(df_noise, "noise", r"Noise std $\gamma$", "darcy_exp_02_noise_level.png")
    plot_sweep(df_coll, "n_int", r"Collocation points $n_{\Omega}$", "darcy_exp_03_collocation_count.png")
    plot_sweep(df_ell, "length_scale", r"Kernel length scale $\ell$", "darcy_exp_04_kernel_length_scale.png", logx=False)
    plot_sweep(df_reg, "lu", r"Regularization $\lambda$", "darcy_exp_05_regularization.png")

    raw = pd.concat([
        df_obs.assign(experiment="observation_count"),
        df_noise.assign(experiment="noise_level"),
        df_coll.assign(experiment="collocation_count"),
        df_ell.assign(experiment="kernel_length_scale"),
        df_reg.assign(experiment="regularization"),
    ], ignore_index=True)

    raw.to_csv("darcy_experiment_raw_results.csv", index=False)
    SAVED_FILES.append("darcy_experiment_raw_results.csv")

    summaries = {
        "E1_n_dat": summarize(df_obs, "n_dat"),
        "E2_noise": summarize(df_noise, "noise"),
        "E3_n_int": summarize(df_coll, "n_int"),
        "E4a_length_scale": summarize(df_ell, "length_scale"),
        "E4b_lambda": summarize(df_reg, "lu"),
    }

    summary_csv = pd.concat(summaries, names=["experiment", "parameter"])
    summary_csv.to_csv("darcy_experiment_summary_tables.csv")
    SAVED_FILES.append("darcy_experiment_summary_tables.csv")

    with pd.option_context("display.float_format", "{:.4e}".format):
        for name, table in summaries.items():
            print(f"\n--- {name} ---")
            print(table)


# Main execution

def main():
    cache = build_truth_cache()
    run_baseline(cache)
    run_experiments(cache)

    if SAVED_FILES:
        print("\nSaved outputs:")
        for file in SAVED_FILES:
            print(f"  {file}")


if __name__ == "__main__":
    main()