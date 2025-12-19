import numpy as np
from scipy.integrate import ode


def compute_matrices(N, Win, n0, reactor_mode="cstr"):

    if n0.ndim == 1:
        n0 = n0.reshape(-1, 1)

    numReactions = N.shape[0]
    numSpecies = N.shape[1]

    # ---------- BATCH ----------
    if reactor_mode == "batch":
        S0T = np.linalg.pinv(N.T)
        return (
            np.zeros((0, numSpecies)),  # q0T
            S0T,
            np.zeros((0, numSpecies)),  # M0T
            np.zeros((0, numSpecies)),  # Q0T
        )

    # ---------- SEMI-BATCH / CSTR ----------
    numInlet = Win.shape[1]

    mat = np.concatenate([N.T, Win], axis=1)
    if np.linalg.matrix_rank(mat) != numReactions + numInlet:
        raise ValueError("Rank error in [N.T Win]")

    U1, _, _ = np.linalg.svd(mat)
    Q = U1[:, numReactions + numInlet:]

    mat2 = np.concatenate([N.T, Q], axis=1)
    U2, _, _ = np.linalg.svd(mat2)
    L = U2[:, numSpecies - numInlet:]

    M = L @ np.linalg.pinv(Win.T @ L)
    ST = np.linalg.pinv(N.T) @ (np.eye(numSpecies) - Win @ M.T)

    # ---------- SEMI-BATCH ----------
    if reactor_mode == "semi-batch":
        return (
            np.zeros((0, numSpecies)),  # q0T
            ST,
            M.T,
            np.zeros((0, numSpecies)),  # Q0T
        )

    # ---------- CSTR ONLY ----------
    q0T = (
        np.ones((Q.shape[1], 1)).T @ Q.T
    ) / (
        np.ones((Q.shape[1], 1)).T @ Q.T @ n0
    )

    S0T = ST @ (np.eye(numSpecies) - n0 @ q0T)
    M0T = M.T @ (np.eye(numSpecies) - n0 @ q0T)
    Q0T = Q.T @ (np.eye(numSpecies) - n0 @ q0T)

    return q0T, S0T, M0T, Q0T





def extents_derivative_v2(t, y, uin, uout, reactor_mode):
    """
    State vectors:
    batch:      y = [m]
    semi-batch: y = [m, *xin]
    cstr:       y = [m, lamda, *xin]
    """

    if reactor_mode == "batch":
        return [0.0]

    elif reactor_mode == "semi-batch":
        m = y[0]
        xin = np.array(y[1:])

        u = uin(t)
        dm_dt = np.sum(u)
        dxin_dt = u

        return [dm_dt, *dxin_dt.tolist()]

    elif reactor_mode == "cstr":
        m = y[0]
        lamda = y[1]
        xin = np.array(y[2:])

        u = uin(t)
        vout = uout(t)[0]

        dm_dt = np.sum(u) - vout
        dl_dt = -(vout / m) * lamda
        dxin_dt = u - (vout / m) * xin

        return [dm_dt, dl_dt, *dxin_dt.tolist()]

    else:
        raise ValueError("Unknown reactor mode")




def compute_inlet_extents(uin, uout, time, n0, Mw, reactor_mode="cstr"):

    m0 = np.sum(n0 @ Mw)
    t0 = time[0]

    if reactor_mode == "batch":
        y0 = [m0]
        sol_dim = 1

    elif reactor_mode == "semi-batch":
        p = uin(0).shape[0]
        y0 = [m0] + [0.0] * p
        sol_dim = p + 1

    else:  # cstr
        p = uin(0).shape[0]
        y0 = [m0, 1.0] + [0.0] * p
        sol_dim = p + 2

    solver = ode(extents_derivative_v2)
    solver.set_integrator("dop853")
    solver.set_f_params(uin, uout, reactor_mode)
    solver.set_initial_value(y0, t0)

    sol = np.zeros((len(time), sol_dim))
    sol[0] = y0

    k = 1
    while solver.successful() and k < len(time):
        solver.integrate(time[k])
        sol[k] = solver.y
        k += 1

    m = sol[:, 0]

    if reactor_mode == "batch":
        return None, None, m

    elif reactor_mode == "semi-batch":
        xin = sol[:, 1:]
        return xin, None, m

    else:
        lamda = sol[:, 1].reshape(-1, 1)
        xin = sol[:, 2:]
        return xin, lamda, m


    