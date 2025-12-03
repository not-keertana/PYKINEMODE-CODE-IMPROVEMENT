import numpy as np
from scipy.integrate import ode


def compute_matrices(N, Win, n0):
    """
    Compute matrices S0, q0, M0, Q0T to compute physically interpretable extents

    Args:
        N (array-like): The stoichiometric matrix.
        Win (array-like): The input stoichiometric matrix.
        n0 (array-like): The initial concentration data.

    Returns:
        tuple: A tuple containing the matrices q0T, S0T, M0T, Q0T.

    Raises:
        ValueError: If the rank of the [N.T Win] matrix is not equal to numReactions + numInlet.

    """
    if len(n0.shape) == 1:
        n0 = np.reshape(n0, (n0.shape[0], 1))

    numReactions = N.shape[0]
    numSpecies = N.shape[1]
    numInlet = Win.shape[1]

    mat = np.concatenate([N.T, Win], axis=1)

    if np.linalg.matrix_rank(mat) == numReactions + numInlet:
        U1, S1, V1T = np.linalg.svd(mat)
        Q = U1[:, numReactions + numInlet:]
        mat2 = np.concatenate([N.T, Q], axis=1)
        U2, S2, V2T = np.linalg.svd(mat2)
        L = U2[:, numSpecies - numInlet:]
        M = L @ np.linalg.pinv(Win.T @ L)
        ST = np.linalg.pinv(N.T) @ (np.identity(numSpecies) - (Win @ M.T))

        q0T = (np.ones((numSpecies - numReactions - numInlet, 1)).T @ Q.T) / (
            np.ones((numSpecies - numReactions - numInlet, 1)).T @ Q.T @ n0
        )
        S0T = ST @ (np.identity(numSpecies) - n0 @ q0T)
        M0T = M.T @ (np.identity(numSpecies) - n0 @ q0T)
        Q0T = Q.T @ (np.identity(numSpecies) - n0 @ q0T)

    else:
        raise ValueError(
            "Rank Error. Rank of [N.T Win] matrix != numReactions + numInlet."
        )

    return q0T, S0T, M0T, Q0T


def extents_derivative_v2(t, y, uin, uout):
    """
    Derivative function for xin, lamda, and mass with respect to time

    Args:
        t (float): The current time.
        y (array-like): The current state vector.
        uin (function): The input concentration function.
        uout (function): The output concentration function.

    Returns:
        array-like: The derivative of the state vector with respect to time.

    """
    m, lamda = y[0], y[1]
    xin = np.array(y[2:])
    dm_dt = np.sum(uin(t)) - uout(t)[0]
    dl_dt = -(uout(t)[0] / m) * lamda
    dxin_dt = uin(t) - (uout(t)[0] / m) * xin
    dxin_dt = list(dxin_dt)
    dydt = [dm_dt, dl_dt]
    dydt.extend(dxin_dt)
    return dydt


def compute_inlet_extents(uin, uout, time, n0, Mw):
    """
    Returns xin, lamda, m
    Computes xin, lamda and m given uin, uout, time, n0, and Mw data

    Uses the algorithm proposed in V2 (Incremental identification. Nirav et al. paper)

    Args:
        uin (function): The input concentration function.
        uout (function): The output concentration function.
        time (array-like): The time points for the simulation.
        n0 (array-like): The initial concentration data.
        Mw (array-like): The molecular weight data.

    Returns:
        tuple: A tuple containing the vectors xin, lamda, and m.

    """
    p = uin(0).shape[0]
    m0 = np.sum(n0 @ Mw)
    t0 = 0.0
    y0 = [m0, 1]
    y0.extend([0] * p)

    solver = ode(extents_derivative_v2)
    solver.set_integrator("dop853")
    solver.set_f_params(uin, uout)
    solver.set_initial_value(y0, t0)

    N_ = time.shape[0]
    t1 = time[-1]

    sol = np.empty((N_, p + 2))
    sol[0] = y0

    k = 1
    while solver.successful() and solver.t < t1:
        solver.integrate(time[k])
        sol[k] = solver.y
        k += 1

    m = sol[:, 0]
    lamda = sol[:, 1]
    lamda = np.reshape(lamda, (lamda.shape[0], 1))
    xin = sol[:, 2:]
    return xin, lamda, m