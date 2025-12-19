import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import plotly.express as px
from functools import partial
from sklearn.metrics import mean_squared_error
from ratelawgen import RateLaw  # custom package
from datatools import DataTools
from preprocess import num_params
import warnings
from joblib import Parallel, delayed
from extents import compute_matrices, compute_inlet_extents
import heapq
import itertools
from simulate import Simulate

# from scipy.integrate import ode


class Incremental(Simulate, DataTools):
    def __init__(
        self, N, Mw, V, Winhat=None, uin=None, uout=None, n0=None, config=None
    ):
        """
        Initializes an Incremental object.

        Args:
            N: Stoichiometric matrix (RxS) where S is the number of species and
                R is the number of reactions.
            Mw: Molecular weight matrix (SxS) where S is the number of species.
            V: Volume data for the reactor.
            Winhat: Inlet flow rate matrix (SxP) where P is the number of inlets.
                Defaults to None.
            uin: Inlet concentration data. Defaults to None.
            uout: Outlet concentration data. Defaults to None.
            n0: Initial concentrations of species. Must be provided and cannot be None.
            config: Additional configuration data for the reactor. Defaults to None.

        Raises:
            ValueError: If n0 is None or if the dimensions of the provided data are inconsistent.

        """
        if n0 is None:
            raise ValueError("n0 cannot be None")

        self.add_stoichiometry_data(N)

        self.add_molweight_data(Mw)
        if self.Mw.shape[0] != self.S:
            raise ValueError(
                f"Dimension of Mw {self.Mw.shape} should be {self.S}x{self.S} \
                    and is not consistent with Stoichiometric Matrix N {self.N.shape}"
            )

        V, Winhat, uin, uout = self.add_reactor_config(V, Winhat, uin, uout, config)
        self.add_volume_data(V, kind="linear")

        self.add_Winhat_data(Winhat)
        if self.Winhat.shape[0] != self.S:
            raise ValueError(
                f"Dimension of Winhat {self.Winhat.shape} should be {self.S}x{self.P} \
                    and is not consistent with Stoichiometric Matrix N {self.N.shape}"
            )

        self.add_uin_data(uin, kind="linear")
        if self.uin(0).shape[0] != self.P:
            raise ValueError(
                f"Dimension of uin {self.uin.shape} is not consistent with \
                    Winhat {self.Win.shape}"
            )

        self.add_uout_data(uout, kind="linear")
        self.add_n0_data(n0)
        if self.n0.shape[0] != self.S:
            raise ValueError(
                f"Required {self.S} species for n0. Received {self.n0.shape[0]} species instead"
            )

        self.Win = np.linalg.pinv(self.Mw) @ self.Winhat  # (SxP) matrix
        if len(self.Win.shape) == 1:
            self.Win = np.reshape(self.Win, (self.Win.shape[0], 1))
        self.m0 = np.sum(self.Mw @ self.n0)

    def reaction_flow_f1(self):
        """
        Computes the reaction flow rate based on the concentration data.

        Returns:
            The computed reaction flow rate.

        Raises:
            ValueError: If concentration data is not specified. Add concentration data
                through the `add_concentration_data` method.
        """
        if not hasattr(self, "n"):
            raise ValueError(
                "Concentration data not specified. Add concentration data through\
                      add_concentration_data method"
            )

        mass = np.sum(self.n(self.time) @ self.Mw, axis=1)
        flow_rate = (
            self.dndt(self.time)
            - (self.uin(self.time) @ self.Win.T)
            + ((self.n(self.time) * ((self.uout(self.time).T / mass).T)))
        )
        return flow_rate

    def reaction_rate_r1(self):
        """
        Computes the reaction rate based on the reaction flow rate and reactor volume.

        Returns:
            The computed reaction rate.
        """
        flow_rate = self.reaction_flow_f1()
        reaction_rate = np.linalg.pinv(self.N.T) @ flow_rate.T * 1 / self.V(self.time)
        reaction_rate = reaction_rate.T
        return reaction_rate

    def rb_objective_function(self, K, reaction_rate, ratelaw, conc_dataT):
        """
        Computes the objective function for parameter estimation.

        Args:
            K: Parameter values for the rate law.
            reaction_rate: Computed reaction rates.
            ratelaw: Rate law function.
            conc_dataT: Transposed concentration data.

        Returns:
            The value of the objective function.

        Notes:
            This function is used in parameter estimation to determine the goodness of
                fit between the computed reaction rates and the rate law.

        """
        return np.sqrt(mean_squared_error(reaction_rate, ratelaw(conc_dataT, K)))

    def rb_est_params(self, reaction_rate, candidates: list):
        """
        Estimates Parameters given candidate rate laws and computed rates.

        Args:
            reaction_rate: Computed reaction rates.
            candidates: A list of candidate rate laws or RateLaw objects.

        Returns:
            Results from the optimization process for each candidate rate law.

        """
        # check for convergence. If it does not converge, reinitialize with a
        # different guessn(TO DO)

        results = []
        for idx, candidate in enumerate(candidates):
            if isinstance(candidate, RateLaw):
                print(f"\tProcessing expression {candidate.expression}")
                ratelaw = candidate.function
            else:
                print(f"\tProcessing {candidate.__str__()[10:-19]} Law....")
                ratelaw = candidate

            conc = self.c(self.time).T
            n_param = num_params(ratelaw, conc)
            x0 = np.random.rand(n_param)
            bnds = []
            for i in range(n_param):
                bnds.append((0, None))
            optim = sc.optimize.minimize(
                self.rb_objective_function,
                x0,
                args=(reaction_rate, ratelaw, conc),
                bounds=bnds,
            )
            results.append(optim)

        return results

    def rb_ci_parallelise(self, reaction_rate, conc, ratelaw, n_param):
        """
        Runs the parameter estimation process in parallel for bootstrapping.

        Args:
            reaction_rate: Computed reaction rates.
            conc: Concentration data.
            ratelaw: Rate law function.
            n_param: Number of parameters in the rate law.

        Returns:
            The optimized parameter values.

        """
        idx = np.random.choice(self.time.shape[0], self.time.shape[0], replace=True)
        bs_mdata = conc[idx, :]
        bs_compRRate = reaction_rate[idx]
        x0 = np.random.rand(n_param)
        bnds = []
        for i in range(n_param):
            bnds.append((0, None))
        optim = sc.optimize.minimize(
            self.rb_objective_function,
            x0,
            args=(bs_compRRate, ratelaw, bs_mdata.T),
            bounds=bnds,
        ).x
        return optim

    def conf_int_rate_based(
        self, reaction_rate, candidate, confidence, bootstraps, n_jobs
    ):
        """
        Computes the confidence interval for the rate-based parameter estimation.

        Args:
            reaction_rate: Computed reaction rates.
            candidate: Candidate rate law or RateLaw object.
            confidence: Confidence level for the interval.
            bootstraps: Number of bootstrap iterations.
            n_jobs: Number of parallel jobs.

        Returns:
            The confidence interval for the parameter estimation.

        """
        if isinstance(candidate, RateLaw):
            ratelaw = candidate.function
        else:
            ratelaw = candidate

        conc = self.c(self.time)
        n_param = num_params(ratelaw, conc.T)

        optims = []
        optims = Parallel(n_jobs=n_jobs)(
            delayed(self.rb_ci_parallelise)(reaction_rate, conc, ratelaw, n_param)
            for i in range(bootstraps)
        )

        CI = np.percentile(
            optims, [100 * (1 - confidence) / 2, 100 * (1 - (1 - confidence) / 2)]
        )
        return CI

    def eb_extent_rate_derivative(self, x, t, K, ratelaw):
        """
        Computes the derivative of computed extents.

        Args:
            x: Extent value.
            t: Time.
            K: Parameter values for the rate law.
            ratelaw: Rate law function.

        Returns:
            The derivative of the computed extents.

        Notes:
            This function is used to numerically find the extent value for a
                particular rate law by solving the derivative equation.

        """
        mass = np.sum(self.n(t) @ self.Mw)
        dxdt = ratelaw(self.c(t), K) * self.V(t) - (self.uout(t)[0] / mass) * x
        return dxdt

    def eb_objective_function(self, K, reaction_extent, ratelaw, time):
        """
        Computes the objective function for parameter estimation based on extent-based modeling.

        Args:
            K: Parameter values for the rate law.
            reaction_extent: Experimental reaction extents.
            ratelaw: Rate law function.
            time: Time values.

        Returns:
            The value of the objective function.

        Notes:
            This function compares the experimental reaction extents with the
            computed extents based on the rate law and parameters.
            It uses the root mean squared error (RMSE) as the measure of
            dissimilarity between the experimental and computed extents.

        """
        computed_extent = odeint(
            self.eb_extent_rate_derivative, 0, time, args=(K, ratelaw)
        )
        computed_extent = np.squeeze(computed_extent)
        loss = np.sqrt(mean_squared_error(reaction_extent, computed_extent))
        return loss

    def eb_est_params(self, reaction_extent, candidates: list, solver):
        """
        Estimates the parameters using extent-based modeling and multiple candidate rate laws.

        Args:
            reaction_extent: Experimental reaction extents.
            candidates: A list of candidate rate laws or rate law functions.
            solver: Optimization solver method.

        Returns:
            A list of optimization results for each candidate rate law.

        Notes:
            This function performs parameter estimation by minimizing the objective
            function based on extent-based modeling. It iterates over the candidates and
            finds the optimal parameter values using the specified optimization solver.

        """
        results = []
        for idx, candidate in enumerate(candidates):
            if isinstance(candidate, RateLaw):
                print(f"\tProcessing expression {candidate.expression}")
                ratelaw = candidate.function
            else:
                print(f"\tProcessing {candidate.__str__()[10:-19]} Law....")
                ratelaw = candidate

            conc = self.c(self.time).T
            n_param = num_params(ratelaw, conc)
            x0 = np.random.rand(n_param)
            bnds = []
            for i in range(n_param):
                bnds.append((0, None))
            optim = sc.optimize.minimize(
                self.eb_objective_function,
                x0,
                args=(reaction_extent, ratelaw, self.time),
                method=solver,
                bounds=bnds,
            )
            results.append(optim)

        return results

    def eb_ci_parallelise(self, reaction_extent, ratelaw, n_param, solver):
        """
        Runs parameter estimation in parallel for bootstrapping in extent-based modeling.

        Args:
            reaction_extent: Experimental reaction extents.
            ratelaw: Rate law function.
            n_param: Number of parameters in the rate law.
            solver: Optimization solver method.

        Returns:
            The optimized parameter values.

        Notes:
            This function is used for parallel computation in bootstrapping for
            extent-based modeling. It performs parameter estimation using a
            subset of the reaction extent data and returns the optimized parameter values.

        """
        idx = np.random.choice(self.time.shape[0], self.time.shape[0], replace=True)
        idx.sort()
        if idx[0] == 0:
            pass
        else:
            idx = idx[:-1]
            idx = np.insert(idx, 0, 0)

        bs_time = self.time[idx]
        bs_extents = reaction_extent[idx]
        x0 = np.random.rand(n_param)
        bnds = []
        for i in range(n_param):
            bnds.append((0, None))
        optim = sc.optimize.minimize(
            self.eb_objective_function,
            x0,
            args=(bs_extents, ratelaw, bs_time),
            method=solver,
            bounds=bnds,
        ).x
        return optim

    def mole_balance_ft(self, y, t, ratelaws):
        """
        Mole balance equation for the simultaneous approach.

        Args:
            y: Concentration vector.
            t: Time.
            ratelaws: List of rate law functions.

        Returns:
            The derivative of the concentration vector.

        Notes:
            This function represents the mole balance equation for the simultaneous
            approach in reaction kinetics. It calculates the derivative of the
            concentration vector based on the rate laws, concentrations, and flow rates.

        """
        c = y / self.V(t)
        rate = np.array([ratelaw(c) for ratelaw in ratelaws])
        m = np.sum(self.Mw @ y)  # current mass
        dydt = (
            (self._N.T @ rate * self.V(t))
            + (self.Win @ self.uin(t))
            - (self.uout(t) * y / m)
        )
        return dydt

    def ft_objective_function(self, K, C, cand_list, num_params):
        """
        Computes the objective function for fine-tuning using the simultaneous approach.

        Args:
            K: Parameter values for the rate laws.
            C: Experimental concentration data.
            cand_list: List of candidate rate laws or rate law functions.
            num_params: Number of parameters for each rate law.

        Returns:
            The value of the objective function.

        Notes:
            This function compares the simulated concentration data with the experimental data.
            It calculates the norm of the difference between the two sets of data
            as the objective function value.

        """
        K_new = []
        temp = 0
        for n_param in num_params:
            K_new.append(K[temp: temp + n_param])
            temp += n_param

        ratelaws = [
            partial(ratelaw.function, K=K_new[idx])
            if isinstance(ratelaw, RateLaw)
            else partial(ratelaw, K=K_new[idx])
            for idx, ratelaw in enumerate(cand_list)
        ]
        sol = odeint(self.mole_balance_ft, self.n0, self.time, args=(ratelaws,))
        return np.linalg.norm(sol - C)

    def ft_est_params(self, cand_list, guess, num_params, solver):
        """
        Estimates the parameters for fine-tuning the model using the simultaneous approach.

        Args:
            cand_list: List of candidate rate laws or rate law functions.
            guess: Initial guess for the parameter values.
            num_params: Number of parameters for each rate law.
            solver: Optimization solver method.

        Returns:
            The optimization result containing the estimated parameter values.

        Notes:
            This function performs parameter estimation by minimizing the objective
            function based on the simultaneous approach. It iterates over the
            candidates and finds the optimal parameter values using the
            specified optimization solver.

        """
        conc = self.c(self.time)
        x0 = guess

        bnds = []
        for i in range(len(x0)):
            bnds.append((0, None))

        if solver is None:
            optim = sc.optimize.minimize(
                self.ft_objective_function,
                x0,
                args=(conc, cand_list, num_params),
                bounds=bnds,
            )
        else:
            optim = sc.optimize.minimize(
                self.ft_objective_function,
                x0,
                args=(conc, cand_list, num_params),
                method=solver,
                bounds=bnds,
            )
        return optim

    def conf_int_extent_based(
        self, reaction_extent, candidate, confidence, bootstraps, n_jobs, solver
    ):
        """
        Computes the confidence interval for parameter estimation based on extent-based modeling.

        Args:
            reaction_extent: Experimental reaction extents.
            candidate: Candidate rate law or rate law function.
            confidence: Confidence level for the interval.
            bootstraps: Number of bootstraps for estimating the interval.
            n_jobs: Number of parallel jobs for bootstrapping.
            solver: Optimization solver method.

        Returns:
            The confidence interval for the estimated parameter values.

        Notes:
            This function performs bootstrapping to estimate the confidence interval for
            the parameter values based on extent-based modeling.
            It uses the specified rate law or rate law function to compute the
            parameter estimates for each bootstrap sample.

        """
        if isinstance(candidate, RateLaw):
            ratelaw = candidate.function
        else:
            ratelaw = candidate

        conc = self.c(self.time)
        n_param = num_params(ratelaw, conc.T)

        optims = []
        optims = Parallel(n_jobs=n_jobs)(
            delayed(self.eb_ci_parallelise)(reaction_extent, ratelaw, n_param, solver)
            for i in range(bootstraps)
        )

        CI = np.percentile(
            optims, [100 * (1 - confidence) / 2, 100 * (1 - (1 - confidence) / 2)]
        )
        return CI

    def interpret_results(
        self, results: list[list], candidates_list, metric, plot=True, top_k=1
    ):
        """
        Interprets the results of parameter estimation by analyzing the metrics and
        selecting the best rate laws.

        Args:
            results: List of optimization results for each set of candidates.
            candidates_list: List of candidate rate laws or rate law functions.
            metric: Metric to evaluate the performance of rate laws (e.g., "rmse", "aic", "aicc").
            plot: Boolean flag indicating whether to plot the metric values.
            top_k: Number of top rate laws to select as the best (default: 1).

        Returns:
            List of indices representing the best rate laws for each reaction.

        Notes:
            This function analyzes the optimization results and evaluates
            the performance of rate laws based on the specified metric.
            It can plot the metric values for visualization.
            The top-k rate laws with the lowest metric values are selected as
            the best rate laws for each reaction.

        """
        # Edited ones
        # Best Rate law
        best = []

        for i in range(len(candidates_list)):
            result = results[i]
            r_strings = [
                candidate.expression
                if isinstance(candidate, RateLaw)
                else candidate.__str__()[10:-19]
                for candidate in candidates_list[i]
            ]  # creates ratelaw name strings

            if metric == "rmse":
                loss = [res.fun for res in result]

            elif metric == "aic":
                Ks = [
                    num_params(candidate.function, self.c(self.time).T)
                    if isinstance(candidate, RateLaw)
                    else num_params(candidate, self.c(self.time).T)
                    for candidate in candidates_list[i]
                ]
                n = self.time.shape[0]
                loss = [
                    ((2 * Ks[idx]) + (n * np.log(res.fun**2)))
                    for idx, res in enumerate(result)
                ]

            elif metric == "aicc":
                Ks = [
                    num_params(candidate.function, self.c(self.time).T)
                    if isinstance(candidate, RateLaw)
                    else num_params(candidate, self.c(self.time).T)
                    for candidate in candidates_list[i]
                ]
                n = self.time.shape[0]
                loss = [
                    (2 * Ks[idx])
                    + (
                        n * np.log(res.fun**2)
                        + (2 * Ks[idx] * (Ks[idx] + 1) / (n - Ks[idx] - 1))
                    )
                    for idx, res in enumerate(result)
                ]

            else:
                raise ValueError(
                    "Metric not supported. Available metrics are rmse, aic and aicc"
                )

            if plot is True:
                # if plotly not there use matplotlib (TO DO)
                fig = px.bar(
                    x=r_strings,
                    y=loss,
                    labels={"x": "Rate Law", "y": metric.upper()},
                    title=f"{metric.upper()} vs Rate Law for Reaction {i+1}",
                )
                fig.update_layout(xaxis={"categoryorder": "total descending"})
                fig.show()

            # pick top_k elements if candidates_list[i] has >= top_k ratelaws.
            # else: pick len(candidates_list[i]) ratelaws instead of top_k
            if top_k > len(candidates_list[i]):
                k = len(result)
            else:
                k = top_k

            best_idx = heapq.nsmallest(k, range(len(loss)), loss.__getitem__)
            best.append(best_idx)

        return best

    def estimate_parameters(
        self,
        candidates_list,
        method="rate_based",
        metric="rmse",
        conf_int=False,
        n_jobs=-1,
        confidence=0.95,
        bootstraps=1000,
        plot=True,
        solver="Nelder-Mead",
    ):
        """
        Estimates the parameters of the reaction kinetics based on the specified method.

        Args:
            candidates_list: List of candidate rate laws or rate law functions for each reaction.
            method: Method for parameter estimation ("rate_based" or "extent_based",
                default: "rate_based").
            metric: Metric to evaluate the performance of rate laws (default: "rmse").
            conf_int: Boolean flag indicating whether to compute confidence intervals
                (default: False).
            n_jobs: Number of parallel jobs to run in parallel (default: -1, using all
                available processors).
            confidence: Confidence level for computing confidence intervals (default: 0.95).
            bootstraps: Number of bootstrap iterations for confidence interval computation
                (default: 1000).
            plot: Boolean flag indicating whether to plot the metric values (default: True).
            solver: Solver method for optimization (default: "Nelder-Mead").

        Returns:
            Dictionary containing the best rate laws, estimated parameters, and
                optionally confidence intervals.

        Notes:
            This function estimates the parameters of the reaction kinetics using
            either the rate-based or extent-based approach. It evaluates the performance
            of different rate laws using the specified metric and selects the best rate laws.
            If conf_int is True, it computes confidence intervals for the estimated
            parameters using bootstrapping. The function returns a dictionary containing
            the best rate laws, estimated parameters, and (optionally) confidence intervals.
        """
        best_result = {}
        best_result = {}
        best_result["best_ratelaws"] = []
        best_result["params"] = []
        best_result["conf_ints"] = []
        if not hasattr(self, "n"):
            raise ValueError(
                "Concentration data not specified. Add concentration \
                    data through add_concentration_data method"
            )

        if len(candidates_list) != self.R:
            warnings.warn(
                f"candidates_list provided is not a list with a first dimension \
                    size of {self.R}. Please ensure that the list you are \
                        passing has R elements in its first dimension. \
                            Estimating only {len(candidates_list)} reaction kinetics.",
                stacklevel=2,
            )

        if method == "rate_based":
            results = []
            if self.methodology == 1 or self.methodology == "rate":
                reaction_rates = self.reaction_rate_r1()
                for i in range(len(candidates_list)):
                    print(f"Processing Reaction {i+1}:")
                    candidates = candidates_list[i]
                    results.append(self.rb_est_params(reaction_rates[:, i], candidates))

                best_idxs = self.interpret_results(
                    results, candidates_list, metric, plot, top_k=1
                )

                for i in range(len(candidates_list)):
                    r_strings = [
                        candidate.expression
                        if isinstance(candidate, RateLaw)
                        else candidate.__str__()[10:-19]
                        for candidate in candidates_list[i]
                    ]  # creates ratelaw name strings
                    print(f"Reaction {i+1}: ")
                    print(f"Best Rate Law: {r_strings[best_idxs[i][0]]}")
                    print(f"Estimated Parameters: {results[i][best_idxs[i][0]].x}")
                    best_candidate = candidates_list[i][best_idxs[i][0]]
                    best_result["best_ratelaws"].append(best_candidate)
                    best_result["params"].append(results[i][best_idxs[i][0]].x)
                    if conf_int:
                        ci = self.conf_int_rate_based(
                            reaction_rates[:, i],
                            best_candidate,
                            confidence,
                            bootstraps,
                            n_jobs,
                        )
                        print(f"Confidence Interval {ci}")
                        best_result["conf_ints"].append(ci)
                    print()
                best_result["results"] = results

        elif method == "extent_based":
            results = []
            if self.methodology == 1:
                q0T, S0T, M0T, Q0T = compute_matrices(self.N, self.Win, self.n0)
                n = self.n(self.time)
                self._xr = n @ S0T.T
                self._xin = n @ M0T.T
                self._lamda = n @ q0T.T
                reaction_extents = self._xr

            elif self.methodology == 2:
                self._xin, self._lamda, _ = compute_inlet_extents(
                    self.uin, self.uout, self.time, self.n0, self.Mw
                )
                self._xr = np.linalg.pinv(self.Na.T) @ (
                    self.n(self.time).T
                    - (self.Wina @ self._xin.T)
                    - ((self._lamda * self.n0a).T)
                )
                self._xr = self._xr.T
                reaction_extents = self._xr
                n = (
                    (self._xr @ self.N)
                    + (self._xin @ self.Win.T)
                    + (self._lamda * self.n0)
                )
                c = (n.T / self.V(self.time)).T
                self.add_concentration_data(c, self.time)

            for i in range(len(candidates_list)):
                print(f"Processing Reaction {i+1}:")
                candidates = candidates_list[i]
                results.append(
                    self.eb_est_params(reaction_extents[:, i], candidates, solver)
                )

            best_idxs = self.interpret_results(
                results, candidates_list, metric, plot, top_k=1
            )

            for i in range(len(candidates_list)):
                r_strings = [
                    candidate.expression
                    if isinstance(candidate, RateLaw)
                    else candidate.__str__()[10:-19]
                    for candidate in candidates_list[i]
                ]  # creates ratelaw name strings
                print(f"Reaction {i+1}: ")
                print(f"Best Rate Law: {r_strings[best_idxs[i][0]]}")
                print(f"Estimated Parameters: {results[i][best_idxs[i][0]].x}")
                best_candidate = candidates_list[i][best_idxs[i][0]]
                best_result["best_ratelaws"].append(best_candidate)
                best_result["params"].append(results[i][best_idxs[i][0]].x)
                if conf_int:
                    ci = self.conf_int_extent_based(
                        reaction_extents[:, i],
                        best_candidate,
                        confidence,
                        bootstraps,
                        n_jobs,
                        solver,
                    )
                    print(f"Confidence Interval {ci}")
                    best_result["conf_ints"].append(ci)
                print()
            best_result["results"] = results

        else:
            raise ValueError(
                f"Valid methods are rate-based and extent-based. \
                    Unsupported method {method} provided"
            )

        self.results = results
        self.candidates_list = candidates_list
        self.metric = metric
        self.best_result = best_result
        return best_result

    def finetune(self, top_k=2, metric=None, solver="Nelder-Mead"):
        """
        Fine-tunes the model by selecting the best rate laws from the estimated parameters.

        Args:
            top_k: Number of top rate laws to consider for fine-tuning (default: 2).
            metric: Metric to evaluate the performance of rate laws (default: None,
            uses the metric from previous estimation).
            solver: Solver method for optimization (default: "Nelder-Mead").

        Returns:
            Dictionary containing the best rate laws and their estimated parameters.

        Notes:
            This method selects the best rate laws from the estimated parameters obtained
            using the `estimate_parameters` method. It considers the top_k rate laws
            and evaluates their performance using the specified metric.
            The method returns a dictionary containing the best rate laws and
            their estimated parameters.
        """
        best_result = {}
        best_result["best_ratelaws"] = []
        best_result["params"] = []
        candidates_list = self.candidates_list
        self._N = self.N[: len(candidates_list), :]  # for fine tuning just in case
        tuned_results = []
        if metric is None:
            metric = self.metric

        if not hasattr(self, "results"):
            raise ValueError("Run estimate parameters method before model finetuning")

        best_idxs = self.interpret_results(
            self.results, self.candidates_list, metric, plot=False, top_k=top_k
        )
        combinations = list(itertools.product(*best_idxs))
        for c in combinations:
            cand_list = []
            guess = []
            n_params = []
            # get ratelaws corresponding to the candidates
            for i in range(len(c)):
                if isinstance(self.candidates_list[i][c[i]], RateLaw):
                    cand_list.append(self.candidates_list[i][c[i]])
                else:
                    cand_list.append(self.candidates_list[i][c[i]])

                # values found in previous method is used as initial guesses
                guess.extend(list(self.results[i][c[i]].x))
                n_params.append(len(list(self.results[i][c[i]].x)))

            tuned_results.append(self.ft_est_params(cand_list, guess, n_params, solver))

        if metric == "rmse":
            loss = [res.fun for res in tuned_results]

        elif metric == "aic":
            Ks = np.array(
                [
                    [
                        num_params(candidates_list[i][c[i]], self.c(self.time).T)
                        if not isinstance(candidates_list[i][c[i]], RateLaw)
                        else num_params(
                            candidates_list[i][c[i]].function, self.c(self.time).T
                        )
                        for i in range(len(c))
                    ]
                    for c in combinations
                ]
            )
            Ks = np.sum(Ks, axis=1)
            n = self.time.shape[0]
            loss = [
                ((2 * Ks[idx]) + (n * np.log(res.fun**2)))
                for idx, res in enumerate(tuned_results)
            ]

        elif metric == "aicc":
            Ks = np.array(
                [
                    [
                        num_params(candidates_list[i][c[i]], self.c(self.time).T)
                        if not isinstance(candidates_list[i][c[i]], RateLaw)
                        else num_params(
                            candidates_list[i][c[i]].function, self.c(self.time).T
                        )
                        for i in range(len(c))
                    ]
                    for c in combinations
                ]
            )
            Ks = np.sum(Ks, axis=1)
            n = self.time.shape[0]
            loss = [
                (2 * Ks[idx])
                + (
                    n * np.log(res.fun**2)
                    + (2 * Ks[idx] * (Ks[idx] + 1) / (n - Ks[idx] - 1))
                )
                for idx, res in enumerate(tuned_results)
            ]

        else:
            raise ValueError(
                "Metric not supported. Available metrics are rmse, aic and aicc"
            )

        best_idx = heapq.nsmallest(1, range(len(loss)), loss.__getitem__)
        best_combination = combinations[best_idx[0]]

        print("Best RateLaws: ")
        n_param = [
            len(list(self.results[i][c].x)) for i, c in enumerate(best_combination)
        ]
        temp = 0
        for idx, c in enumerate(best_combination):
            if isinstance(self.candidates_list[idx][c], RateLaw):
                print(
                    f"For reaction {idx+1}, best ratelaw is \
                        {self.candidates_list[idx][c].expression}"
                )
                print(
                    f"Parameters estimated: {tuned_results[best_idx[0]].x[temp:temp+n_param[idx]]}"
                )
                best_result["best_ratelaws"].append(
                    self.candidates_list[idx][c].function
                )
                best_result["params"].append(
                    tuned_results[best_idx[0]].x[temp: temp + n_param[idx]]
                )
            else:
                print(
                    f"For reaction {idx+1}, best ratelaw is \
                        {self.candidates_list[idx][c].__str__()[10:-19]}"
                )
                print(
                    f"Parameters estimated: {tuned_results[best_idx[0]].x[temp:temp+n_param[idx]]}"
                )
                best_result["best_ratelaws"].append(self.candidates_list[idx][c])
                best_result["params"].append(
                    tuned_results[best_idx[0]].x[temp: temp + n_param[idx]]
                )
            temp += n_param[idx]

        best_result["tuned_results"] = tuned_results
        self.best_result = best_result
        return best_result

    def simulate_best_rl(self, plot=True):
        """
        Simulates the best rate laws obtained from model finetuning.

        Args:
            plot: Boolean flag indicating whether to plot the simulation results (default: True).

        Returns:
            Dictionary containing the simulation results.

        Notes:
            This method simulates the best rate laws obtained from model finetuning.
            It adds the rate laws and their estimated parameters to the model and
            runs the simulation.
            The method returns a dictionary containing the simulation results.
            If plot is True, it also plots the concentration profiles over time.
        """
        if not hasattr(self, "best_result"):
            raise ValueError(
                "Run estimate parameters method before simulating best ratelaws"
            )

        act_ratelaws = self.best_result["best_ratelaws"]
        K = self.best_result["params"]
        # sim = Simulate(self.N, self.Mw, self.V, self.Winhat, self.uin, self.uout, self.n0)
        self.add_ratelaws(act_ratelaws, K)
        results_normal = self.run_simulation(self.time, alpha=0)

        if plot:
            c = results_normal["moles"].shape[1]
            for i in range(c):
                plt.plot(self.time, results_normal["moles"][:, i], label=str(i))
            plt.xlabel("Time [min]")
            plt.ylabel("Conc [mol L-1]")
            plt.title("fitted plot")
            plt.legend()
            plt.show()
        return results_normal