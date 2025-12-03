import pandas as pd
import numpy as np
import scipy as sc
from preprocess import cubic_smooth, baseline_shift, read_file
from preprocess import threshold_cutoff, savgol_filter
import warnings


class DataTools:
    def __init__(self):
        self.N = None
        self.Na = None 
        self.Nu = None 
        self.R = None 
        self.S = None 
        self.Mw = None 
        self.V = None
        self.Winhat = None 
        self.Win = None
        self.Wina = None 
        self.Winu = None 
        self.P = None
        self.uin = None 
        self.uout = None 
        self.n0 = None 
        self.n0a = None 
        self.n0u = None 
        self.time = None 
        self.methodology = None
        
    # wanted to use something like default dict but in a callable way
    # instead of subscriptable, hence this hack
    def default_dict_v(self, x):
        """Returns an array with self.v_value with length of x.

        Args:
            x: A value or list-like object.

        Returns:
            numpy.ndarray or scalar: If x is list-like, returns a 1D numpy array of length len(x)
                with all elements set to self.v_value. If x is not list-like, returns self.v_value.
        """
        if pd.api.types.is_list_like(x):
            return np.array([self.v_value] * len(x))
        return self.v_value

    def default_dict_uin(self, x):
        """Returns a broadcasted numpy array of shape (len(x),) + self.uin_value.shape
        if x is list-like, otherwise returns self.uin_value.

        Args:
            x: A value or list-like object.

        Returns:
            numpy.ndarray or scalar: If x is list-like, returns a broadcasted numpy array of shape
                (len(x),) + self.uin_value.shape with all elements set to self.uin_value.
                If x is not list-like, returns self.uin_value.
        """
        if pd.api.types.is_list_like(x):
            return np.broadcast_to(self.uin_value, (len(x),) + self.uin_value.shape)
        return self.uin_value

    def default_dict_uout(self, x):
        """Returns an array with self.uout_value with length of x.

        Args:
            x: A value or list-like object.

        Returns:
            numpy.ndarray or scalar: If x is list-like, returns a 1D numpy array of length len(x)
                with all elements set to self.uout_value.
                If x is not list-like, returns self.uout_value.
        """
        if pd.api.types.is_list_like(x):
            return np.array([self.uout_value] * len(x))
        return self.uout_value

    def add_stoichiometry_data(self, N):
        """Adds stoichiometry data to the class instance and updates related attributes.

        Args:
            N: A list-like or numpy array representing the stoichiometry matrix.

        Note:
            - If N is a list, it will be converted to a numpy array.
            - The shape of the stoichiometry matrix should be (R, S),
            where R represents the number of reactions
            and S represents the number of species.
        """
        if isinstance(N, list):
            N = np.array(N)
        self.N = N
        self.R = N.shape[0]
        self.S = N.shape[1]

    def add_molweight_data(self, Mw):
        """Adds molecular weight data to the class instance and updates the Mw attribute.

        Args:
            Mw: A list-like or numpy array representing the molecular weight matrix.

        Raises:
            ValueError: If the molecular weight matrix Mw is not a square matrix.
        """
        if isinstance(Mw, list):
            if isinstance(Mw[0], list):
                Mw = np.array(Mw)
            else:
                Mw = np.diag(Mw)

        if Mw.shape[0] != Mw.shape[1]:
            raise ValueError("Molecular weight matrix: Mw must be a square matrix")

        self.Mw = Mw

    def add_volume_data(self, V, kind="linear"):
        """Adds volume data to the class instance and updates the V attribute.

        Args:
            V: A value, dataframe, dictionary, or file location representing the volume data.
            kind (str, optional): The interpolation method for extending the data.
                Defaults to "linear".

        Raises:
            ValueError: If the volume data has an incorrect format.
        """
        if isinstance(V, (int, float)):
            self.v_value = V
            self.V = self.default_dict_v

        else:
            if isinstance(V, pd.DataFrame):
                time = V.index
                y = np.array(V.iloc[:, 0])

            elif isinstance(V, dict):
                time = list(V.keys())
                y = list(V.values())

            elif isinstance(V, str):
                V = read_file(V)
                time = V.index
                y = np.array(V.iloc[:, 0])

            else:
                raise ValueError(
                    "Incorrect format for volume data. Supported format: \
                        (int, float), dataframe, dict or file location"
                )

            # extending the data linearly using the last two data points.
            # Since It is normal for the odeint solver to evaluate your function
            # at time values past the last requested time. Most ODE solvers work
            # this way--they take internal time steps with sizes determined by their
            # error control algorithm, and then use their own interpolation
            # to evaluate the solution at the times requested by the user.

            dt = max(3, (time[-1] - time[0]) * 5)
            # Slope of the last segment.
            m = (y[-1] - y[-2]) / (time[-1] - time[-2])
            # Extended final time.
            time_ext = time[-1] + dt
            # Extended final data value.
            y_ext = y[-1] + m * dt
            # Extended arrays.
            time = np.append(time, time_ext)
            y = np.append(y, y_ext)

            if kind == "cubic_smooth":  # TEST CAREFULLY
                self.V, _ = cubic_smooth(
                    time, y, smooth=0.98, refit=False
                )  # _ is first derivative

            else:  # Returns (t,p). Will have to fix.
                self.V = sc.interpolate.interp1d(time, y, kind=kind)

    def add_Winhat_data(self, Winhat):
        """Adds Winhat data to the class instance and updates the Winhat attribute.

        Args:
            Winhat: A list-like or numpy array representing the Winhat data.

        """
        if isinstance(Winhat, list):
            Winhat = np.array(Winhat)
        self.Winhat = Winhat

        if len(Winhat.shape) == 1:
            self.P = 1

        else:
            self.P = Winhat.shape[1]

    def add_uin_data(self, uin, kind="linear"):
        """Adds uin data to the class instance and updates the uin attribute.

        Args:
            uin: A value, list, numpy array, pandas DataFrame, or file location
                representing the uin data.
            kind (str, optional): The interpolation method for extending the data.
                Defaults to "linear".

        Raises:
            ValueError: If the uin data has an unsupported file format.
        """
        if isinstance(uin, (float, int)):
            uin = np.array([uin])
            self.uin_value = uin
            self.uin = self.default_dict_uin

        elif isinstance(uin, list):
            uin = np.array(uin)
            uin = np.squeeze(uin)
            self.uin_value = uin
            self.uin = self.default_dict_uin

        elif isinstance(uin, np.ndarray):
            uin = np.squeeze(uin)
            self.uin_value = uin
            self.uin = self.default_dict_uin

        elif isinstance(uin, (pd.DataFrame, str)):
            if isinstance(uin, str):
                uin = read_file(uin)

            time = uin.index
            values = uin.values

            dt = max(3, (time[-1] - time[0]) * 4)
            # Slope of the last segment.
            m = (values[-1] - values[-2]) / (time[-1] - time[-2])
            # Extended final time.
            time_ext = time[-1] + dt
            # Extended final data value.
            values_ext = values[-1] + m * dt
            # Extended arrays.
            time = np.append(time, time_ext)
            values = np.vstack([values, values_ext])

            if kind == "cubic_smooth":  # TEST CAREFULLY
                self.uin, _ = cubic_smooth(time, values, smooth=0.98, refit=False)
            else:
                self.uin = sc.interpolate.interp1d(time, values, kind=kind, axis=0)

        else:
            raise ValueError(
                "Unsupported file format. uin should be either int, float, list, \
                    numpy array (p,1) shape, pandas dataframe or file directory."
            )

    def add_uout_data(self, uout, kind="linear"):
        """Adds uout data to the class instance and updates the uout attribute.

        Args:
            uout: A value, pandas DataFrame, or file location representing the uout data.
            kind (str, optional): The interpolation method for extending the
                data. Defaults to "linear".

        Raises:
            ValueError: If the uout data has an unsupported file format.
        """
        if isinstance(uout, (float, int)):
            self.uout_value = np.array([uout])
            self.uout = self.default_dict_uout

        elif isinstance(uout, (pd.DataFrame, str)):
            if isinstance(uout, str):
                uout = read_file(uout)

            time = uout.index
            values = uout.values

            dt = max(3, (time[-1] - time[0]) * 4)
            # Slope of the last segment.
            m = (values[-1] - values[-2]) / (time[-1] - time[-2])
            # Extended final time.
            time_ext = time[-1] + dt
            # Extended final data value.
            values_ext = values[-1] + m * dt
            # Extended arrays.
            time = np.append(time, time_ext)
            values = np.vstack([values, values_ext])

            if kind == "cubic_smooth":  # TEST CAREFULLY
                self.uout, _ = cubic_smooth(time, values, smooth=0.98, refit=False)
            else:
                self.uout = sc.interpolate.interp1d(time, values, kind=kind, axis=0)

        else:
            raise ValueError(
                "Unsupported file format. uout should be either int, float, pandas \
                    dataframe or file directory."
            )

    def add_n0_data(self, n0):
        """Adds n0 data to the class instance and updates the n0 attribute.

        Args:
            n0: A list-like or numpy array representing the n0 data.
        """
        if isinstance(n0, list):
            n0 = np.array(n0)
        self.n0 = n0

    def add_reactor_config(self, V, Winhat, uin, uout, config):
        """Adds reactor configuration data to the class instance.

        Args:
            V: The volume of the reactor.
            Winhat: The Winhat data.
            uin: The uin data.
            uout: The uout data.
            config (str): The configuration type of the reactor. Supported values:
                "batch", "semi-batch", "cstr".

        Returns:
            tuple: A tuple containing the updated values for V, Winhat, uin, and uout.

        Raises:
            ValueError: If the configuration is invalid or if required arguments
                are not provided for a specific configuration.
        """
        if config is None and ((Winhat is None) or (uin is None) or (uout is None)):
            raise ValueError(
                "If config is not specified, other arguments should be provided"
            )

        if config == "batch":
            if Winhat is not None:
                warnings.warn(
                    f"Winhat is provided for config = {config}. Using Winhat = {[0]*self.S}",
                    stacklevel=2,
                )
            if uin is not None:
                if uin == 0:
                    pass
                else:
                    raise ValueError(
                        f"uin should not be specified for config = {config}. \
                            Non-zero uin provided."
                    )
            if uout is not None:
                if uout == 0:
                    pass
                else:
                    raise ValueError(
                        f"uout should not be specified for config = {config}. \
                            Non-zero uout provided."
                    )

            Winhat = [0] * self.S
            uin = 0
            uout = 0

        elif config == "semi-batch":
            if Winhat is None:
                raise ValueError(f"Winhat cannot be none/empty for config = {config}")
            if uin is None:
                raise ValueError(f"uin cannot be none/empty for config = {config}")
            if uout is not None:
                if uout == 0:
                    pass
                else:
                    raise ValueError(
                        f"uout should not be specified for config = {config}. \
                            Non-zero uout value provided."
                    )
            uout = 0

        elif config == "cstr":
            pass

        else:
            pass

        return V, Winhat, uin, uout

    def add_concentration_data(
        self,
        conc,
        time=None,
        kind="cubic_smooth",
        smooth=None,
        refit=False,
        preprocess=None,
        threshold=0,
        window_length=None,
        polyorder=3,
        deriv=0,
        delta=1.0,
        axis=0,
        mode="interp",
        cval=0.0,
    ):
        """Adds concentration data to the class instance.

        Args:
            - conc (Union[pd.DataFrame, np.ndarray, str]): The concentration data.
                It can be provided as a pandas DataFrame, numpy array, or a file directory (str).
            - time (Optional[np.ndarray]): The time data corresponding to
                the concentration measurements.
            - kind (str): The interpolation method for smoothing the concentration data.
                 Default is "cubic_smooth".
            - smooth (Optional[float]): The smoothing parameter for cubic smoothing.
                 Default is None.
            - refit (bool): Whether to refit the smoothing function after extending the data.
                 Default is False.
            - preprocess (Optional[str]): Preprocessing method for the concentration data.
                 Supported values: "baseline_shift","threshold_cutoff", "savgol_filter".
                 Default is None.
            - threshold (float): The threshold value for threshold_cutoff preprocessing.
                 Default is 0.
            - window_length (Optional[int]): The window length for savgol_filter
                preprocessing. Default is None.
            - polyorder (int): The polynomial order for savgol_filter
                preprocessing. Default is 3.
            - deriv (int): The derivative order for savgol_filter preprocessing. Default is 0.
            - delta (float): The spacing between the time points for savgol_filter
                preprocessing. Default is 1.0.
            - axis (int): The axis along which the savgol_filter is applied. Default is 0.
            - mode (str): The extrapolation mode for savgol_filter
                preprocessing. Default is "interp".
            - cval (float): The constant value used for extrapolation
                when mode is "constant". Default is 0.0.

        Raises:
            ValueError: If the concentration data is not provided or
            if the problem is non-identifiable with the concentration provided.
        """
        self.methodology = None
        if conc is None:
            raise ValueError(
                "Concentration data is not provided. Missing concentration data"
            )

        if time is None:
            if isinstance(conc, (pd.DataFrame, str)):
                if isinstance(conc, str):
                    conc = read_file(conc)

                time = np.array(conc.index)
                conc = conc.values

        self.time = time
        unavailable_idx = [idx for idx, c in enumerate(conc[0]) if (np.isnan(c))]
        available_idx = [idx for idx, c in enumerate(conc[0]) if (not np.isnan(c))]

        if unavailable_idx == []:
            if True in np.isnan(self.N):
                raise ValueError("Stoichiometric Matrix contains NaN Value(s)")
            if True in np.isnan(self.Mw):
                raise ValueError("Molecular weight matrix (Mw) contains NaN Value(s). ")
            if True in np.isnan(self.Winhat):
                raise ValueError("Winhat contains NaN values")
            if True in np.isnan(self.n0):
                warnings.warn("n0 contains NaN values.", stacklevel=2)
                self.methodology = (
                    "rate"  # if n0 contains nan, only rate based can be possible
                )

            n0_ = np.reshape(self.n0, (self.n0.shape[0], 1))
            mat = np.concatenate([self.N.T, self.Win, n0_], axis=1)

            if (np.linalg.matrix_rank(mat) == self.R + self.P + 1) and (
                self.methodology is None
            ):
                self.methodology = 1  # Here, it can be both rate based or extent based
            else:
                raise ValueError(
                    "Rank of [N.T, Win, n0_] != R + p + 1. Problem is not identifiable."
                )

        else:
            self.unavailable_idx = unavailable_idx
            self.available_idx = available_idx
            conc = conc[:, available_idx]

            if hasattr(self, "N"):
                self.Na = self.N[:, available_idx]
                self.Nu = self.N[:, unavailable_idx]

            if hasattr(self, "Win"):
                self.Wina = self.Win[available_idx, :]
                self.Winu = self.Win[unavailable_idx, :]

            if hasattr(self, "n0"):
                self.n0a = self.n0[available_idx]
                self.n0u = self.n0[unavailable_idx]

            if True in np.isnan(self.N):
                raise ValueError("Stoichiometric Matrix contains NaN Value(s)")
            if True in np.isnan(self.Mw):
                raise ValueError("Molecular weight matrix (Mw) contains NaN Value(s). ")
            if True in np.isnan(self.Winhat):
                raise ValueError("Winhat contains NaN values")
            if True in np.isnan(self.n0):
                raise ValueError("n0 contains NaN values.")

            if np.linalg.matrix_rank(self.Na) != self.R:
                raise ValueError(
                    f"Rank of Available stoichiometric matrix (Na) should be atleast rank {self.R}"
                )

            self.methodology = 2  # Here, it is only extent based

        if preprocess == "baseline_shift":
            conc = baseline_shift(conc)

        elif preprocess == "threshold_cutoff":
            conc = threshold_cutoff(conc, threshold=threshold)

        elif preprocess == "savgol_filter":
            if window_length is None:
                window_length = int(
                    time.shape[0] / 5
                )  # default value for window length is timesteps/5

            conc = savgol_filter(
                conc, window_length, polyorder, deriv, delta, axis, mode, cval
            )

        n = (conc.T * self.V(time)).T  # number of moles
        dt = max(3, (time[-1] - time[0]) * 4)
        # Slope of the last segment.
        m1 = (conc[-1] - conc[-2]) / (time[-1] - time[-2])
        m2 = (n[-1] - n[-2]) / (time[-1] - time[-2])
        # Extended final time.
        time_ext = time[-1] + dt
        # Extended final data value.
        conc_ext = conc[-1] + m1 * dt
        n_ext = n[-1] + m2 * dt
        # Extended arrays.
        time = np.append(time, time_ext)
        conc = np.vstack([conc, conc_ext])
        n = np.vstack([n, n_ext])

        if kind == "cubic_smooth":
            self.n, self.dndt = cubic_smooth(
                time, n, smooth=smooth, refit=refit
            )  # this is a function
            # self.c = ((self.n(time)).T / self.V(time)).T # smoothed conc values, this is np array
            self.c, self.dcdt = cubic_smooth(
                time, conc, smooth=smooth, refit=refit
            )  # this is a function
        else:
            self.n = sc.interpolate.interp1d(time, n, kind)  # this is a function
            # self.c = ((self.n(time)).T / self.V(time)).T # smoothed conc values, this is np array
            self.c = sc.interpolate.interp1d(time, conc, kind)  # this is a function
