import pandas as pd
import numpy as np
import scipy as sc
from csaps import csaps  # for cubic smoothing splines
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed


def spline_calculator(t, y, smooth, refit, n_jobs):
    """Does cubic smooth spline approximation on a 1d dataset.

    Args:
        t: The time array.
        y: The 1D dataset array.
        smooth: The smoothing factor or regularization parameter.
        refit: Boolean flag indicating whether to fit the smoothing spline on the entire data.
        n_jobs: Number of cores to be used for parallelization.

    Returns:
        The cubic smoothing spline.

    """
    if isinstance(smooth, (int, float)):
        spline = csaps(t, y, smooth=smooth)

    else:
        t_train, t_test, y_train, y_test = train_test_split(
            t[1:-1], y[1:-1], test_size=0.3, random_state=42
        )

        t_train = np.append(t_train, [t[0], t[-1]])
        y_train = np.append(y_train, [y[0], y[-1]])

        merge_train = list(zip(t_train, y_train))
        merge_test = list(zip(t_test, y_test))
        merge_train.sort(key=lambda x: x[0])
        merge_test.sort(key=lambda x: x[0])

        t_train, y_train = zip(*merge_train)
        t_test, y_test = zip(*merge_test)

        t_train = np.array(t_train)
        y_train = np.array(y_train)
        t_test = np.array(t_test)
        y_test = np.array(y_test)

        param_grid = np.linspace(0, 1, 101)

        err = float("inf")
        results = Parallel(n_jobs=n_jobs)(
            delayed(spline_error)(t_train, y_train, t_test, y_test, param)
            for param in param_grid
        )

        for result in results:
            param, curr_err = result
            if curr_err < err:
                err = curr_err
                best_lambda = param

        if refit:
            spline = csaps(t, y, smooth=best_lambda)

        else:
            spline = csaps(t_train, y_train, smooth=best_lambda)

    return spline


def spline_error(t_train, y_train, t_test, y_test, param):
    """Calculates the error for a given smoothing parameter.

    Args:
        t_train: The time array of the training data.
        y_train: The training dataset array.
        t_test: The time array of the test data.
        y_test: The test dataset array.
        param: The smoothing parameter.

    Returns:
        Tuple containing the parameter value and the error.

    """
    spline = csaps(t_train, y_train, smooth=param)
    ytest_pred = spline(t_test)
    error = mean_squared_error(y_test, ytest_pred)
    return (param, error)


def cubic_smooth(t, y, smooth, refit, n_jobs=-1):
    """Does cubic smoothing spline on a 2d dataset fitting a spline to each column.

    Args:
        t: The time array.
        y: The 2D dataset array.
        smooth: The smoothing factor or regularization parameter.
        refit: Boolean flag indicating whether to fit the smoothing spline on the entire data.
        n_jobs: Number of cores to be used. Default is -1 (all available cores).

    Returns:
        The final spline function and its derivative.

    """
    if len(y.shape) == 1:
        final_spline = spline_calculator(t, y, smooth, refit, n_jobs)
        final_derivative = final_spline.spline.derivative(nu=1)

    elif len(y.shape) == 2:
        splines = [
            spline_calculator(t, y[:, i], smooth, refit, n_jobs)
            for i in range(y.shape[1])
        ]
        final_spline = lambda x: np.array(  # noqa: E731
            [spline(x) for spline in splines]
        ).T
        final_derivative = lambda x: np.array(  # noqa: E731
            [spline.spline.derivative(nu=1)(x) for spline in splines]  # noqa: E731
        ).T

    else:
        raise ValueError(
            "y should be of shape Nx1 or Nxm for which m splines would be \
                fit. Cannot be more than 2-dimensions."
        )
    return final_spline, final_derivative


def read_file(file_path):
    """Reads a file from various formats using pandas library in Python.

    Args:
        file_path: The path of the file to be read.

    Returns:
        A pandas DataFrame object.

    """
    file_extension = file_path.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(file_path, index_col=0)
    elif file_extension == "xls" or file_extension == "xlsx":
        df = pd.read_excel(file_path, index_col=0, header=None)
    elif file_extension == "json":
        df = pd.read_json(file_path)
        df = df.set_index(df.columns[0])
    elif file_extension == "txt":
        df = pd.read_table(file_path, header=None, index_col=0)
    elif file_extension == "tsv":
        df = pd.read_csv(file_path, sep="\t", header=None, index_col=0)
    else:
        raise ValueError(
            "Unsupported file format. Please provide a CSV, Excel, JSON, TSV or Text file."
        )
    return df


def num_params(f, args, max_lim=1000):
    """Returns the number of parameters in a function.

    Args:
        f: The function for which the number of arguments needs to be found.
        args: The arguments for the function.
        max_lim: The maximum number of parameters to consider. Default is 1000.

    Returns:
        The number of parameters used in the function.

    """
    for i in range(max_lim):
        temp = np.ones(i)
        try:
            f(args, temp)
            return len(temp)
        except:  # noqa: E722
            pass

    raise ValueError(f"Number of Parameters exceed {max_lim}")


def baseline_shift(values):
    """Performs baseline shift if values contain negative values.

    Args:
        values: The array of values.

    Returns:
        The shifted values.

    """
    mini = np.min(values, axis=0)
    add_ = np.abs(np.minimum(mini, np.array([0] * len(mini))))
    values_new = values + add_
    return values_new


def threshold_cutoff(values, threshold):
    """Replaces values less than the threshold with the threshold value.

    Args:
        values: The array of values.
        threshold: The threshold value.

    Returns:
        The modified values.

    """
    values[values < threshold] = threshold
    return values


def savgol_filter(
    values,
    window_length,
    polyorder,
    deriv=0,
    delta=1.0,
    axis=0,
    mode="interp",
    cval=0.0,
):
    """Applies the Savitzky-Golay filter to the values.

    Args:
        values: The array of values.
        window_length: The length of the window for filtering.
        polyorder: The order of the polynomial to fit.
        deriv: The order of the derivative to compute. Default is 0.
        delta: The sampling interval. Default is 1.0.
        axis: The axis along which to apply the filter. Default is 0.
        mode: The extrapolation mode. Default is "interp".
        cval: The constant value for boundary extrapolation. Default is 0.0.

    Returns:
        The filtered values.

    """
    r = sc.signal.savgol_filter(
        values, window_length, polyorder, deriv, delta, axis=axis, mode=mode, cval=cval
    )
    return r
