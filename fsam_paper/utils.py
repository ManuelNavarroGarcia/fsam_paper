from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import rpy2.rinterface as rinterface
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import norm

SCENARIO_DEFINITION = {
    1: {"true_feat_lin": 6, "true_feat_nonlin": 6, "u_min": -1, "u_max": 1},
    2: {"true_feat_lin": 0, "true_feat_nonlin": 4, "u_min": -2.5, "u_max": 2.5},
    3: {"true_feat_lin": 9, "true_feat_nonlin": 0, "u_min": -1, "u_max": 1},
}


def get_y(
    X: np.ndarray, true_feat_lin: int, true_feat_nonlin: int, scenario: int = 1
) -> np.ndarray:
    """
    Given a data matrix `X` and a scenario, generates a true response variable.

    Parameters
    ----------
    X : np.ndarray
        The data matrix
    true_feat_lin : int
        Number of linear terms that are involved in the
        construction of the `y` data
    true_feat_nonlin : int
        Number of nonlinear terms that are involved in the
        construction of the `y` data
    scenario : int, optional
        The number of the scenario to generate the data, by default 1

    Returns
    -------
    np.ndarray
        The generated data

    Raises
    ------
    ValueError
        If the scenario number inputted is not implemented.
    """
    # Original proposal
    if scenario == 1:
        f1 = np.multiply(X[:, : int(true_feat_lin / 2)], 5 / 3)
        f2 = 2 * np.square(X[:, int(true_feat_nonlin / 2) : true_feat_nonlin]) - 1
        f31 = np.multiply(
            X[:, true_feat_lin : true_feat_lin + int(true_feat_lin / 2)], -5 / 4
        )
        f32 = np.cos(
            (X[:, true_feat_nonlin : true_feat_nonlin + int(true_feat_nonlin / 2)] - 1)
            * 6
        )
        y = (f1 + f2 + f31 + f32).sum(axis=1)
    # # Ravikumar, P., Lafferty, J., Liu, H., & Wasserman, L. (2009). Sparse
    # additive models. Journal of the Royal Statistical Society: Series B
    # (Statistical Methodology), 71(5), 1009-1030.
    elif scenario == 2:
        f1 = np.multiply(-1, np.sin(np.multiply(1.5, X[:, :true_feat_nonlin])))
        f2 = np.power(X[:, :true_feat_nonlin], 3) + np.multiply(
            1.5, np.square(X[:, :true_feat_nonlin] - 0.5)
        )
        f3 = np.multiply(-1, norm.pdf(X[:, :true_feat_nonlin], loc=0.5, scale=0.8))
        f4 = np.sin(np.exp(np.multiply(-0.5, X[:, :true_feat_nonlin])))
        y = (f1 + f2 + f3 + f4).sum(axis=1)
    # Original proposal
    elif scenario == 3:
        y = (
            np.multiply(1.5, X[:, : int(true_feat_lin / 3)])
            + np.multiply(
                11 / 3, X[:, int(true_feat_lin / 3) : int(2 * true_feat_lin / 3)]
            )
            + np.multiply(1 / 3, X[:, int(2 * true_feat_lin / 3) : true_feat_lin])
        ).sum(axis=1)
    else:
        raise ValueError("Scenario not implemented.")
    return y


def get_r_results(
    script_path: Path,
    method_name: str,
    X: rinterface.FloatSexpVector,
    y: rinterface.FloatSexpVector,
    X_test: rinterface.FloatSexpVector,
    y_test: rinterface.FloatSexpVector,
    seed: int,
    output_cols: list[str],
    criterion: str = "mse",
    train_size: Union[int, float] = 0.7,
    **kwargs,
) -> pd.DataFrame:
    """Computes the algorithm given by `method_name` found in the R script
    `script_path` using the data provided. It returns a Pandas DataFrame with
    `output_cols` columns. The `kwargs` are referred to the hyperparameters each
    method may contain.

    Parameters
    ----------
    script_path : Path
        Path of the R script where the algorithm is located.
    method_name : str
        Name of the R function that computes the algorithm.
    X : rinterface.FloatSexpVector
        The data matrix used to fit the algorithm.
    y : rinterface.FloatSexpVector
        The response variable sample used to fit the algorithm.
    X_test : rinterface.FloatSexpVector
        The test data matrix used to test the algorithm
    y_test : rinterface.FloatSexpVector
        The response test variable sample.
    seed : int
        The initial random seed of the experiments.
    output_cols : list[str]
        The columns of the output dataframe.
    criterion : str
        Indicates the criterion under the best hyperparameter is selected. By
        default `mse`
    train_size : Union[int, float], optional
        A number between 0 and 1 that indicates the fraction `X` to be used
        for training. By default 0.7
            - If float, should be between 0.0 and 1.0 and represent the
                proportion of the dataset to include in the train split.
            - If int, represents the absolute number of train samples.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns `output_cols` containing the metrics computed
        in the algorithm.

    Raises
    -------
    ValueError
        The criterion must be a string valued "mse" or "mae".
    """
    if criterion not in ["mse", "mae"]:
        raise ValueError("The criterion must be a string valued 'mse' or 'mae'.")
    r = robjects.r
    _ = r["source"](str(script_path))
    method = robjects.globalenv[method_name]
    with localconverter(robjects.default_converter + pandas2ri.converter):
        out = robjects.conversion.rpy2py(
            method(
                X=X,
                y=y,
                X_test=X_test,
                y_test=y_test,
                seed=seed,
                train_size=train_size,
                output_cols=output_cols,
                criterion=criterion,
                **kwargs,
            )
        )
    return out
