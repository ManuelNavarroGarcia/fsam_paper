#!/usr/bin/env python
import ast
import itertools
import os
import urllib.request
import zipfile
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import typer
from fsam.fsam_fit import FSAM
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from sklearn.linear_model import Lasso, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from utils import get_r_results


def download_url(url, save_path):
    try:
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, "wb") as out_file:
                out_file.write(dl_file.read())
    except HTTPError:
        raise HTTPError("File Not Found")


def main(
    deg: int = 3,
    ord_d: int = 2,
    n_int: int = 20,
    K: list[int] = list(range(1, 79 * 2 + 1)),
    seed: int = 0,
    cv: int = 5,
):
    """Solve the feature selection problem with the superconductivity dataset.

    Parameters
    ----------
    deg : int, optional
        The degree of the B-spline basis, by default 3
    ord_d : int, optional
        The penalty order, by default 2
    n_int : int, optional
        The number of intervals used to defined the knot sequences of the B-spline
        bases, by default 20
    seed : int, optional
        The initial random seed of the experiments, by default 0
    cv : int, optional
        The number of folds used in the cross-validatino, by default 5
    """

    download_url(
        "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
        "superconductors.zip",
    )

    with zipfile.ZipFile("superconductors.zip", "r") as zip_ref:
        zip_ref.extractall("")

    os.remove("unique_m.csv")
    os.remove("superconductors.zip")

    np.random.seed(seed)
    # Configuration and model parameters
    conf_gurobi = {
        "OutputFlag": 1,
        "threads": 1,
        "timelimit": 3600,
        "MIPFocus": 0,
        "MIPGap": 1e-4,
    }
    conf_model = {
        "criterion": "mae",
        "q": 30,
        "max_time": 1000000,
        "max_iter": 1000000,
        "n_iter_no_change": 20,
        "n_iter_pgl": 2000,
        "eps": 1e-2,
        "tol_pgl": 0.001,
        "n_alphas": 50,
        "patience": 10,
        "min_edf": 1,
    }
    output_cols = ["MSE", "MAE", "nzero_bool", "Method"]

    df = pd.read_csv("train.csv").drop(columns=["number_of_elements", "range_Valence"])
    df["critical_temp"] = np.log(df["critical_temp"])
    X, y = df.drop(columns=["critical_temp"]), df["critical_temp"]
    m = X.shape[1]

    df_list = []
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    for i, (train_val_index, test_index) in enumerate(kf.split(X)):
        X_train_val = X.loc[train_val_index, :].sort_index()
        y_train_val = y.loc[train_val_index].sort_index()
        X_test = X.loc[test_index, :].sort_index()
        y_test = y.loc[test_index].sort_index()

        train_size = 0.7
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            train_size=train_size,
            shuffle=False,
            random_state=0,
        )
        X_train_val = pd.concat((X_train, X_val))

        prediction = [
            {"backwards": col.min(), "forward": col.max()}
            for col in np.concatenate([X_train_val, X_test], axis=0).T
        ]

        var_sec = FSAM(
            deg=[deg] * X.shape[1],
            ord_d=[ord_d] * X.shape[1],
            n_int=[n_int] * X.shape[1],
            prediction=prediction,
        )
        var_sec.fit(
            X=X_train_val.values,
            y=y_train_val.values,
            K=K,
            train_size=train_size,
            conf_gurobi=conf_gurobi,
            conf_model=conf_model,
            warm_start=True,
            frac_row_bounds=0,
        )

        # Compute and save measures in dataset
        pred_test = var_sec.predict(X=X_test.values).flatten()
        out_fsam = pd.DataFrame(
            data=[
                mean_squared_error(y_test, pred_test),
                mean_absolute_error(y_test, pred_test),
                var_sec.sol["z"],
                "FSAM",
            ],
            index=output_cols,
        ).T

        # Convert python arrays into r objects
        with localconverter(robjects.default_converter + numpy2ri.converter):
            X_ = robjects.conversion.py2rpy(X_train_val.values)
            x_te = robjects.conversion.py2rpy(X_test.values)
            y_ = robjects.conversion.py2rpy(y_train_val.values)
            y_te = robjects.conversion.py2rpy(y_test.values)

        # SAM
        try:
            out_sam = get_r_results(
                script_path="fsam_paper/R_scripts/SAM.R",
                method_name="sam_out",
                X=X_,
                y=y_,
                X_test=x_te,
                y_test=y_te,
                seed=seed,
                output_cols=output_cols,
                train_size=train_size,
                criterion=conf_model["criterion"],
            )
        except RRuntimeError:
            out_sam = pd.DataFrame(
                [np.nan, np.nan, str([0] * 2 * m), "SAM"], index=output_cols
            ).T
        # GAMSEL
        try:
            out_gamsel = get_r_results(
                script_path="fsam_paper/R_scripts/Gamsel.R",
                method_name="gamsel_out",
                X=X_,
                y=y_,
                X_test=x_te,
                y_test=y_te,
                seed=seed,
                output_cols=output_cols,
                train_size=train_size,
                criterion=conf_model["criterion"],
            )
        except RRuntimeError:
            out_gamsel = pd.DataFrame(
                [np.nan, np.nan, str([0] * 2 * m), "GAMSEL"], index=output_cols
            ).T
        # Relgam
        out_relgam = []
        for relgam_sel in [False, True]:
            try:
                out_relgam_ = get_r_results(
                    script_path="fsam_paper/R_scripts/relgam.R",
                    method_name="relgam_out",
                    X=X_,
                    y=y_,
                    X_test=x_te,
                    y_test=y_te,
                    seed=seed,
                    output_cols=output_cols,
                    train_size=train_size,
                    criterion=conf_model["criterion"],
                    **{"relgam_sel": relgam_sel},
                )
            except RRuntimeError:
                out_relgam_ = pd.DataFrame(
                    [np.nan, np.nan, str([0] * 2 * m), "RELGAM"], index=output_cols
                ).T
            out_relgam.append(out_relgam_)
        out_r = pd.concat([out_sam, out_gamsel] + out_relgam).reset_index(drop=True)

        # Lasso
        alphas_path = lasso_path(X=X_train, y=y_train, random_state=0)[0]
        lasso_models = []
        lasso_mses = []

        for alpha in alphas_path:
            lasso = Lasso(alpha=alpha, random_state=0).fit(X=X_train, y=y_train)
            lasso_models.append(lasso)
            lasso_mses.append(mean_squared_error(y_val, lasso.predict(X=X_val)))

        opt_alpha = np.argmin(lasso_mses)
        lasso = lasso_models[opt_alpha]

        mse_lasso = mean_squared_error(y_test, lasso.predict(X=X_test))
        mae_lasso = mean_absolute_error(y_test, lasso.predict(X=X_test))
        lasso_vars = list((lasso.coef_ != 0).astype(int))
        out_lasso = pd.DataFrame(
            data=[
                mse_lasso,
                mae_lasso,
                list(itertools.chain(lasso_vars, [0] * m)),
                "LASSO",
            ],
            index=output_cols,
        ).T

        df_ = pd.concat((out_fsam, out_r, out_lasso)).reset_index(drop=True)
        df_["nzero_bool"] = df_["nzero_bool"].apply(str).apply(ast.literal_eval)
        df_ = df_.assign(
            **{
                "seed": i,
                "n_lin": df_["nzero_bool"].apply(lambda x: sum(x[:m])),
                "n_nonlin": df_["nzero_bool"].apply(lambda x: sum(x[m:])),
            }
        ).drop(["nzero_bool"], axis=1)
        df_list.append(df_)

    pd.concat(df_list).astype({"MSE": float, "MAE": float}).to_csv(
        "data/results_superconductivity.csv", index=False
    )

    os.remove("train.csv")


if __name__ == "__main__":
    typer.run(main)
