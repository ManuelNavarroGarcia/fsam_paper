#!/usr/bin/env python
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import typer
from fsam.fsam_fit import FSAM
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from sklearn.linear_model import Lasso, lasso_path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from utils import get_r_results

from fsam_paper.utils import SCENARIO_DEFINITION, get_y


def main(
    deg: int = 3,
    ord_d: int = 2,
    n_int: int = 20,
    scenario: int = 1,
    seed: int = 0,
    n_iter: int = 50,
    snr: int = 1,
    n_vars: int = 100,
    n_samples: int = 400,
    K: list[int] = list(range(1, 2 * 100 + 1)),
):
    """Given the elements to construct the B-spline bases and a scenario, solves our
    model and compares it with other state-of-the-art algorithms. The results are saved
    into a CSV file.

    Parameters
    ----------
    deg : int, optional
        The degree of the B-spline basis, by default 3
    ord_d : int, optional
        The penalty order, by default 2
    n_int : int, optional
        The number of intervals used to defined the knot sequences of the B-spline
        bases, by default 20
    scenario : int, optional
        The ID number of scenario, by default 1
    seed : int, optional
        The initial random seed of the experiments, by default 0
    n_iter : int, optional
        The number of iterations, by default 50
    snr : int, optional
        The signal-to-noise ratio parameter, by default 1
    n_vars : int, optional
        The number of features, by default 100
    n_samples : int, optional
        The number of available features, both for training and validation. By default
        400
    K : list[int], optional
        A list with different values for the sparsity parameter, by default
        list(range(1, 2 * 100 + 1))
    """
    df_list = []

    conf_gurobi = {
        "OutputFlag": 0,
        "threads": 1,
        "timelimit": 60,
        "MIPFocus": 0,
        "MIPGap": 1e-4,
    }
    conf_model = {
        "criterion": "mae",
        "q": 30,
        "max_time": 3600,
        "max_iter": 1000000,
        "n_iter_no_change": 20,
        "n_iter_pgl": 10000,
        "eps": 1e-1,
        "tol_pgl": 0.1,
        "n_alphas": 200,
        "patience": 10,
        "min_edf": 1,
    }
    cols = ["MSE", "MAE", "nzero_bool", "Method"]

    true_feat_lin = SCENARIO_DEFINITION[scenario]["true_feat_lin"]
    true_feat_nonlin = SCENARIO_DEFINITION[scenario]["true_feat_nonlin"]
    u_min = SCENARIO_DEFINITION[scenario]["u_min"]
    u_max = SCENARIO_DEFINITION[scenario]["u_max"]

    n_test = 5000
    train_size = 0.8

    for i in tqdm(range(n_iter), desc="Iterations: ", leave=True, colour="CYAN"):
        np.random.seed(seed + i)

        # Generate from Uniform (`u_min`, `u_max`) the train and test sets
        X_sim = np.random.uniform(
            low=u_min, high=u_max, size=(n_samples + n_test, n_vars)
        )
        x_min = X_sim.min(axis=0)
        x_max = X_sim.max(axis=0)
        X_sim = np.concatenate(
            [
                np.random.uniform(low=xmi, high=xma, size=(n_samples + n_test - 2, 1))
                for xmi, xma in zip(x_min, x_max)
            ],
            axis=1,
        )
        X_sim = np.concatenate(
            [np.expand_dims(x_max, axis=0), np.expand_dims(x_min, axis=0), X_sim]
        )
        # Generate train and test response variable, where the first is
        # perturbed with Gaussian noise for given SNR
        y_sim = get_y(
            X=X_sim,
            true_feat_lin=true_feat_lin,
            true_feat_nonlin=true_feat_nonlin,
            scenario=scenario,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_sim, y_sim, train_size=n_samples, shuffle=False, random_state=seed
        )
        assert all(X_test.min(axis=0) > X_train.min(axis=0))
        assert all(X_test.max(axis=0) < X_train.max(axis=0))

        scale = y_train.std() / np.sqrt(snr)

        y_train += np.random.normal(loc=0, scale=scale, size=n_samples)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, train_size=train_size, shuffle=False, random_state=seed
        )
        assert all(X_val.min(axis=0) > X_train.min(axis=0))
        assert all(X_val.max(axis=0) < X_train.max(axis=0))

        X = np.concatenate([X_train, X_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)

        # Solve our model and save the results
        var_sec = FSAM(
            deg=[deg] * n_vars, ord_d=[ord_d] * n_vars, n_int=[n_int] * n_vars
        )
        var_sec.fit(
            X=X,
            y=y,
            K=K,
            train_size=X_train.shape[0],
            conf_gurobi=conf_gurobi,
            conf_model=conf_model,
        )

        mse = np.array([mean_squared_error(y_test, var_sec.predict(X=X_test))])
        mae = np.array([mean_absolute_error(y_test, var_sec.predict(X=X_test))])
        out_fsam = pd.DataFrame(
            data=[mse, mae, [var_sec.sol["z"]], ["FSAM"]], index=cols
        ).T

        # Convert python arrays into r objects
        with localconverter(robjects.default_converter + numpy2ri.converter):
            X = robjects.conversion.py2rpy(X)
            x_te = robjects.conversion.py2rpy(X_test)
            y = robjects.conversion.py2rpy(y)
            y_te = robjects.conversion.py2rpy(y_test)

        print("\nWorking with RELGAM ")
        out_relgam = []
        for relgam_sel in [False, True]:
            out = get_r_results(
                script_path=Path("fsam_paper/R_scripts/relgam.R"),
                method_name="relgam_out",
                X=X,
                y=y,
                X_test=x_te,
                y_test=y_te,
                seed=seed,
                train_size=train_size,
                output_cols=cols,
                criterion=conf_model["criterion"],
                **{"relgam_sel": relgam_sel},
            )
            out_relgam.append(out)

        print("\nWorking with GAMSEL ")
        out_gamsel = get_r_results(
            script_path=Path("fsam_paper/R_scripts/Gamsel.R"),
            method_name="gamsel_out",
            X=X,
            y=y,
            X_test=x_te,
            y_test=y_te,
            seed=seed,
            train_size=train_size,
            output_cols=cols,
            criterion=conf_model["criterion"],
        )

        print("\nWorking with SAM ")
        out_sam = get_r_results(
            script_path=Path("fsam_paper/R_scripts/SAM.R"),
            method_name="sam_out",
            X=X,
            y=y,
            X_test=x_te,
            y_test=y_te,
            seed=seed,
            train_size=train_size,
            output_cols=cols,
            criterion=conf_model["criterion"],
        )

        out_r = pd.concat([out_sam, out_gamsel] + out_relgam).reset_index(drop=True)

        # Solve the LASSO model and save the results
        print("\nWorking with LASSO ")
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
                list(itertools.chain(lasso_vars, [0] * n_vars)),
                "LASSO",
            ],
            index=cols,
        ).T

        df = pd.concat((out_fsam, out_r, out_lasso)).reset_index(drop=True)
        df.loc[:, "nzero_bool"] = df.loc[:, "nzero_bool"].apply(
            lambda s: list(eval(s)) if not isinstance(s, list) else s
        )
        df = df.assign(
            **{"n_iter": i, "Scenario": scenario, "SNR": snr, "num_vars": n_vars}
        )
        df_list.append(df)

    # Concat all the DataFrame (there was one by iteration) and save the CSV file
    pd.concat(df_list).to_csv(f"data/scenario_{scenario}_{n_vars}.csv", index=False)
    print("Finished")


if __name__ == "__main__":
    typer.run(main)
