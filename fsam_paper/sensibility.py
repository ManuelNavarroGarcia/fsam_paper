#!/usr/bin/env python
import time

import numpy as np
import pandas as pd
import typer
from fsam.fsam_fit import FSAM
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from fsam_paper.utils import SCENARIO_DEFINITION, get_y


def main(
    deg: int = 3,
    ord_d: int = 2,
    n_int: int = 20,
    scenario: int = 1,
    seed: int = 0,
    n_iter: int = 50,
    snr: int = 4,
    n_vars: int = 200,
    n_samples: int = 400,
    qs: list[int] = [5, 10, 20, 30],
    n_iter_no_change: list[int] = [10, 20, 30, 40],
):
    """Conducts a sensitivity analysis on the parameters `q` (destroy size) and
    `n_iter_no_change` (patience)

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
        The signal-to-noise ratio parameter, by default 4
    n_vars : int, optional
        The number of features, by default 200
    n_samples : int, optional
        The number of available features, both for training and validation. By default
        400
    qs : list[int], optional
        The candidates for the destroy size parameter. By default [5, 10, 20, 30]
    n_iter_no_change : list[int], optional
        The candidates for the patience parameter. By default [10, 20, 30, 40]
    """
    df_list = []

    conf_gurobi = {
        "OutputFlag": 0,
        "threads": 1,
        "timelimit": 3600,
        "MIPFocus": 0,
        "MIPGap": 1e-4,
    }
    cols = ["MSE", "MAE", "Time", "nzero_bool", "Method"]

    true_feat_lin = SCENARIO_DEFINITION[scenario]["true_feat_lin"]
    true_feat_nonlin = SCENARIO_DEFINITION[scenario]["true_feat_nonlin"]
    K = [true_feat_lin + true_feat_nonlin]
    u_min = SCENARIO_DEFINITION[scenario]["u_min"]
    u_max = SCENARIO_DEFINITION[scenario]["u_max"]

    n_test = 5000
    train_size = 0.8

    for i in tqdm(range(n_iter), desc="Iterations: ", leave=True, colour="CYAN"):
        for q in qs:
            for n_i in n_iter_no_change:
                conf_model = {
                    "criterion": "mae",
                    "q": q,
                    "max_time": 3600,
                    "max_iter": 1000000,
                    "n_iter_no_change": n_i,
                    "n_iter_pgl": 2000,
                    "eps": 1e-2,
                    "tol_pgl": 1e-3,
                    "n_alphas": 50,
                    "patience": 10,
                    "min_edf": 1,
                }
                np.random.seed(seed + i)

                # Generate from Uniform (`u_min`, `u_max`) the train and test sets
                X_sim = np.random.uniform(
                    low=u_min, high=u_max, size=(n_samples + n_test, n_vars)
                )
                x_min = X_sim.min(axis=0)
                x_max = X_sim.max(axis=0)
                X_sim = np.concatenate(
                    [
                        np.random.uniform(
                            low=xmi, high=xma, size=(n_samples + n_test - 2, 1)
                        )
                        for xmi, xma in zip(x_min, x_max)
                    ],
                    axis=1,
                )
                X_sim = np.concatenate(
                    [
                        np.expand_dims(x_max, axis=0),
                        np.expand_dims(x_min, axis=0),
                        X_sim,
                    ]
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
                    X_train,
                    y_train,
                    train_size=train_size,
                    shuffle=False,
                    random_state=seed,
                )
                assert all(X_val.min(axis=0) > X_train.min(axis=0))
                assert all(X_val.max(axis=0) < X_train.max(axis=0))

                X = np.concatenate([X_train, X_val], axis=0)
                y = np.concatenate([y_train, y_val], axis=0)

                # Solve our model and save the results
                start = time.time()
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
                end = time.time()

                mse = np.array([mean_squared_error(y_test, var_sec.predict(X=X_test))])
                mae = np.array([mean_absolute_error(y_test, var_sec.predict(X=X_test))])
                out_fsam = pd.DataFrame(
                    data=[
                        mse,
                        mae,
                        np.array([end - start]),
                        [var_sec.sol["z"]],
                        ["FSAM"],
                    ],
                    index=cols,
                ).T

                df = pd.concat((out_fsam,)).reset_index(drop=True)
                df.loc[:, "nzero_bool"] = df.loc[:, "nzero_bool"].apply(
                    lambda s: list(eval(s)) if not isinstance(s, list) else s
                )
                df = df.assign(
                    **{
                        "n_iter": i,
                        "Scenario": scenario,
                        "SNR": snr,
                        "num_vars": n_vars,
                        "q": q,
                        "n_iter_no_change": n_i,
                    }
                )
                df_list.append(df)

    # Concat all the DataFrame (there was one by iteration) and save the CSV file
    pd.concat(df_list).to_csv(f"data/sensibility_{scenario}.csv", index=False)
    print("Finished")


if __name__ == "__main__":
    typer.run(main)
