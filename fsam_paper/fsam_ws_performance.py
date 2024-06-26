#!/usr/bin/env python

import numpy as np
import pandas as pd
import typer
from fsam.fsam_fit import FSAM
from tqdm.auto import tqdm

from fsam_paper.utils import SCENARIO_DEFINITION, get_y


def main(
    deg: int = 3,
    ord_d: int = 2,
    n_int: int = 20,
    seed: int = 0,
    n_iter: int = 50,
    scenario: int = 1,
    n_cols: list[int] = [40],
    n_rows: list[int] = [100, 200, 400],
):
    """Solve the variable selection problem using the result of FSAM as warm start.

    The sparsity parameter `K` is the true sparsity parameter of the simulation scenario
    and the number of features is set to 40 in order to solve it in a reasonable time.
    The CPU Gurobi time, optimal objective function value and the GAP are saved into a
    CSV file.

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
        The random seed of the experiments, by default 0
    n_iter : int, optional
        The number of iterations, by default 50
    scenario : int, optional
        The ID number of scenario, by default 1
    n_cols : list[int], optional
        The number of variables in the data matrix, by default [40]
    n_rows : list[int], optional
        The number of rows in the data matrix, by default [100, 200, 400]
    """
    K = (
        SCENARIO_DEFINITION[scenario]["true_feat_lin"]
        + SCENARIO_DEFINITION[scenario]["true_feat_nonlin"]
    )
    np.random.seed(seed)

    conf_gurobi = {
        "OutputFlag": 0,
        "threads": 1,
        "timelimit": 3600,
        "MIPFocus": 0,
        "MIPGap": 1e-4,
    }
    conf_model = {
        "criterion": "mae",
        "q": 30,
        "max_time": 3600,
        "max_iter": 1000000,
        "n_iter_no_change": 20,
        "n_iter_pgl": 2000,
        "eps": 1e-2,
        "tol_pgl": 1e-3,
        "n_alphas": 50,
        "patience": 10,
        "min_edf": 1,
    }

    conf_gurobi_ = conf_gurobi.copy()
    conf_gurobi_.update({"OutputFlag": 1, "timelimit": 3600})

    true_feat_lin = SCENARIO_DEFINITION[scenario]["true_feat_lin"]
    true_feat_nonlin = SCENARIO_DEFINITION[scenario]["true_feat_nonlin"]
    u_min = SCENARIO_DEFINITION[scenario]["u_min"]
    u_max = SCENARIO_DEFINITION[scenario]["u_max"]
    SNR = 2

    df_lists = []
    output_cols = [
        "CPU_time_FSAM",
        "CPU_time_OPTIMALITY",
        "Obj_function_warm_start",
        "Obj_function",
        "GAP",
        "Bounds_coef",
        "Sparsity",
        "n_samples",
        "n_features",
    ]
    loop_iterations = n_iter * len(n_cols) * len(n_rows) * 2
    progress_bar = tqdm(range(loop_iterations), leave=False, colour="RED")
    for i in range(n_iter):
        np.random.seed(seed + i)
        # Generate the largest data matrix
        X = np.random.uniform(low=u_min, high=u_max, size=(max(n_rows), max(n_cols)))
        df_list = []

        # Iterate over number of samples
        for n_samples in n_rows:
            # Iterate over number of features
            for n_vars in n_cols:
                for compute_coef_bounds in [True, False]:
                    # Generate train and test response variable, where the first is
                    # perturbed with Gaussian noise for given SNR
                    conf_model_ = conf_model.copy()
                    conf_model_.update({"q": 2 * n_vars, "max_iter": 1, "min_edf": 70})
                    y = get_y(
                        X=X[:n_samples, :n_vars],
                        true_feat_lin=true_feat_lin,
                        true_feat_nonlin=true_feat_nonlin,
                        scenario=scenario,
                    )
                    error = np.random.normal(
                        loc=0, scale=y.std() / np.sqrt(SNR), size=n_samples
                    )
                    y += error
                    var_sec = FSAM(
                        deg=[deg] * n_vars,
                        ord_d=[ord_d] * n_vars,
                        n_int=[n_int] * n_vars,
                    )
                    var_sec.fit(
                        X=X[:n_samples, :n_vars],
                        y=y,
                        K=K,
                        train_size=None,
                        conf_gurobi=conf_gurobi,
                        conf_model=conf_model,
                        warm_start=True,
                        scale_y=False,
                        compute_coef_bounds=compute_coef_bounds,
                        frac_row_bounds=0.0,
                    )
                    var_sec_ = FSAM(
                        deg=[deg] * n_vars,
                        ord_d=[ord_d] * n_vars,
                        n_int=[n_int] * n_vars,
                    )
                    var_sec_.fit(
                        X=X[:n_samples, :n_vars],
                        y=y,
                        K=K,
                        train_size=None,
                        conf_gurobi=conf_gurobi_,
                        conf_model=conf_model_,
                        warm_start=var_sec.init_sol,
                        scale_y=False,
                        compute_coef_bounds=compute_coef_bounds,
                        frac_row_bounds=0.0,
                    )
                    df_out = pd.DataFrame(
                        data=[
                            var_sec.obj_evolution[K].Times.sum(),
                            var_sec_.obj_evolution[K].Times.sum(),
                            var_sec.init_sol["obj"],
                            var_sec_.obj_evolution[K].Best_objective.min(),
                            var_sec_.model.MIPGap,
                            compute_coef_bounds,
                            K,
                            n_samples,
                            n_vars,
                        ],
                        index=output_cols,
                    ).T
                    df_list.append(df_out)
                    progress_bar.update()
        df_lists.append(pd.concat(df_list).assign(**{"seed": i}))
    df = pd.concat(df_lists).assign(**{"scenario": scenario})
    df.to_csv(f"data/matheuristic_ws_{scenario}.csv", index=False)


if __name__ == "__main__":
    typer.run(main)
