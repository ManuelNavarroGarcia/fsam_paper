# fsam_paper

`fsam_paper` is a GitHub repository containing all the tables and simulations results
shown in the paper:

All the simulation studies carried out in this work use the routines implemented in
[fsam](https://github.com/ManuelNavarroGarcia/fsam), which requires a
[GUROBI](https://www.gurobi.com) license to solve the optimization problems.

## Project structure

The current version of the project is structured as follows:

* **fsam_paper**: the main directory of the project, which
  consist of:
  * **R_scripts**: contains the code to the other state-of-the-art algorithms used to
    compare our approach.
  * **fsam_comparison.py**: the code used in the comparison of FSAM with the other
    state-of-the-art methodologies (Section 5.1.2).
  * **fsam_ws_performance.py**: the code used in the comparison of the MIQP
    formulation and FSAM used as warm start (Section 5.1.1).
  * **matheuristic_performance.py**: the code used in the comparison of the MIQP
    formulation and FSAM (Section 5.1.1).
  * **matheuristic_performance.py**: the sensibility analysis performed on the destroy
    size and the patience parameters (Supplementary Material).
  * **small_datasets.py**: the code used in the real-world data sets (Section 5.2.1).
  * **superconductors.py**: the code used in the real data set application of
    superconductivity data (Section 5.2.2).
  * **tables.ipynb**: A Jupyter notebook containing the code used to generate the tables
    and the figures of the paper.
  * **utils.py**: Auxiliary code for the rest of the scripts.
* **data**: a folder containing CSV and parquet files with simulated and real data sets
  results.

## Package dependencies

`fsam_paper` mainly depends on the following packages:

* [fsam](https://pypi.org/project/fsam/).
* [pyarrow](https://pypi.org/project/pyarrow/)
* [rpy2](https://pypi.org/project/rpy2/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [typer](https://pypi.org/project/typer/)

## Contact Information and Citation

If you have encountered any problem or doubt while using `fsam`, please feel free to let
me know by sending me an email:

* Name: Manuel Navarro García (he/his)
* Email: <manuelnavarrogithub@gmail.com>

If you find `fsam` or `fsam_paper` useful, please cite it in your publications.
