source("fsam_paper/R_scripts/R_utils.R")
improved_install_packages(c("gamsel", "gensvm"))

gamsel_out <- function(X, y, X_test, y_test, seed, train_size, output_cols, criterion, ...) {
  set.seed(seed)

  dim(y) <- c(length(y), 1)
  # Split into train and validation sets
  split <- gensvm.train.test.split(X, y, train.size = train_size, shuffle = FALSE)
  X_train <- split$x.train
  X_val <- split$x.test
  y_train <- split$y.train
  y_val <- split$y.test

  start_time <- Sys.time()
  # Given data matrix `X` and target data `y`, fit GAMSEL
  gamselfit <- gamsel(X_train, y_train)
  end_time <- Sys.time()
  times <- difftime(end_time, start_time, units = "secs")

  # Predict over `X_val` and get best `lambda` minimizing `criterion`
  y_pred_val <- predict(gamselfit, newdata = X_val, type = "response")
  if (criterion == "mse") {
    val_error <- sapply(1:dim(y_pred_val)[2], function(i) mean((y_pred_val[, i] - y_val)^2))
  } else if (criterion == "mae") {
    val_error <- sapply(1:dim(y_pred_val)[2], function(i) mean(abs(y_pred_val[, i] - y_val)))
  } else {
    stop("Criterion not chosen correctly.")
  }
  # Get best lambda
  opt_lambda_idx <- which.min(val_error)

  # Predict over `X_test` and get metrics
  y_pred_test <- predict(gamselfit, newdata = X_test, type = "response")[, opt_lambda_idx]
  mse <- mean((y_test - as.vector(y_pred_test))^2)
  mae <- mean(abs(y_test - as.vector(y_pred_test)))
  frac_null <- mse / mean((rep(mean(y_train), length(y_test)) - y_test)^2)

  # Get the number of linear and nonlinear features (In summarynz: 1 -> Linear,
  # 2 -> Nonlinear, 3 -> Nonzero)
  nzero_lin <- gamsel:::summarynz(gamselfit)[opt_lambda_idx, 1]
  nzero_nonlin <- gamsel:::summarynz(gamselfit)[opt_lambda_idx, 2]
  alpha_selected <- which(gamselfit$alphas[, opt_lambda_idx] != 0)
  betas_selected <- gamselfit$betas[, opt_lambda_idx]

  # gamselfit$betas returns a betas_selected object sized sum(`degrees`) *
  # `num_lambda`( 50, by default).
  # As we sometimes need our degrees to be not constant, this is the way we fix
  # the access to `beta_selected`
  deg <- cumsum(c(0, gamselfit$degrees))
  beta_selected <- which(sapply(1:dim(X_train)[2], function(i) any(betas_selected[deg[i]:deg[i + 1]] != 0)))

  nzero_lin_bool <- replace(rep(0, dim(X_train)[2]), alpha_selected, 1)
  nzero_nonlin_bool <- replace(rep(0, dim(X_train)[2]), beta_selected, 1)
  nzero_bool <- c(nzero_lin_bool, nzero_nonlin_bool)

  data <- t(data.frame(c(
    mse,
    mae,
    paste("[", toString(nzero_bool), "]"),
    "GAMSEL"
  )))
  colnames(data) <- unlist(output_cols)
  return(as.data.frame(data))
}
