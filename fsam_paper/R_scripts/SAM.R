source("fsam_paper/R_scripts/R_utils.R")
improved_install_packages(c("SAM","gensvm"))

sam_out <- function(X, y, X_test, y_test, seed, train_size, output_cols, criterion, ...){
  set.seed(seed)

  dim(y) <- c(length(y), 1)
  # Split into train and validation sets
  split = gensvm.train.test.split(X, y, train.size = train_size, shuffle = FALSE)
  X_train = split$x.train
  X_val = split$x.test
  y_train = split$y.train
  y_val = split$y.test

  # Given data matrix `X` and target data `y`, fit samQL
  start_time <- Sys.time()
  samfit <- samQL(X_train, y_train)
  end_time <- Sys.time()
  times <- difftime(end_time, start_time, units = "secs")

  # Predict over `X_val` and get best `lambda` minimizing `criterion`
  y_pred_val <- predict(samfit, X_val)$values
  if (criterion == "mse"){
    val_error <- sapply(1:dim(y_pred_val)[2], function(i) mean((y_pred_val[, i] - y_val)^2))
  }
  else if (criterion == "mae"){
    val_error = sapply(1:dim(y_pred_val)[2], function(i) mean(abs(y_pred_val[, i] - y_val)))
  }
  else {
    stop("Criterion not chosen correctly.") 
  }
  # Get best lambda
  opt_lambda_idx = which.min(val_error)

  # Predict over `X_test` and get metrics
  y_pred_test <- predict(samfit, X_test)$values[, opt_lambda_idx]
  mse = mean((y_test - y_pred_test)^2)
  mae = mean(abs(y_test - y_pred_test))
  frac_null <- mse / mean((rep(mean(y_train), length(y_test)) - y_test)^2)

  # get coefficients
  coefs <- samfit$w[, opt_lambda_idx]

  # number of nonzero features
  feat_selected <- which(sapply(1:dim(X_train)[2], function(i)
    any(coefs[(i - 1) * samfit$p + 1:samfit$p] != 0)))

  nzero_lin = 0 #nzero_lin is 0 because the method itself is full nonlinear.
  nzero_nonlin = length(feat_selected)

  nzero_lin_bool = rep(0, dim(X_train)[2])
  nzero_nonlin_bool = replace(rep(0, dim(X_train)[2]), feat_selected, 1)

  nzero_bool = c(nzero_lin_bool,nzero_nonlin_bool)

  data <- t(data.frame(c(mse,
                         mae,
                         paste("[", toString(nzero_bool), "]"),
                         "SAM")))
  colnames(data) <- unlist(output_cols)
  return(as.data.frame(data))
}


