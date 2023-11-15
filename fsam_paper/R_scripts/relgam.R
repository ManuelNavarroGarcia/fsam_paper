source("fsam_paper/R_scripts/R_utils.R")
improved_install_packages(c("relgam","gensvm"))

relgam_out <- function(X, y, X_test, y_test, seed, train_size, output_cols, criterion, ...){
  set.seed(seed)

  dim(y) <- c(length(y), 1)
  args = list(...)
  relgam_sel = args$relgam_sel
  if (relgam_sel){
    init_nz <- c()
    method <- "RELGAM_SEL"
  } else { # Internally, if you do not pass init_nz, the code assign it to 1:p
    init_nz <- 1:ncol(X)
    method <- "RELGAM"
  }

  # Split into train and validation sets
  split = gensvm.train.test.split(X, y, train.size = train_size, shuffle = FALSE)
  X_train = split$x.train
  X_val = split$x.test
  y_train = split$y.train
  y_val = split$y.test

  # `foldid` is an argument of `rgam` that identifies each crossvalidation fold 
  # of the lasso fit with each observation. If `foldid` is not supplied then it 
  # is chosen by a random shuffle. For reproducibility reasons, we need to set 
  # it by a fixed shuffle of the indices.
  # `cv` = 5 is the default number of folds in crossvalidations in `relgam`
  CV = 5
  foldid = sort(rep(seq(CV), length = nrow(X_train)))
  start_time <- Sys.time()
  # Given data matrix `X` and target data `y`, fit RELGAM.
  relgamfit = rgam(X_train, y_train, init_nz = init_nz,
                  foldid = foldid, verbose = FALSE)
  end_time <- Sys.time()
  times <- difftime(end_time, start_time, units = "secs")

  # Predict over `X_val` and get best `lambda` minimizing `criterion`
  y_pred_val <- predict(relgamfit, xnew = X_val)
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
  y_pred_test <- predict(relgamfit, xnew = X_test, s = relgamfit$lambda[opt_lambda_idx])
  mse <- mean((y_test - as.vector(y_pred_test))^2)
  mae <- mean(abs(y_test - as.vector(y_pred_test)))
  frac_null <- mse / mean((rep(mean(y_train), length(y_test)) - y_test)^2) 

  nzero_lin <- relgamfit$nzero_lin[opt_lambda_idx]
  nzero_nonlin <- relgamfit$nzero_nonlin[opt_lambda_idx]

  nzero_lin_bool = replace(rep(0, dim(X_train)[2]), unlist(relgamfit$linfeat[opt_lambda_idx]), 1)
  nzero_nonlin_bool = replace(rep(0, dim(X_train)[2]), unlist(relgamfit$nonlinfeat[opt_lambda_idx]), 1)
  nzero_bool = c(nzero_lin_bool, nzero_nonlin_bool)

  data <- t(data.frame(c(mse, 
                         mae,
                         paste("[", toString(nzero_bool), "]"),
                         method)))
  colnames(data) <- unlist(output_cols)
  return(as.data.frame(data))
}
