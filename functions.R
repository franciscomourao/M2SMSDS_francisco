# Calcul de la vraie valeur
ace <- function(ddata, Y, X, Z) {
  probs <- ddata  |> 
    summarise(
      n = n(),
      PY_XZ = mean(.data[[Y]], na.rm = TRUE),
      .by = all_of(c(Z, X))
    ) |> 
    mutate(
      PZ = sum(n) / nrow(ddata),
      .by = all_of(Z)
    )
  
  PY_do_x <- probs  |> 
    summarise(
      py_do_x = sum(PY_XZ * PZ, na.rm = TRUE),
      .by = all_of(X)
    )
  
  PY_do_x[PY_do_x[[X]] == 1, ][["py_do_x"]] - PY_do_x[PY_do_x[[X]] == 0, ][["py_do_x"]]
}

# Simulation
simulate_yzx <- function(df, X, Y, Z, n_observations) {
  
  ## On simule les z par rééchantillonnage
  df_z <- df |> 
    dplyr::select({{ Z }}) |> 
    dplyr::slice_sample(n = n_observations, replace = T)
  
  ## On simule les x par rééchantillonnage dans chaque strate de z
  df_zx <- df |> 
    dplyr::select({{ Z }}, {{ X }}) |> 
    dplyr::slice()
  
  z_count <- df_z |> 
    count(across(Z))
  
  for (i in 1:nrow(z_count)) {
    
    df_temp <- df |> 
      dplyr::select({{ Z }}, {{ X }})
    
    for (j in 1:length(Z)) {
      df_temp <- df_temp[df_temp[[Z[j]]] == z_count[[i, Z[j]]], ]
    }
    
    df_zx <- df_zx |> 
      bind_rows(
        df_temp |> 
          dplyr::slice_sample(n = z_count[[i, "n"]], replace = T)
      )
    
  }
  
  ## On simule y sachant x, z
  df_y_zx <- df |> 
    dplyr::select({{ Z }}, {{ X }}, {{ Y }}) |> 
    dplyr::slice()
  
  zx_count <- df_zx |> 
    count(across(c({{ Z }}, {{ X }})))
  
  for (i in 1:nrow(zx_count)) {
    
    df_temp <- df |> 
      dplyr::select({{ Z }}, {{ X }}, {{ Y }})
    
    for (j in 1:length(Z)) {
      df_temp <- df_temp[df_temp[[Z[j]]] == zx_count[[i, Z[j]]], ]
    }
    
    df_temp <- df_temp[df_temp[[X]] == zx_count[[i, X]], ]
    
    df_y_zx <- df_y_zx |> 
      bind_rows(
        df_temp |> 
          dplyr::slice_sample(n = zx_count[[i, "n"]], replace = T)
      )
    
  }
  
  df_y_zx
  
}


######

glm_Y_X <- function(ddata, X, Y, Z){
  
  mod_Y_X <- glm(formule_Y_X, data = ddata, family = binomial)
  
  comp <- avg_comparisons(mod_Y_X, variables = X, conf_level = 0.9, comparison = "difference")
  
  ate.mod_Y_X <- comp$estimate
  borne_inf <- comp$conf.low
  borne_sup <- comp$conf.high
  
  return(c(ate = ate.mod_Y_X, borne_inf = borne_inf, borne_sup = borne_sup))
  
}

# G-computation avec glm()
g_computation <- function(ddata, X, Y, Z){ 
  
  if (n_distinct(ddata[[Y]]) == 1) {
    return(0)
  } else {
  
  mod.glm <- glm(formule_Y_XZ, data = ddata, family = binomial)
  
  ate.glm <- predict(mod.glm, type = "response", newdata = ddata |> mutate("{X}" := 1)) |> mean() -
    predict(mod.glm, type = "response", newdata = ddata |> mutate("{X}" := 0)) |> mean()
  
  ate.glm
  
  }
}


# pondération inverse avec weight it 
ponderation_weightit <- function(ddata, X, Y, Z){ 
  
  W.ATE <- weightit(formule_X_Z,
                    data = ddata, 
                    method = "ps",
                    estimand = "ATE",
                    stabilize = FALSE
  )
  
  fit <- glm_weightit(formule_Y_X,
                      data = ddata,
                      weightit = W.ATE,
                      family = binomial)
  
  
  comp <- avg_comparisons(fit, variables = X, conf_level = 0.9, comparison = "difference")

  ate.iptw <- comp$estimate
  borne_inf <- comp$conf.low
  borne_sup <- comp$conf.high

  return(c(ate = ate.iptw, borne_inf = borne_inf, borne_sup = borne_sup))
  
}



# g_computation avec trees with boosting (xgboost model)

g_computation_xgb <- function(ddata, X, Y, Z) {
  
  if (n_distinct(ddata[[Y]]) == 1) {
    return(0)
  } else {
  
  fit <- boost_tree() |> 
    set_mode("classification") |> 
    set_engine("xgboost") |> 
    fit(formule_Y_XZ, data = ddata |> mutate("{Y}" := as.factor(ddata[[Y]])))
  
  predictions.0 <- predict(fit, new_data = ddata |> mutate("{X}" := 0), type = "prob") |> 
    pull(`.pred_1`) |> 
    mean()
  
  predictions.1 <- predict(fit, new_data = ddata |> mutate("{X}" := 1), type = "prob") |> 
    pull(`.pred_1`) |> 
    mean()
  
  predictions.1 - predictions.0
  }
}

# pondération inverse avec xgboost

ponderation_xgb <- function(ddata, X, Y, Z){ 
  
  if (n_distinct(ddata[[Y]]) == 1) {
    return(0)
  } else {
  
  fit <- boost_tree() |> 
    set_mode("classification") |> 
    set_engine("xgboost") |> 
    fit(formule_X_Z, data = ddata |> mutate("{X}" := as.factor(ddata[[X]])))
  
  xgb_ps <- predict(fit, new_data = ddata, type = "prob") |> 
    pull('.pred_1')
  
  ddata.ps <- ddata |>
    mutate(poids.xgb = if_else(ddata[[X]] == 1, 1/xgb_ps, 1/(1-xgb_ps)))
  
  fit.1 <- glm(formule_Y_X, 
               weights = poids.xgb, 
               data = ddata.ps |> mutate("{Y}" := as.factor(ddata.ps[[Y]])), 
               family = binomial)
  
  plogis(fit.1$coefficients[[1]] + fit.1$coefficients[[2]]) - plogis(fit.1$coefficients[[1]])
  }
}







ponderation_aipw <- function(ddata, X, Y, Z){ 
  set.seed(1234)
  
  AIPW_SL <- AIPW$new(Y = ddata[[ Y ]],
                      A = ddata[[ X ]],
                      W = ddata |> select(all_of(Z)), 
                      Q.SL.library = c("SL.mean","SL.glm"),
                      g.SL.library = c("SL.mean","SL.glm"),
                      verbose=FALSE)$
    fit()$
    summary()
  
  ate.aipw <- AIPW_SL$result[3,1]
  
  borne_inf.aipw <- AIPW_SL$result[3,1] - qnorm(1 - ((1 - 0.9) / 2)) * AIPW_SL$result[3,2]
  borne_sup.aipw <- AIPW_SL$result[3,1] + qnorm(1 - ((1 - 0.9) / 2)) * AIPW_SL$result[3,2]
  
  return(c(ate.aipw = ate.aipw, borne_inf.aipw = borne_inf.aipw, borne_sup.aipw = borne_sup.aipw))
  
}



# Estimation --------------------------------------------------------------

estimation <- function(ddata, n_boot) {
  
  start.boot <- tic()
  boot <- map(1:n_boot, ~ slice_sample(ddata, n = nrow(ddata), replace = TRUE))
  stop.boot <- toc(quiet = T)
  temps.boot <- stop.boot$toc - stop.boot$tic
  
  start.mod_Y_X <- tic()
  mod_Y_X <- glm_Y_X(ddata,
                     X = X,
                     Y = Y,
                     Z = Z)
  
  mod_Y_X_est <- mod_Y_X["ate"]
  mod_Y_X_borne_inf <- mod_Y_X["borne_inf"]
  mod_Y_X_borne_sup <- mod_Y_X["borne_sup"]
  
  
  stop.mod_Y_X <- toc(quiet = T)
  temps.mod_Y_X <- stop.mod_Y_X$toc - stop.mod_Y_X$tic
  
  
  
  start.g_computation <- tic()
  g_computation_est <- g_computation(ddata,
                                     X = X,
                                     Y = Y,
                                     Z = Z)
  
  g_computation_res <- map_dbl(boot,
                               ~ {g_computation(.x,
                                                X = X,
                                                Y = Y,
                                                Z = Z)}) 
  
  g_computation_borne_inf <- quantile(g_computation_res, 0.05)
  g_computation_borne_sup <- quantile(g_computation_res, 0.95)
  stop.g_computation <- toc(quiet = T)
  temps.g_computation <- stop.g_computation$toc - stop.g_computation$tic + temps.boot
  
  # pondération inverse avec weightit
  start.weighit <- tic()
  ponderation_weightit_res <- ponderation_weightit(ddata,
                                                   X = X,
                                                   Y = Y,
                                                   Z = Z)
  
  ponderation_weightit_est <- ponderation_weightit_res["ate"]
  ponderation_weightit_borne_inf <- ponderation_weightit_res["borne_inf"]
  ponderation_weightit_borne_sup <- ponderation_weightit_res["borne_sup"]
  stop.weighit <- toc(quiet = T)
  temps.weighit <- stop.weighit$toc - stop.weighit$tic
  
  #xgboost
  start.xgb <- tic()
  xgb_est <- g_computation_xgb(ddata,
                               X = X,
                               Y = Y,
                               Z = Z)
  
  xgb_res <- map_dbl(boot,
                     ~ {g_computation_xgb(.x,
                                          X = X,
                                          Y = Y,
                                          Z = Z)}) 
  xgb_borne_inf <- quantile(xgb_res, 0.05)
  xgb_borne_sup <- quantile(xgb_res, 0.95)
  stop.xgb <- toc(quiet = T)
  temps.xgb <- stop.xgb$toc - stop.xgb$tic + temps.boot
  
  start.iptw.xgb <- tic()
  iptw.xgb_est <- ponderation_xgb(ddata,
                                  X = X,
                                  Y = Y,
                                  Z = Z)
  
  iptw.xgb_res <- map_dbl(boot,
                          ~ {ponderation_xgb(.x,
                                             X = X,
                                             Y = Y,
                                             Z = Z)}) 
  iptw.xgb_borne_inf <- quantile(iptw.xgb_res, 0.05)
  iptw.xgb_borne_sup <- quantile(iptw.xgb_res, 0.95)
  stop.iptw.xgb <- toc(quiet = T)
  temps.iptw.xgb <- stop.iptw.xgb$toc - stop.iptw.xgb$tic + temps.boot
  
  start.aipw <- tic()
  library(SuperLearner)
  ponderation_aipw_res <- ponderation_aipw(ddata,
                                                   X = X,
                                                   Y = Y,
                                                   Z = Z)
  
  ponderation_aipw_est <- ponderation_aipw_res["ate.aipw"]
  ponderation_aipw_borne_inf <- ponderation_aipw_res["borne_inf.aipw"]
  ponderation_aipw_borne_sup <- ponderation_aipw_res["borne_sup.aipw"]
  stop.aipw <- toc(quiet = T)
  temps.aipw <- stop.aipw$toc - stop.aipw$tic
  
  res <- tibble(
    methode = c("glm_Y_X", "g_computation_glm", "iptw.weightit", "g_computation_xgb", "iptw.xgb", "aipw"),
    est = c(mod_Y_X_est, g_computation_est, ponderation_weightit_est , xgb_est, iptw.xgb_est, ponderation_aipw_est),
    borne_inf = c(mod_Y_X_borne_inf, g_computation_borne_inf, ponderation_weightit_borne_inf, xgb_borne_inf, iptw.xgb_borne_inf, ponderation_aipw_borne_inf),
    borne_sup = c(mod_Y_X_borne_sup, g_computation_borne_sup, ponderation_weightit_borne_sup, xgb_borne_sup, iptw.xgb_borne_sup, ponderation_aipw_borne_sup),
    temps_calcul = c(temps.mod_Y_X, temps.g_computation, temps.weighit, temps.xgb, temps.iptw.xgb, temps.aipw)
  )
  
  return(res)
}


