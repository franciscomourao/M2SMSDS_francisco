# Librairies utilisées ----------------------------------------------------
library(WeightIt)
library(survey)
library(furrr)
library(purrr)
library(tictoc)
library(tidyverse)
library(randomForest)
library(medicaldata)
library(xgboost)
library(fastDummies)
library(tidymodels)
library(TabPFN)
library(bcaboot)
library(tmle)
library(AIPW)
library(SuperLearner)
library(marginaleffects)
library(readr)

# Préparation des données -------------------------------------------------

donnee <- indo_rct |>
  filter(site != "4_Case", site != "3_UK") |>
  select(site, age, gender, sod, psphinc, outcome) |> 
  mutate(across(everything(), ~ substr(., 1, 1))) |> 
  mutate(
    site = as.factor(fct_drop(site)),
    age = as.factor(ntile(age, 2)),
    gender = as.factor(gender),
    sod = as.factor(sod),
    psphinc = as.integer(psphinc),
    outcome = as.integer(outcome)
  )

donnee <- donnee |> 
  left_join(
      donnee |> 
      count(site, age, gender, sod, psphinc) |> 
      complete(site, age, gender, sod, psphinc, fill = list(n = 0)) |> 
      filter(n == 0) |> 
      select(-psphinc)
  ) |> 
  filter(is.na(n)) |> 
  select(-n)


# Définitions -------------------------------------------------------------

X = "psphinc"
Y = "outcome"
Z = c("age", "gender", "site", "sod")

formule_Y_XZ <- as.formula(paste(Y, "~", paste(c(X, Z), collapse = " + ")))
formule_X_Z <- as.formula(paste(X, "~", paste(Z, collapse = " + ")))
formule_Y_X <- as.formula(paste(Y, "~", X))


# True causal effect ------------------------------------------------------

# using the data before the simulations, considered our population in this case

ace.res <- ace(donnee,
    X = X, 
    Y = Y, 
    Z = Z)

# Simulation --------------------------------------------------------------

indo_1000sim_200obs <- map(1:1000, ~ simulate_yzx(df = donnee,
                                            X = X,
                                            Y = Y,
                                            Z = Z,
                                            n_observations = 200)) |>
   list_rbind(names_to = "n_sim")


indo_1000sim_500obs <- map(1:1000, ~ simulate_yzx(df = donnee,
                                            X = X,
                                            Y = Y,
                                            Z = Z,
                                            n_observations = 500)) |>
   list_rbind(names_to = "n_sim")
