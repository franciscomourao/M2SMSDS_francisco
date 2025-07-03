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

library(survival)

rotterdam
# X : chemo (chemotherapy) or hormon (hormonal treatment)
# Y : rtime (days to relapse or last follow-up) or recur () or dtime (days to death or last follow-up), death ()
# Z : year (year of surgery), age (age at surgery), meno (menopausal status), size (tumor size), grade (differentiation grade)
#     nodes (number of positive lymph nodes), pgr (progesterone receptors), er (estrogen receptors)

# The original dataset comprised 2982 primary breast cancer patients whose records were included in the Rotterdam tumour bank, of whom 1546 had node-positive disease.

rotterdam.1 <- rotterdam |> 
  mutate(mort_recur_5_an = ifelse((dtime<= 365*5 & (death == 1)) | (rtime<= 365*5 & (recur == 1)), 1, 0),
         year_bin = ntile(year, 2) |> as.factor(),
         age_bin = ntile(age, 2) |> as.factor(),
         nodes_bin = ifelse(nodes > 1, 1, 0) |> as.factor(),
         meno = as.factor(meno),
         size = ifelse(size == "<=20", 0, 1) |> as.factor(),
         grade = as.factor(grade)
         ) |> 
  filter(!(dtime <= 365*5 & (death == 0) & (rtime <= 365*5 & (recur == 0)))) |>
  select(chemo,
         mort_recur_5_an,
         year_bin,
         age_bin,
         nodes_bin,
         meno,
         size, 
         grade)

rotterdam.2 <- rotterdam.1 |> 
  left_join(
    rotterdam.1 |> 
      count(chemo,
            year_bin,
            age_bin,
            nodes_bin,
            meno,
            size, 
            grade
            ) |> 
      complete(chemo,
               year_bin,
               age_bin,
               nodes_bin,
               meno,
               size, 
               grade,
               fill = list(n = 0)) |> 
      filter(n == 0) |> 
      select(-chemo)
  ) |> 
  filter(is.na(n)) |> 
  select(-n)

# Définitions -------------------------------------------------------------

X = "chemo"
Y = "mort_recur_5_an"
Z = c("year_bin",
      "age_bin",
      "nodes_bin",
      "meno",
      "size", 
      "grade")

formule_Y_XZ <- as.formula(paste(Y, "~", paste(c(X, Z), collapse = " + ")))
formule_X_Z <- as.formula(paste(X, "~", paste(Z, collapse = " + ")))
formule_Y_X <- as.formula(paste(Y, "~", X))

# True causal effect ------------------------------------------------------

ace.res <- ace(rotterdam.2,
               X = X, 
               Y = Y, 
               Z = Z)

# Simulation --------------------------------------------------------------

rotterdam_1000sim_200obs <- map(1:1000, ~ simulate_yzx(df = rotterdam.2,
                                           X = X,
                                           Y = Y,
                                           Z = Z,
                                           n_observations = 200)) |>
  list_rbind(names_to = "n_sim")

rotterdam_1000sim_500obs <- map(1:1000, ~ simulate_yzx(df = rotterdam.2,
                                           X = X,
                                           Y = Y,
                                           Z = Z,
                                           n_observations = 500)) |>
  list_rbind(names_to = "n_sim")








