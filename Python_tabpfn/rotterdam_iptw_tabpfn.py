import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed
from tabpfn import TabPFNClassifier
from sklearn.linear_model import LogisticRegression


input_file = sys.argv[1]
print(f"Loading input file: {input_file}")
simulations = pd.read_csv(input_file)

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def iptw_tabpfn(tmp_ddata, n_sim_value, n_boot):
    boot_results = []
    start = time.time()

    Y = tmp_ddata['chemo']
    X = tmp_ddata.drop(columns=['chemo', 'mort_recur_5_an', 'n_sim'])

    model = TabPFNClassifier()
    model.fit(X, Y)

    tmp_ddata['ps'] = model.predict_proba(X)[:, 1]
    tmp_ddata['poids'] = np.where(tmp_ddata['chemo'] == 1, 1 / tmp_ddata['ps'],
                1 / (1 - tmp_ddata['ps']))
    
    logreg = LogisticRegression()
    logreg.fit(X=tmp_ddata['chemo'].values.reshape(-1, 1),
                y=tmp_ddata['mort_recur_5_an'],
                sample_weight=tmp_ddata['poids'])
        
    est = sigmoid(logreg.intercept_[0] + logreg.coef_[0][0]) - sigmoid(logreg.intercept_[0])

    for _ in range(n_boot):
        df_bootstrap_sample = tmp_ddata.sample(n=len(tmp_ddata), replace=True)

        if df_bootstrap_sample['mort_recur_5_an'].nunique() == 1:
            ate = 0

        else:
        
            Y = df_bootstrap_sample['chemo']
            X = df_bootstrap_sample.drop(columns=['chemo', 'mort_recur_5_an', 'n_sim'])
    
            model = TabPFNClassifier()
            model.fit(X, Y)
    
            df_bootstrap_sample['ps'] = model.predict_proba(X)[:, 1]
            df_bootstrap_sample['poids'] = np.where(
                df_bootstrap_sample['chemo'] == 1,
                1 / df_bootstrap_sample['ps'],
                1 / (1 - df_bootstrap_sample['ps'])
            )
    
            logreg = LogisticRegression()
            logreg.fit(
                X=df_bootstrap_sample['chemo'].values.reshape(-1, 1),
                y=df_bootstrap_sample['mort_recur_5_an'],
                sample_weight=df_bootstrap_sample['poids']
            )
            
            ate = sigmoid(logreg.intercept_[0] + logreg.coef_[0][0]) - sigmoid(logreg.intercept_[0])

        boot_results.append(ate)

    boot_results_series = pd.Series(boot_results)
    borne_inf = boot_results_series.quantile(0.05)
    borne_sup = boot_results_series.quantile(0.95)

    end = time.time()
    temps_calcul = end - start

    return {
        'n_sim': n_sim_value,
        'est': est,
        'borne_inf': borne_inf,
        'borne_sup': borne_sup,
        'temps_calcul': temps_calcul
    }

def iptw_tabpfn_parallel(ddata, n_boot, n_jobs):
    unique_n_sim = ddata['n_sim'].unique()
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(iptw_tabpfn)(ddata.loc[ddata['n_sim'] == i].copy(), i, n_boot)
        for i in unique_n_sim
    )
    return pd.DataFrame(results)

start = time.time()
df_results_iptw = iptw_tabpfn_parallel(simulations, n_boot=200, n_jobs=64)
end = time.time()
print(df_results_iptw)
print(f"Total computation time: {end - start:.2f} seconds")

base_name = os.path.splitext(os.path.basename(input_file))[0]

n_sim_min = df_results_iptw['n_sim'].min()
n_sim_max = df_results_iptw['n_sim'].max()

output_filename = f"iptw_results_{base_name}_nsim_{n_sim_min}_to_{n_sim_max}.csv"
df_results_iptw.to_csv(output_filename, index=False)
print(f"Saved results to: {output_filename}")




