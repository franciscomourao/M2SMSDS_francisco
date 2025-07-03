import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys
import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed
from tabpfn import TabPFNClassifier


input_file = sys.argv[1]
print(f"Loading input file: {input_file}")
simulations = pd.read_csv(input_file)

def g_computation_tabpfn(tmp_ddata, n_sim_value, n_boot):
    boot_results = []
    start = time.time()

    Y = tmp_ddata['outcome']
    X = tmp_ddata.drop(columns=['outcome', 'n_sim'])

    model = TabPFNClassifier()
    model.fit(X, Y)

    X_0 = X.copy()
    X_0['psphinc'] = 0
    probabilities_0 = model.predict_proba(X_0)

    X_1 = X.copy()
    X_1['psphinc'] = 1
    probabilities_1 = model.predict_proba(X_1)

    est = np.mean(probabilities_1[:, 1]) - np.mean(probabilities_0[:, 1])

    for _ in range(n_boot):
        df_bootstrap_sample = tmp_ddata.sample(n=len(tmp_ddata), replace=True)

        if df_bootstrap_sample['outcome'].nunique() == 1:
            ate = 0
        
        else:
	
            Y = df_bootstrap_sample['outcome']
            X = df_bootstrap_sample.drop(columns=['outcome', 'n_sim'])

            model = TabPFNClassifier()
            model.fit(X, Y)

            X_0 = X.copy()
            X_0['psphinc'] = 0
            probabilities_0 = model.predict_proba(X_0)

            X_1 = X.copy()
            X_1['psphinc'] = 1
            probabilities_1 = model.predict_proba(X_1)

            ate = np.mean(probabilities_1[:, 1]) - np.mean(probabilities_0[:, 1])
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

def g_computation_tabpfn_parallel(ddata, n_boot, n_jobs):
    unique_n_sim = ddata['n_sim'].unique()
    results = Parallel(n_jobs=n_jobs, verbose = 10)(
        delayed(g_computation_tabpfn)(ddata.loc[ddata['n_sim'] == i].copy(), i, n_boot)
        for i in unique_n_sim
    )
    return pd.DataFrame(results)

start = time.time()
df_results = g_computation_tabpfn_parallel(simulations, n_boot=200, n_jobs=64)
end = time.time()
print(df_results)
print(f"Total computation time: {end - start:.2f} seconds")

base_name = os.path.splitext(os.path.basename(input_file))[0]

n_sim_min = df_results['n_sim'].min()
n_sim_max = df_results['n_sim'].max()

output_filename = f"g_computation_results_{base_name}_nsim_{n_sim_min}_to_{n_sim_max}.csv"
df_results.to_csv(output_filename, index=False)
print(f"Saved results to: {output_filename}")
