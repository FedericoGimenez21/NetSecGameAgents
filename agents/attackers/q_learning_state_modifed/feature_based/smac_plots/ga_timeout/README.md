# SMAC3 – Resultados de Optimización de Hiperparámetros — GA-timeout only
Generado: 2026-03-18 18:56:04  
Trials en este subconjunto: 7  
Mejor win rate: **28.00%** (trial 6)  

---
## Espacio de búsqueda (rango de hiperparámetros)

| Hiperparámetro | Etiqueta | Tipo | Mínimo | Máximo |
|---|---|---|---|---|
| `n_generations` | Number of Generations | Entero | 10 | 230 |
| `pm_eta` | PM Eta | Float | 0.0 | 60.0 |
| `pm_prob_var` | PM Prob Var | Float | 0.01 | 0.6 |
| `population_size` | Population Size | Entero | 90 | 200 |
| `sbx_eta` | SBX Eta | Float | 0.0 | 60.0 |
| `sbx_prob` | SBX Probability | Float | 0.8 | 1.0 |

**Valores fijos (no optimizados):**
- `sbx_prob_var` = 1.0
- `pm_prob` = 1.0
- `test_episodes` = 25
- `reward_range` = (−1, 1)

---
## Tabla de resultados (ordenada por Win Rate)

| Trial | Win Rate | Costo | Term. Reason | Pop | Gens | SBX_P | SBX_eta | PM_pv | PM_eta |
|------:|----------:|------:|:-------------|----:|----:|------:|--------:|------:|-------:|
| 6 | 28.00% | 0.7200 | ga_timeout | 105 | 109 | 0.8317 | 35.72 | 0.3302 | 17.42 |
| 7 | 28.00% | 0.7200 | ga_timeout | 105 | 109 | 0.8317 | 35.72 | 0.3302 | 17.42 |
| 8 | 28.00% | 0.7200 | ga_timeout | 105 | 109 | 0.8317 | 35.72 | 0.3302 | 17.42 |
| 9 | 28.00% | 0.7200 | ga_timeout | 147 | 158 | 0.9136 | 22.81 | 0.1319 | 57.04 |
| 10 | 28.00% | 0.7200 | ga_timeout | 147 | 158 | 0.9136 | 22.81 | 0.1319 | 57.04 |
| 1 | 24.00% | 0.7600 | ga_timeout | 145 | 185 | 0.9738 | 11.18 | 0.5965 | 3.51 |
| 2 | 24.00% | 0.7600 | ga_timeout | 145 | 185 | 0.9738 | 11.18 | 0.5965 | 3.51 |

---
## Gráficas generadas

![smac_boxplot_n_generations.png](smac_boxplot_n_generations.png)

![smac_boxplot_pm_eta.png](smac_boxplot_pm_eta.png)

![smac_boxplot_pm_prob_var.png](smac_boxplot_pm_prob_var.png)

![smac_boxplot_population_size.png](smac_boxplot_population_size.png)

![smac_boxplot_sbx_eta.png](smac_boxplot_sbx_eta.png)

![smac_boxplot_sbx_prob.png](smac_boxplot_sbx_prob.png)

![smac_correlation_matrix.png](smac_correlation_matrix.png)

![smac_cost_per_trial.png](smac_cost_per_trial.png)

![smac_hyperparameters_vs_winrate.png](smac_hyperparameters_vs_winrate.png)

![smac_hyperparams_evolution.png](smac_hyperparams_evolution.png)

![smac_n_generations_vs_winrate.png](smac_n_generations_vs_winrate.png)

![smac_pm_eta_vs_winrate.png](smac_pm_eta_vs_winrate.png)

![smac_pm_prob_var_vs_winrate.png](smac_pm_prob_var_vs_winrate.png)

![smac_population_size_vs_winrate.png](smac_population_size_vs_winrate.png)

![smac_sbx_eta_vs_winrate.png](smac_sbx_eta_vs_winrate.png)

![smac_sbx_prob_vs_winrate.png](smac_sbx_prob_vs_winrate.png)

![smac_win_rate_distribution.png](smac_win_rate_distribution.png)

![smac_win_rate_per_trial.png](smac_win_rate_per_trial.png)

