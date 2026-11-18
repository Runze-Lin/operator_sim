# PR holdout data contract

`budget.json` preregisters cases and costs; `freeze.json.predictions[]` has `id,op,shape,pr_us,dt_us,prior_pr_us,pr_aic,pr_aiv,dt_aic,dt_aiv`. `holdout.json` is an array of qualified rows with the same flat fields plus `pr_ca_us,dt_ca_us,active_aic,active_aiv,role,contract,coverage,dt,pr,pr_cost`. `pr` includes `kernel`, `correctness`, `soc`, per-core windows, sha256, log and trace references; `pr_cost` includes `seconds,cpu_seconds,raw_bytes,exit_code`. `attempts.json` lists all attempts (including rejected) with cost. DT CA role is historical (not independent of DT calibration); PR timing was never accessed before freeze.

`dt-preflight.json` records all 36 DT checks. `campaign-cost.json` includes total campaign elapsed wall, simulator process-seconds, CPU-seconds, raw bytes and count of DT launches. `coverage.md` / `coverage.json` break down relative error by workload size and compare three controlled shape pairs.
