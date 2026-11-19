# 连续 Vector 流水的成对验证

|算子|数据角色|SKU|样例数|v0.20 WAPE / max|v0.21 WAPE / max|
|---|---|---|---:|---:|---:|
|sigmoid_fp32|development replay|Ascend950DT_9582|13|26.52 / 55.54|6.24 / 11.44|
|sigmoid_fp32|development replay|Ascend950PR_957b|10|75.37 / 120.50|15.53 / 24.72|
|sigmoid_fp32|post-freeze independent|Ascend950DT_9582|10|31.17 / 46.68|5.59 / 12.85|
|sigmoid_fp32|post-freeze independent|Ascend950PR_957b|10|84.33 / 105.31|17.66 / 26.96|
|cast_bf16_fp32|development replay|Ascend950DT_9582|16|31.44 / 42.53|6.63 / 21.76|
|cast_bf16_fp32|development replay|Ascend950PR_957b|16|9.02 / 19.83|11.08 / 29.95|
|cast_bf16_fp32|post-freeze independent|Ascend950DT_9582|10|28.43 / 38.60|10.22 / 22.33|
|cast_bf16_fp32|post-freeze independent|Ascend950PR_957b|10|7.92 / 22.53|6.67 / 22.53|

误差单位为百分比。新留出完成 40 / 40 次。分母只包括通过时间窗和全元素校验的记录。每个 SKU 和算子分别统计。Sigmoid DT 的三个早期开发点用原版 v0.20 wheel 回放，逐组两版比较使用完全相同的 CA 窗口。

|算子|数据角色|每卡 shape|SKU|v0.20 / µs|v0.21 / µs|CA / µs|v0.21 误差|970DT 外推 / µs|
|---|---|---|---|---:|---:|---:|---:|---:|
|sigmoid_fp32|development replay|[262144, 12]|Ascend950DT_9582|10.28|8.00|7.76|+3.1%|5.55|
|sigmoid_fp32|development replay|[262144, 12]|Ascend950PR_957b|21.73|12.34|11.43|+8.0%|5.55|
|sigmoid_fp32|development replay|[72, 12]|Ascend950DT_9582|2.79|2.79|3.11|-10.4%|2.56|
|sigmoid_fp32|development replay|[72, 12]|Ascend950PR_957b|2.79|2.79|2.67|+4.5%|2.56|
|sigmoid_fp32|development replay|[1048576, 12]|Ascend950DT_9582|28.86|19.72|18.55|+6.3%|11.32|
|sigmoid_fp32|development replay|[1048576, 12]|Ascend950PR_957b|75.60|39.63|34.28|+15.6%|11.32|
|sigmoid_fp32|development replay|[2048, 12]|Ascend950DT_9582|3.34|3.34|3.18|+5.3%|3.07|
|sigmoid_fp32|development replay|[2048, 12]|Ascend950PR_957b|3.40|3.35|2.79|+19.8%|3.07|
|sigmoid_fp32|development replay|[2097152, 12]|Ascend950DT_9582|54.18|45.90|42.69|+7.5%|19.35|
|sigmoid_fp32|development replay|[2097152, 12]|Ascend950PR_957b|147.42|104.72|88.59|+18.2%|19.35|
|sigmoid_fp32|development replay|[32768, 12]|Ascend950DT_9582|4.41|4.12|3.70|+11.4%|3.58|
|sigmoid_fp32|development replay|[32768, 12]|Ascend950PR_957b|5.90|4.34|4.13|+5.1%|3.58|
|sigmoid_fp32|development replay|[9, 96]|Ascend950DT_9582|2.79|2.79|3.03|-8.0%|2.56|
|sigmoid_fp32|development replay|[9, 96]|Ascend950PR_957b|2.79|2.79|2.66|+5.1%|2.56|
|sigmoid_fp32|development replay|[4096, 24]|Ascend950DT_9582|3.64|3.57|3.43|+4.3%|3.28|
|sigmoid_fp32|development replay|[4096, 24]|Ascend950PR_957b|3.98|3.59|3.19|+12.5%|3.28|
|sigmoid_fp32|development replay|[16384, 48]|Ascend950DT_9582|5.43|4.85|4.85|-0.0%|3.93|
|sigmoid_fp32|development replay|[16384, 48]|Ascend950PR_957b|8.43|6.02|4.82|+24.7%|3.93|
|sigmoid_fp32|development replay|[131072, 12]|Ascend950DT_9582|6.86|5.71|5.99|-4.6%|4.66|
|sigmoid_fp32|development replay|[131072, 12]|Ascend950PR_957b|12.86|8.11|7.88|+2.9%|4.66|
|sigmoid_fp32|development replay|[1, 12]|Ascend950DT_9582|2.69|2.69|2.72|-1.3%|2.47|
|sigmoid_fp32|development replay|[300, 12]|Ascend950DT_9582|3.04|3.04|3.08|-1.4%|2.79|
|sigmoid_fp32|development replay|[3072, 96]|Ascend950DT_9582|4.15|3.94|3.62|+8.7%|3.48|
|cast_bf16_fp32|development replay|[262144, 32]|Ascend950DT_9582|8.36|8.36|9.97|-16.1%|6.05|
|cast_bf16_fp32|development replay|[262144, 32]|Ascend950PR_957b|13.05|13.05|14.08|-7.3%|6.05|
|cast_bf16_fp32|development replay|[1048576, 32]|Ascend950DT_9582|23.27|36.37|35.53|+2.3%|13.34|
|cast_bf16_fp32|development replay|[1048576, 32]|Ascend950PR_957b|64.36|88.94|70.71|+25.8%|13.34|
|cast_bf16_fp32|development replay|[2097152, 32]|Ascend950DT_9582|69.52|99.92|106.76|-6.4%|42.54|
|cast_bf16_fp32|development replay|[2097152, 32]|Ascend950PR_957b|208.16|232.75|223.49|+4.1%|42.54|
|cast_bf16_fp32|development replay|[8192, 5120]|Ascend950DT_9582|31.77|52.26|55.29|-5.5%|19.23|
|cast_bf16_fp32|development replay|[8192, 5120]|Ascend950PR_957b|100.31|124.89|114.13|+9.4%|19.23|
|cast_bf16_fp32|development replay|[4096, 7168]|Ascend950DT_9582|20.79|28.42|29.72|-4.4%|12.13|
|cast_bf16_fp32|development replay|[4096, 7168]|Ascend950PR_957b|46.38|70.97|54.61|+29.9%|12.13|
|cast_bf16_fp32|development replay|[262144, 64]|Ascend950DT_9582|13.33|13.33|17.04|-21.8%|8.48|
|cast_bf16_fp32|development replay|[262144, 64]|Ascend950PR_957b|25.03|25.03|23.76|+5.3%|8.48|
|cast_bf16_fp32|development replay|[32, 4096]|Ascend950DT_9582|2.74|2.74|2.77|-1.3%|2.51|
|cast_bf16_fp32|development replay|[32, 4096]|Ascend950PR_957b|2.68|2.68|2.87|-6.5%|2.51|
|cast_bf16_fp32|development replay|[1, 64]|Ascend950DT_9582|1.96|1.96|2.17|-9.9%|1.79|
|cast_bf16_fp32|development replay|[1, 64]|Ascend950PR_957b|1.96|1.96|2.00|-1.9%|1.79|
|cast_bf16_fp32|development replay|[72, 4096]|Ascend950DT_9582|2.93|2.93|2.85|+2.8%|3.05|
|cast_bf16_fp32|development replay|[72, 4096]|Ascend950PR_957b|2.91|2.91|2.98|-2.3%|3.05|
|cast_bf16_fp32|development replay|[1, 3584]|Ascend950DT_9582|2.10|2.10|2.22|-5.3%|1.92|
|cast_bf16_fp32|development replay|[1, 3584]|Ascend950PR_957b|2.10|2.10|2.10|-0.2%|1.92|
|cast_bf16_fp32|development replay|[96, 7168]|Ascend950DT_9582|3.40|3.40|3.10|+9.9%|3.28|
|cast_bf16_fp32|development replay|[96, 7168]|Ascend950PR_957b|3.44|3.44|2.87|+19.8%|3.28|
|cast_bf16_fp32|development replay|[9, 4096]|Ascend950DT_9582|2.28|2.28|2.33|-2.0%|2.09|
|cast_bf16_fp32|development replay|[9, 4096]|Ascend950PR_957b|2.28|2.28|2.22|+2.7%|2.09|
|cast_bf16_fp32|development replay|[48, 7168]|Ascend950DT_9582|2.99|2.99|2.89|+3.4%|3.08|
|cast_bf16_fp32|development replay|[48, 7168]|Ascend950PR_957b|2.97|2.97|2.91|+2.0%|3.08|
|cast_bf16_fp32|development replay|[16, 7168]|Ascend950DT_9582|2.66|2.66|2.72|-2.4%|2.44|
|cast_bf16_fp32|development replay|[16, 7168]|Ascend950PR_957b|2.66|2.66|2.83|-6.1%|2.44|
|cast_bf16_fp32|development replay|[128, 5120]|Ascend950DT_9582|3.36|3.36|3.06|+9.7%|3.26|
|cast_bf16_fp32|development replay|[128, 5120]|Ascend950PR_957b|3.40|3.40|3.56|-4.5%|3.26|
|cast_bf16_fp32|development replay|[144, 6144]|Ascend950DT_9582|3.64|3.64|3.52|+3.2%|3.39|
|cast_bf16_fp32|development replay|[144, 6144]|Ascend950PR_957b|3.72|3.72|4.11|-9.6%|3.39|
|sigmoid_fp32|post-freeze independent|[3073, 12]|Ascend950PR_957b|3.55|3.42|2.93|+16.7%|3.13|
|sigmoid_fp32|post-freeze independent|[3073, 12]|Ascend950DT_9582|3.42|3.42|3.07|+11.5%|3.13|
|cast_bf16_fp32|post-freeze independent|[65, 5120]|Ascend950DT_9582|2.98|2.98|3.05|-2.4%|3.07|
|cast_bf16_fp32|post-freeze independent|[65, 5120]|Ascend950PR_957b|2.96|2.96|2.94|+0.7%|3.07|
|sigmoid_fp32|post-freeze independent|[524288, 12]|Ascend950DT_9582|16.47|11.91|11.23|+6.0%|7.56|
|cast_bf16_fp32|post-freeze independent|[393216, 32]|Ascend950DT_9582|10.85|10.85|13.73|-21.0%|7.27|
|cast_bf16_fp32|post-freeze independent|[393216, 32]|Ascend950PR_957b|19.04|19.04|18.83|+1.1%|7.27|
|sigmoid_fp32|post-freeze independent|[65537, 24]|Ascend950DT_9582|6.90|5.77|5.87|-1.7%|4.66|
|sigmoid_fp32|post-freeze independent|[65537, 24]|Ascend950PR_957b|12.86|8.11|7.59|+7.0%|4.66|
|cast_bf16_fp32|post-freeze independent|[3072, 5120]|Ascend950DT_9582|12.71|12.71|16.37|-22.3%|8.18|
|sigmoid_fp32|post-freeze independent|[524288, 12]|Ascend950PR_957b|39.88|21.81|19.42|+12.3%|7.56|
|cast_bf16_fp32|post-freeze independent|[3072, 5120]|Ascend950PR_957b|23.53|23.53|23.24|+1.3%|8.18|
|sigmoid_fp32|post-freeze independent|[1572864, 12]|Ascend950DT_9582|41.49|30.14|28.56|+5.5%|15.41|
|sigmoid_fp32|post-freeze independent|[1572864, 12]|Ascend950PR_957b|111.50|69.04|54.38|+27.0%|15.41|
|cast_bf16_fp32|post-freeze independent|[1572864, 32]|Ascend950DT_9582|44.36|68.14|72.24|-5.7%|27.00|
|sigmoid_fp32|post-freeze independent|[3, 96]|Ascend950DT_9582|2.72|2.72|3.04|-10.3%|2.49|
|sigmoid_fp32|post-freeze independent|[3, 96]|Ascend950PR_957b|2.72|2.72|2.63|+3.6%|2.49|
|cast_bf16_fp32|post-freeze independent|[3, 7168]|Ascend950DT_9582|2.20|2.20|2.38|-7.4%|2.02|
|cast_bf16_fp32|post-freeze independent|[3, 7168]|Ascend950PR_957b|2.20|2.20|2.44|-9.7%|2.02|
|sigmoid_fp32|post-freeze independent|[129, 12]|Ascend950DT_9582|2.93|2.93|2.93|-0.0%|2.68|
|sigmoid_fp32|post-freeze independent|[129, 12]|Ascend950PR_957b|2.93|2.93|2.72|+7.8%|2.68|
|cast_bf16_fp32|post-freeze independent|[17, 32]|Ascend950DT_9582|1.99|1.99|2.41|-17.1%|1.83|
|cast_bf16_fp32|post-freeze independent|[17, 32]|Ascend950PR_957b|1.99|1.99|1.95|+2.2%|1.83|
|sigmoid_fp32|post-freeze independent|[4545, 24]|Ascend950DT_9582|3.66|3.60|3.27|+10.3%|3.30|
|sigmoid_fp32|post-freeze independent|[4545, 24]|Ascend950PR_957b|4.07|3.61|3.18|+13.5%|3.30|
|cast_bf16_fp32|post-freeze independent|[111, 7168]|Ascend950DT_9582|3.53|3.53|3.17|+11.3%|3.34|
|cast_bf16_fp32|post-freeze independent|[111, 7168]|Ascend950PR_957b|3.59|3.59|3.74|-4.1%|3.34|
|sigmoid_fp32|post-freeze independent|[16385, 48]|Ascend950DT_9582|5.45|4.91|4.77|+3.0%|3.93|
|cast_bf16_fp32|post-freeze independent|[1572864, 32]|Ascend950PR_957b|136.26|160.85|149.96|+7.3%|27.00|
|sigmoid_fp32|post-freeze independent|[16385, 48]|Ascend950PR_957b|8.43|6.02|5.72|+5.1%|3.93|
|cast_bf16_fp32|post-freeze independent|[511, 5120]|Ascend950DT_9582|4.94|4.94|5.38|-8.1%|4.38|
|cast_bf16_fp32|post-freeze independent|[511, 5120]|Ascend950PR_957b|5.09|5.09|6.58|-22.5%|4.38|
|sigmoid_fp32|post-freeze independent|[4097, 96]|Ascend950DT_9582|4.43|4.17|3.70|+12.9%|3.58|
|sigmoid_fp32|post-freeze independent|[4097, 96]|Ascend950PR_957b|5.90|4.34|4.25|+2.1%|3.58|
|cast_bf16_fp32|post-freeze independent|[32769, 32]|Ascend950DT_9582|3.83|3.83|3.68|+4.3%|3.49|
|cast_bf16_fp32|post-freeze independent|[32769, 32]|Ascend950PR_957b|3.94|3.94|4.48|-12.0%|3.49|
|sigmoid_fp32|post-freeze independent|[98304, 12]|Ascend950DT_9582|6.28|5.48|5.52|-0.9%|4.32|
|sigmoid_fp32|post-freeze independent|[98304, 12]|Ascend950PR_957b|10.34|6.68|6.87|-2.7%|4.32|
|cast_bf16_fp32|post-freeze independent|[98305, 64]|Ascend950DT_9582|7.12|7.12|8.23|-13.5%|5.44|
|cast_bf16_fp32|post-freeze independent|[98305, 64]|Ascend950PR_957b|10.05|10.05|11.22|-10.4%|5.44|

[冻结模型与预测](freeze.json) · [机器可读结果与 DT/PR 加速比](evaluation.json) · [参数来源](method.md) · [独立长 shape 流水](independent-figures.md)
