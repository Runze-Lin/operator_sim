# DynamicMXQuant · 大输入共享搬运

|输入与计时口径|芯片|形状|公开 / 旧资源 / 新情景 (µs)|实测 (µs)|公开 / 新情景有符号误差|
|---|---|---|---:|---:|---:|
|historical native B8 · native wrapper envelope|950DT|[262144, 7168]|1200.415 / 1394.986 / 1867.553|1778.988|-32.52% / +4.98%|
|previously running native B4 · native wrapper envelope|950DT|[131072, 7168]|602.614 / 683.003 / 924.228|待测|N/A|
|official forward pair · all-core instruction window|950DT|[16384, 7168]|79.958 / 73.360 / 98.355|待测|N/A|
|official forward pair · all-core instruction window|950PR|[16384, 7168]|178.669 / 178.669 / 291.546|待测|N/A|
