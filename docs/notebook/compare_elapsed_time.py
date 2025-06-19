import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

plt.style.use("seaborn-v0_8-colorblind")
plt.rcParams["lines.linewidth"] = 3.0
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 15
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["font.family"] = "DejaVu Sans"

# Parse log data
log_data_1np = """
2025-02-25 17:32:20 | INFO | pytdscf.simulator_cls:_execute:376 - End     0 step; propagated    0.200 [fs]; AVG Krylov iteration: 0.04
2025-02-25 17:35:02 | INFO | pytdscf.simulator_cls:_execute:376 - End   100 step; propagated   20.200 [fs]; AVG Krylov iteration: 0.42
2025-02-25 17:38:00 | INFO | pytdscf.simulator_cls:_execute:376 - End   200 step; propagated   40.200 [fs]; AVG Krylov iteration: 0.77
2025-02-25 17:41:13 | INFO | pytdscf.simulator_cls:_execute:376 - End   300 step; propagated   60.200 [fs]; AVG Krylov iteration: 1.05
2025-02-25 17:44:40 | INFO | pytdscf.simulator_cls:_execute:376 - End   400 step; propagated   80.200 [fs]; AVG Krylov iteration: 1.39
2025-02-25 17:48:21 | INFO | pytdscf.simulator_cls:_execute:376 - End   500 step; propagated  100.200 [fs]; AVG Krylov iteration: 1.68
2025-02-25 17:52:19 | INFO | pytdscf.simulator_cls:_execute:376 - End   600 step; propagated  120.200 [fs]; AVG Krylov iteration: 2.00
2025-02-25 17:56:33 | INFO | pytdscf.simulator_cls:_execute:376 - End   700 step; propagated  140.200 [fs]; AVG Krylov iteration: 2.34
2025-02-25 18:01:05 | INFO | pytdscf.simulator_cls:_execute:376 - End   800 step; propagated  160.200 [fs]; AVG Krylov iteration: 2.66
2025-02-25 18:05:57 | INFO | pytdscf.simulator_cls:_execute:376 - End   900 step; propagated  180.200 [fs]; AVG Krylov iteration: 2.97
2025-02-25 18:11:09 | INFO | pytdscf.simulator_cls:_execute:376 - End  1000 step; propagated  200.200 [fs]; AVG Krylov iteration: 3.29
2025-02-25 18:16:41 | INFO | pytdscf.simulator_cls:_execute:376 - End  1100 step; propagated  220.200 [fs]; AVG Krylov iteration: 3.48
2025-02-25 18:22:36 | INFO | pytdscf.simulator_cls:_execute:376 - End  1200 step; propagated  240.200 [fs]; AVG Krylov iteration: 3.65
2025-02-25 18:28:52 | INFO | pytdscf.simulator_cls:_execute:376 - End  1300 step; propagated  260.200 [fs]; AVG Krylov iteration: 3.82
2025-02-25 18:35:27 | INFO | pytdscf.simulator_cls:_execute:376 - End  1400 step; propagated  280.200 [fs]; AVG Krylov iteration: 3.96
2025-02-25 18:42:24 | INFO | pytdscf.simulator_cls:_execute:376 - End  1500 step; propagated  300.200 [fs]; AVG Krylov iteration: 4.12
2025-02-25 18:49:48 | INFO | pytdscf.simulator_cls:_execute:376 - End  1600 step; propagated  320.200 [fs]; AVG Krylov iteration: 4.28
2025-02-25 18:57:39 | INFO | pytdscf.simulator_cls:_execute:376 - End  1700 step; propagated  340.200 [fs]; AVG Krylov iteration: 4.45
2025-02-25 19:05:55 | INFO | pytdscf.simulator_cls:_execute:376 - End  1800 step; propagated  360.200 [fs]; AVG Krylov iteration: 4.60
2025-02-25 19:14:41 | INFO | pytdscf.simulator_cls:_execute:376 - End  1900 step; propagated  380.200 [fs]; AVG Krylov iteration: 4.77
"""
log_data_4np = """
2025-02-26 21:41:29 | INFO | pytdscf.simulator_cls:_execute:376 - End     0 step; propagated    0.200 [fs]; AVG Krylov iteration: 0.00
2025-02-26 21:42:15 | INFO | pytdscf.simulator_cls:_execute:376 - End   100 step; propagated   20.200 [fs]; AVG Krylov iteration: 0.00
2025-02-26 21:43:11 | INFO | pytdscf.simulator_cls:_execute:376 - End   200 step; propagated   40.200 [fs]; AVG Krylov iteration: 0.00
2025-02-26 21:44:17 | INFO | pytdscf.simulator_cls:_execute:376 - End   300 step; propagated   60.200 [fs]; AVG Krylov iteration: 0.35
2025-02-26 21:45:33 | INFO | pytdscf.simulator_cls:_execute:376 - End   400 step; propagated   80.200 [fs]; AVG Krylov iteration: 0.96
2025-02-26 21:46:59 | INFO | pytdscf.simulator_cls:_execute:376 - End   500 step; propagated  100.200 [fs]; AVG Krylov iteration: 1.61
2025-02-26 21:48:35 | INFO | pytdscf.simulator_cls:_execute:376 - End   600 step; propagated  120.200 [fs]; AVG Krylov iteration: 2.17
2025-02-26 21:50:22 | INFO | pytdscf.simulator_cls:_execute:376 - End   700 step; propagated  140.200 [fs]; AVG Krylov iteration: 2.85
2025-02-26 21:52:23 | INFO | pytdscf.simulator_cls:_execute:376 - End   800 step; propagated  160.200 [fs]; AVG Krylov iteration: 3.50
2025-02-26 21:54:39 | INFO | pytdscf.simulator_cls:_execute:376 - End   900 step; propagated  180.200 [fs]; AVG Krylov iteration: 4.07
2025-02-26 21:57:08 | INFO | pytdscf.simulator_cls:_execute:376 - End  1000 step; propagated  200.200 [fs]; AVG Krylov iteration: 4.72
2025-02-26 21:59:50 | INFO | pytdscf.simulator_cls:_execute:376 - End  1100 step; propagated  220.200 [fs]; AVG Krylov iteration: 4.91
2025-02-26 22:02:42 | INFO | pytdscf.simulator_cls:_execute:376 - End  1200 step; propagated  240.200 [fs]; AVG Krylov iteration: 4.98
2025-02-26 22:05:51 | INFO | pytdscf.simulator_cls:_execute:376 - End  1300 step; propagated  260.200 [fs]; AVG Krylov iteration: 5.00
2025-02-26 22:09:13 | INFO | pytdscf.simulator_cls:_execute:376 - End  1400 step; propagated  280.200 [fs]; AVG Krylov iteration: 5.00
2025-02-26 22:12:50 | INFO | pytdscf.simulator_cls:_execute:376 - End  1500 step; propagated  300.200 [fs]; AVG Krylov iteration: 5.00
2025-02-26 22:16:43 | INFO | pytdscf.simulator_cls:_execute:376 - End  1600 step; propagated  320.200 [fs]; AVG Krylov iteration: 5.00
2025-02-26 22:21:07 | INFO | pytdscf.simulator_cls:_execute:376 - End  1700 step; propagated  340.200 [fs]; AVG Krylov iteration: 5.00
2025-02-26 22:26:17 | INFO | pytdscf.simulator_cls:_execute:376 - End  1800 step; propagated  360.200 [fs]; AVG Krylov iteration: 5.00
2025-02-26 22:32:36 | INFO | pytdscf.simulator_cls:_execute:376 - End  1900 step; propagated  380.200 [fs]; AVG Krylov iteration: 5.00
"""

# Extract data
steps_1np = []
times_1np = []
# krylov_iterations = []
start_time = None

for line in log_data_1np.strip().split("\n"):
    parts = line.split(" | ")
    time_str = parts[0]
    step_str = parts[2].split(";")[0].split()[-2]
    # krylov_str = parts[2].split(";")[-1].split(":")[-1].strip()
    time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    if start_time is None:
        start_time = time
    elapsed_time = (time - start_time).total_seconds()
    steps_1np.append(int(step_str))
    times_1np.append(elapsed_time)
    # krylov_iterations.append(float(krylov_str))

steps_4np = []
times_4np = []
start_time = None
for line in log_data_4np.strip().split("\n"):
    parts = line.split(" | ")
    time_str = parts[0]
    step_str = parts[2].split(";")[0].split()[-2]
    # krylov_str = parts[2].split(";")[-1].split(":")[-1].strip()
    time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    if start_time is None:
        start_time = time
    elapsed_time = (time - start_time).total_seconds()
    steps_4np.append(int(step_str))
    times_4np.append(elapsed_time)

# Plot
fig, ax1 = plt.subplots(figsize=(9, 7))

ax1.plot(
    np.array(steps_1np) * 0.2 - 3, np.array(times_1np), "x-", lw=3, c="navy"
)
ax1.plot(
    np.array(steps_4np) * 0.2 + 3, np.array(times_4np), "x-", lw=3, c="orange"
)
ax1.bar(
    np.array(steps_1np) * 0.2 - 3,
    np.array(times_1np),
    width=6,
    label="OMP_NUM_THREADS=48, # of processes=1",
    color="navy",
)
ax1.bar(
    np.array(steps_4np) * 0.2 + 3,
    np.array(times_4np),
    width=6,
    label="OMP_NUM_THREADS=12, # of processes=4",
    color="orange",
)
ax1.set_xlabel("Propagated time [fs]")
ax1.set_xticks(np.arange(0, 401, 40))
ax1.set_yticks(np.arange(0, 6001, 600))
ax1.set_ylabel("Elapsed Time [sec]")  # , color='tab:blue')
ax1.tick_params(axis="y")  # , labelcolor='tab:blue')

# # Right axis (Krylov iteration)
# ax2 = ax1.twinx()
# ax2.plot(steps, krylov_iterations, 's-', color='tab:red', label="AVG Krylov iteration")
# ax2.set_ylabel("AVG Krylov iteration", color='tab:red')
# ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")

plt.title("Propagated Time vs Elapsed Time with D=60")
plt.legend()
plt.grid(axis="y")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("propagated_time_vs_elapsed_time.pdf")
plt.show()

print(np.array(times_1np) / np.array(times_4np))
