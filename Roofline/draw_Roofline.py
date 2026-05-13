import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import pandas as pd
import os
from matplotlib.patches import Patch


# =========================
# 1) HW 상수/피크 (SI 단위)
# =========================
PEAK_INST_PER_CYCLE = 128 * 4 * 1
SM_HZ               = 2.52e9
PEAK_INST_PER_S = PEAK_INST_PER_CYCLE * SM_HZ
PEAK_GIPS = PEAK_INST_PER_S / 1e9

#RTX 4090 (Ada) 기준, 128 SM
NUM_SM = 128

BW_DRAM = 1008e9  # GB/s
S_DRAM = 32      # Bytes/TXN (32B/사이클 가정)
BW_DRAM_TXN_S = BW_DRAM / S_DRAM

L2_BYTES_PER_CYCLE = 1708
L2_HZ              = SM_HZ
S_L2               = 32
BW_L2_TXN_S = (L2_BYTES_PER_CYCLE * L2_HZ) / S_L2

L1_BYTES_PER_CYCLE = 121.2 * NUM_SM
L1_HZ = SM_HZ
S_L1 = 32
BW_L1_TXN_S = (L1_BYTES_PER_CYCLE * L1_HZ) / S_L1

SMEM_BYTES_PER_CYCLE = 127.9 * NUM_SM
SMEM_HZ = SM_HZ
S_SMEM = 128
BW_SMEM_TXN_S = (SMEM_BYTES_PER_CYCLE * SMEM_HZ) / S_SMEM

# X축 범위(inst/TXN)
II_MIN, II_MAX = 1e-2, 1e3
X_OVERRIDE = (II_MIN, II_MAX)
Y_OVERRIDE = None  # (ymin, ymax) 수동 지정 원하면 값 넣기 (단위=GIPS)

# =========================
# 2) Achieved points from counters
# =========================

PROFILE_DIR = "./profile"
figure_dir = "./figure"

os.makedirs(PROFILE_DIR, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# =========================
# ceiling & wall color
# =========================

Color_setting = {
    "c1_ceiling": "black",
    "L1_ceiling": "#2196F3",    # 파랑
    "L2_ceiling": "#FF9800",    # 주황
    "DRAM_ceiling": "#F44336",  # 빨강
    "global_memory_wall": "#9E9E9E",  # 회색
    "c2_ceiling": "black",
    "SMEM_ceiling": "#2196F3",
    "shared_memory_wall": "#9E9E9E",
}
# =========================
# gloabal_Memory + Cache (DRAM + L2 + L1) Roofline 계산
# =========================

def parse_ncu_csv(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if '"ID"' in l)
    df = pd.read_csv(filepath, skiprows=start)
    df['Metric Value'] = df['Metric Value'].astype(str).str.replace(',', '').astype(float)
    return df.groupby('Metric Name')['Metric Value'].sum().to_dict()

def make_ach_from_csv(ncu_csv, kernel_time_sec, color, marker, label_prefix):
    m = parse_ncu_csv(ncu_csv)
    return {
        "kernel_execution_time":              kernel_time_sec,
        "warp_level_executed_instructions":   m.get("smsp__inst_executed.sum", 0),
        "thread_level_executed_instructions": m.get("smsp__thread_inst_executed.sum", 0),
        "global_ld_inst": m.get("smsp__inst_executed_op_global_ld.sum", 0),
        "global_st_inst": m.get("smsp__inst_executed_op_global_st.sum", 0),
        "shared_ld_inst": m.get("smsp__inst_executed_op_shared_ld.sum", 0),
        "shared_st_inst": m.get("smsp__inst_executed_op_shared_st.sum", 0),
        "gld_txn":    m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", 0),
        "gst_txn":    m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", 0),
        "smem_ld_txn": m.get("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", 0),
        "smem_st_txn": m.get("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum", 0),
        "l2_rd_txn":  m.get("lts__t_sectors_op_read.sum", 0),
        "l2_wr_txn":  m.get("lts__t_sectors_op_write.sum", 0),
        "dram_rd_txn": m.get("dram__sectors_read.sum", 0),
        "dram_wr_txn": m.get("dram__sectors_write.sum", 0),
        "color": color, "marker": marker, "size": 80, "label_prefix": label_prefix,
    }


def roofline_y_txn(bw_txn_per_s, peak_inst_per_s, ai_inst_per_txn):
    return np.minimum(peak_inst_per_s, bw_txn_per_s * ai_inst_per_txn)

def knee_ai_txn(peak_inst_per_s, bw_txn_per_s):
    return peak_inst_per_s / bw_txn_per_s

def wall_y_max(x, bw_txn_per_s):
    return min(PEAK_GIPS, bw_txn_per_s * x / 1e9)

def draw_global_cache_roofline(ax):
    ii = np.logspace(math.log10(II_MIN), math.log10(II_MAX), 800)
    y_dram = roofline_y_txn(BW_DRAM_TXN_S, PEAK_INST_PER_S, ii) / 1e9
    y_l2   = roofline_y_txn(BW_L2_TXN_S,   PEAK_INST_PER_S, ii) / 1e9
    y_l1   = roofline_y_txn(BW_L1_TXN_S,   PEAK_INST_PER_S, ii) / 1e9

    knee_dram = knee_ai_txn(PEAK_INST_PER_S, BW_DRAM_TXN_S)
    knee_l2   = knee_ai_txn(PEAK_INST_PER_S, BW_L2_TXN_S)
    knee_l1   = knee_ai_txn(PEAK_INST_PER_S, BW_L1_TXN_S)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "axes.edgecolor": "#999999",
        "axes.linewidth": 0.8,
    })

    # 루프라인
    ax.loglog(ii, y_dram, color=Color_setting["DRAM_ceiling"], lw=2.6, label="DRAM roof (GTXN/s slope)")
    ax.loglog(ii, y_l2,   color=Color_setting["L2_ceiling"],   lw=2.6, label="L2 roof (GTXN/s slope)")
    ax.loglog(ii, y_l1,   color=Color_setting["L1_ceiling"],   lw=2.6, label="L1 roof (GTXN/s slope)")

    # ax.text(II_MIN * 1.5, BW_L1_TXN_S * II_MIN * 1.5 / 1e9,
    #         "L1", fontsize=10, color=Color_setting["L1_ceiling"], va='bottom')
    # ax.text(II_MIN * 1.5, BW_L2_TXN_S * II_MIN * 1.5 / 1e9,
    #         "L2", fontsize=10, color=Color_setting["L2_ceiling"], va='bottom')
    # ax.text(II_MIN * 1.5, BW_DRAM_TXN_S * II_MIN * 1.5 / 1e9,
    #         "DRAM", fontsize=10, color=Color_setting["DRAM_ceiling"], va='bottom')

    ax.hlines(PEAK_GIPS, xmin=knee_l1, xmax=II_MAX,
            linestyles="-", color=Color_setting["c1_ceiling"], lw=2.4, label="Compute peak [GIPS]")
    # memory access pattern

    memory_walls = [1/32, 1/4, 1.0]  # stride 8, stride 1, stride 0 (float)
    labels = ["Stride 8(float)", "Stride 1(float)", "Stride 0(float)"]

    ymin = ax.get_ylim()[0]
    for x_wall, label in zip(memory_walls, labels):
        y_top = wall_y_max(x_wall, BW_L1_TXN_S)
        ax.plot([x_wall, x_wall], [ymin, y_top], color=Color_setting["global_memory_wall"], lw=2)
        ax.text(x_wall * 1.05, y_top * 0.5, label,
            fontsize=9, color=Color_setting["global_memory_wall"],
            rotation=90, va='center', ha='left')

    ceiling_handles, ceiling_labels = ax.get_legend_handles_labels()

    # 색상 legend (행렬)
    color_handles = [
        Patch(color="#1B5E20", label="Matrix 1"),
        Patch(color="#4A148C", label="Matrix 2"),
        Patch(color="#BF360C", label="Matrix 3"),
    ]

    # 마커 legend (메모리 계층)
    marker_handles = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=8, label='L1'),
        Line2D([0], [0], marker='^', color='gray', linestyle='None', markersize=8, label='L2'),
        Line2D([0], [0], marker='s', color='gray', linestyle='None', markersize=8, label='DRAM'),
    ]

    all_handles = ceiling_handles + color_handles + marker_handles
    ax.legend(handles=all_handles, loc="lower right", frameon=True, framealpha=0.9)



def draw_shared_roofline(ax):
    ii = np.logspace(math.log10(II_MIN), math.log10(II_MAX), 800)
    y_shared = roofline_y_txn(BW_SMEM_TXN_S, PEAK_INST_PER_S, ii) / 1e9
    knee_shared = knee_ai_txn(PEAK_INST_PER_S, BW_SMEM_TXN_S)

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "axes.edgecolor": "#999999",
        "axes.linewidth": 0.8,
    })

    # 루프라인
    ax.loglog(ii, y_shared, color=Color_setting["SMEM_ceiling"], lw=2.6, label="Shared Memory roof (GTXN/s slope)")
    # ax.text(II_MIN * 1.5, BW_SMEM_TXN_S * II_MIN * 1.5 / 1e9,
    #         "Shared memory ceiling", fontsize=10, color=Color_setting["SMEM_ceiling"], va='bottom')
    ax.hlines(PEAK_GIPS, xmin=knee_shared, xmax=II_MAX,
            linestyles="-", color=Color_setting["c2_ceiling"], lw=2.4, label="Compute peak [GIPS]")
    #shared memory wall
    memory_walls = [1/32, 1.0]  # stride 8, stride 1, stride 0 (float)
    labels = ["32-way bank conflict", "No bank conflict"]
    ymin = ax.get_ylim()[0]
    for x_wall, label in zip(memory_walls, labels):
        y_top = wall_y_max(x_wall, BW_SMEM_TXN_S)
        ax.plot([x_wall, x_wall], [ymin, y_top], color=Color_setting["shared_memory_wall"], lw=2)
        ax.text(x_wall * 1.05, y_top * 0.5, label,
            fontsize=9, color=Color_setting["shared_memory_wall"],
            rotation=90, va='center', ha='left')

    ceiling_handles, ceiling_labels = ax.get_legend_handles_labels()

    all_handles = ceiling_handles
    ax.legend(handles=all_handles, loc="lower right", frameon=True, framealpha=0.9)


def plot_from_counters_global(ax, ach):
    t_sec        = float(ach["kernel_execution_time"])
    inst_sum     = float(ach["thread_level_executed_instructions"]) / 32

    ax.hlines(  ach["warp_level_executed_instructions"]/ ach["kernel_execution_time"] / 1e9, xmin=II_MIN, xmax=II_MAX,
                linestyles=(0, (3, 3)), color=ach["color"], ls='-.', lw=2.4, label="warp-level achieved [GIPS]")

    gips_total = (inst_sum / t_sec) / 1e9  # y값
    
    global_inst = float(ach["global_ld_inst"]) + float(ach["global_st_inst"])
    gips_global = (global_inst / t_sec) / 1e9

    tx_dram = float(ach["dram_rd_txn"]) + float(ach["dram_wr_txn"])
    tx_l2   = float(ach["l2_rd_txn"])  + float(ach["l2_wr_txn"])
    tx_l1   = float(ach["gld_txn"])    + float(ach["gst_txn"]) + 4 * (float(ach["smem_ld_txn"]) + float(ach["smem_st_txn"]))
    

    pts = []
    level_markers = {"DRAM": "s", "L2": "^", "L1": "o"}

    if tx_dram > 0: pts.append(("DRAM", inst_sum / tx_dram, gips_total))
    if tx_l2   > 0: pts.append(("L2",   inst_sum / tx_l2,   gips_total))
    if tx_l1   > 0: pts.append(("L1",   inst_sum / tx_l1,   gips_total))
    if tx_l1 > 0:
        ax.scatter([global_inst / tx_l1], [gips_global], marker='o', facecolors='none', edgecolors=ach["color"], s=ach["size"], zorder=7)

    for level, ii_pt, gips_pt in pts:
        ax.scatter([ii_pt], [gips_pt],
                   s=ach.get("size", 56),
                   marker=level_markers.get(level, "D"),
                   c=ach.get("color", "#444"),
                   edgecolors="white", linewidths=0.9, zorder=7)
        # ax.annotate(f'{ach.get("label_prefix","achieved")}@{level}',
        #             (ii_pt, gips_pt),
        #             fontsize=9, ha='left', va='bottom')

def plot_from_counters_shared(ax, ach):
    t_sec = float(ach["kernel_execution_time"])

    # shared ld/st instruction만 사용
    shared_inst = float(ach["shared_ld_inst"]) + float(ach["shared_st_inst"])
    gips_shared = (shared_inst / t_sec) / 1e9

    tx_smem = float(ach["smem_ld_txn"]) + float(ach["smem_st_txn"])

    pts = []
    level_markers = {"SMEM": "o"}

    if tx_smem > 0 and shared_inst > 0:
        pts.append(("SMEM", shared_inst / tx_smem, gips_shared))

    for level, ii_pt, gips_pt in pts:
        ax.scatter([ii_pt], [gips_pt],
                   s=ach.get("size", 56),
                   marker=level_markers.get(level, "D"),
                   c=ach.get("color", "#444"),
                   edgecolors="white", linewidths=0.9, zorder=7)



cuSPARSE_data = [
    make_ach_from_csv(f"{PROFILE_DIR}/ncu_cusparse_bottleneck_1_block_group_projection_block_group4.csv", 0.02304 * 1e-3,  "#1B5E20", "D", "cuSPARSE_1"),
    make_ach_from_csv(f"{PROFILE_DIR}/ncu_cusparse_bottleneck_1_block_group_projection_block_group3.csv", 0.0108544 * 1e-3, "#4A148C", "D", "cuSPARSE_2"),
    make_ach_from_csv(f"{PROFILE_DIR}/ncu_cusparse_bottleneck_1_block_group_projection_block_group2.csv", 0.0093184 * 1e-3, "#BF360C", "D", "cuSPARSE_3"),
]

Ginkgo_data = [
    make_ach_from_csv(f"{PROFILE_DIR}/ncu_ginkgo_bottleneck_1_block_group_projection_block_group4.csv", 0.159846 * 1e-3,  "#1B5E20", "o", "Ginkgo_1"),
    make_ach_from_csv(f"{PROFILE_DIR}/ncu_ginkgo_bottleneck_1_block_group_projection_block_group3.csv", 0.0279552 * 1e-3, "#4A148C", "o", "Ginkgo_2"),
    make_ach_from_csv(f"{PROFILE_DIR}/ncu_ginkgo_bottleneck_1_block_group_projection_block_group2.csv", 0.0110592 * 1e-3, "#BF360C", "o", "Ginkgo_3"),
]



cuSPARSE_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
draw_global_cache_roofline(ax1)
draw_shared_roofline(ax2)

for ach in cuSPARSE_data:
    plot_from_counters_global(ax1, ach)
    plot_from_counters_shared(ax2, ach)

cuSPARSE_fig.savefig(f"{figure_dir}/cuSPARSE_roofline.png", dpi=300)

Ginkgo_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
draw_global_cache_roofline(ax1)
draw_shared_roofline(ax2)

for ach in Ginkgo_data:
    plot_from_counters_global(ax1, ach)
    plot_from_counters_shared(ax2, ach)

Ginkgo_fig.savefig(f"{figure_dir}/Ginkgo_roofline.png", dpi=300)

Instruction_Roofline_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
draw_global_cache_roofline(ax1)
draw_shared_roofline(ax2)

Instruction_Roofline_fig.savefig(f"{figure_dir}/Instruction_Roofline.png", dpi=300)