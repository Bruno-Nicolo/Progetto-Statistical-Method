"""
Generazione grafici dai risultati dei test su dataset.

Legge i CSV aggregati prodotti da test_dataset.py e genera tutti i grafici
di analisi: sweep parametri, impatto delta, robustezza (rumore, blur, JPEG),
confronto strategie ROI, e grafici riassuntivi cross-test.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# ============================================================================
# COSTANTE: percorso della cartella contenente i CSV aggregati
# ============================================================================
CSV_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test_output", "dataset",
)

# Directory di output per i grafici
PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "test_output", "plots",
)

# ============================================================================
# Stile globale
# ============================================================================
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]


def _safe_float(series: pd.Series) -> pd.Series:
    """Converte una colonna in float, gestendo 'inf' e valori mancanti."""
    return pd.to_numeric(series, errors="coerce")


def _load_csv(filename: str) -> pd.DataFrame | None:
    """Carica un CSV dalla directory configurata. Ritorna None se non esiste."""
    path = os.path.join(CSV_DIR, filename)
    if not os.path.isfile(path):
        print(f"  ⚠  File non trovato: {path}")
        return None
    df = pd.read_csv(path)
    # Converte colonne numeriche
    for col in df.columns:
        if col not in ("sv_range", "msg_label", "strategy"):
            df[col] = _safe_float(df[col])
    return df


def _savefig(fig, name: str):
    """Salva la figura nella cartella di output."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ Salvato: {path}")


# ============================================================================
# 1. SWEEP PARAMETRI (agg_sweep.csv)
# ============================================================================

def plot_sweep(df: pd.DataFrame):
    """Genera i grafici dal test 1 – sweep parametri."""

    # --- 1a. Heatmap PSNR medio per (block_size × sv_range), delta fissato ---
    for delta_val in sorted(df["delta"].dropna().unique()):
        sub = df[(df["delta"] == delta_val) & (df["msg_label"] == "corto")]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            values="psnr_mean", index="block_size", columns="sv_range", aggfunc="mean",
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)
        ax.set_xlabel("sv_range")
        ax.set_ylabel("block_size")
        ax.set_title(f"PSNR medio (dB) — δ={delta_val}")
        fig.colorbar(im, ax=ax, label="PSNR (dB)")
        _savefig(fig, f"sweep_heatmap_delta{delta_val:.0f}.png")

    # --- 1b. Bar chart PSNR vs delta per sv_range (msg_label = corto) ---
    sub = df[df["msg_label"] == "corto"]
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sv_ranges = sorted(sub["sv_range"].unique())
        deltas = sorted(sub["delta"].dropna().unique())
        x = np.arange(len(deltas))
        width = 0.8 / max(len(sv_ranges), 1)

        for i, sv in enumerate(sv_ranges):
            means = []
            for d in deltas:
                vals = sub[(sub["sv_range"] == sv) & (sub["delta"] == d)]["psnr_mean"]
                means.append(vals.mean() if len(vals) > 0 else np.nan)
            ax.bar(x + i * width, means, width, label=f"sv_range={sv}", color=PALETTE[i % len(PALETTE)])

        ax.set_xticks(x + width * (len(sv_ranges) - 1) / 2)
        ax.set_xticklabels([f"{d:.0f}" for d in deltas])
        ax.set_xlabel("Delta (δ)")
        ax.set_ylabel("PSNR medio (dB)")
        ax.set_title("PSNR vs Delta per sv_range (msg corto, tutti i block_size)")
        ax.legend()
        _savefig(fig, "sweep_psnr_vs_delta_bar.png")

    # --- 1c. Scatter PSNR vs BER ---
    valid = df.dropna(subset=["psnr_mean", "ber_mean"])
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, sv in enumerate(sorted(valid["sv_range"].unique())):
            sub = valid[valid["sv_range"] == sv]
            ax.scatter(
                sub["psnr_mean"], sub["ber_mean"],
                alpha=0.6, s=30, label=f"sv_range={sv}", color=PALETTE[i % len(PALETTE)],
            )
        ax.set_xlabel("PSNR medio (dB)")
        ax.set_ylabel("BER medio")
        ax.set_title("Trade-off PSNR vs BER")
        ax.legend()
        _savefig(fig, "sweep_psnr_vs_ber_scatter.png")


# ============================================================================
# 2. IMPATTO DEL DELTA (agg_delta.csv)
# ============================================================================

def plot_delta(df: pd.DataFrame):
    """Genera i grafici dal test 2 – impatto del delta."""

    deltas = df["delta"].values
    psnr = df["psnr_mean"].values
    psnr_std = df["psnr_std"].fillna(0).values
    ssim = df["ssim_mean"].values
    ssim_std = df["ssim_std"].fillna(0).values
    ber = df["ber_mean"].values
    correct = df["correct_pct"].values

    # --- 2a. PSNR vs Delta con banda d'errore ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(deltas, psnr, "o-", color=PALETTE[0], linewidth=2, markersize=5)
    ax.fill_between(deltas, psnr - psnr_std, psnr + psnr_std, alpha=0.2, color=PALETTE[0])
    ax.set_xlabel("Delta (δ)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs Delta (±1σ)")
    _savefig(fig, "delta_psnr.png")

    # --- 2b. SSIM vs Delta con banda d'errore ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(deltas, ssim, "s-", color=PALETTE[1], linewidth=2, markersize=5)
    ax.fill_between(deltas, ssim - ssim_std, ssim + ssim_std, alpha=0.2, color=PALETTE[1])
    ax.set_xlabel("Delta (δ)")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM vs Delta (±1σ)")
    _savefig(fig, "delta_ssim.png")

    # --- 2c. BER vs Delta ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(deltas, ber, "^-", color=PALETTE[2], linewidth=2, markersize=5)
    ax.set_xlabel("Delta (δ)")
    ax.set_ylabel("BER")
    ax.set_title("BER vs Delta")
    _savefig(fig, "delta_ber.png")

    # --- 2d. Dual-axis: PSNR + Correct% vs Delta ---
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    color_psnr = PALETTE[0]
    color_corr = PALETTE[2]

    ax1.plot(deltas, psnr, "o-", color=color_psnr, linewidth=2, label="PSNR")
    ax1.set_xlabel("Delta (δ)")
    ax1.set_ylabel("PSNR (dB)", color=color_psnr)
    ax1.tick_params(axis="y", labelcolor=color_psnr)

    ax2 = ax1.twinx()
    ax2.plot(deltas, correct, "s--", color=color_corr, linewidth=2, label="Correct%")
    ax2.set_ylabel("Correct (%)", color=color_corr)
    ax2.tick_params(axis="y", labelcolor=color_corr)
    ax2.set_ylim(-5, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("PSNR e Accuratezza vs Delta")
    _savefig(fig, "delta_psnr_correct_dual.png")


# ============================================================================
# 3. ROBUSTEZZA AL RUMORE (agg_noise.csv)
# ============================================================================

def plot_noise(df: pd.DataFrame):
    """Genera i grafici dal test 3 – robustezza al rumore."""

    deltas = sorted(df["delta"].unique())

    # --- 3a. BER vs Sigma, una linea per delta ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, delta in enumerate(deltas):
        sub = df[df["delta"] == delta].sort_values("noise_sigma")
        ax.plot(
            sub["noise_sigma"], sub["ber_mean"], "o-",
            color=PALETTE[i % len(PALETTE)], linewidth=2, markersize=5,
            label=f"δ={delta:.0f}",
        )
        std = sub["ber_std"].fillna(0).values
        ax.fill_between(sub["noise_sigma"], sub["ber_mean"] - std, sub["ber_mean"] + std,
                         alpha=0.15, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Sigma rumore gaussiano")
    ax.set_ylabel("BER medio")
    ax.set_title("BER vs Rumore Gaussiano (±1σ)")
    ax.legend()
    _savefig(fig, "noise_ber_vs_sigma.png")

    # --- 3b. Heatmap BER (Delta × Sigma) ---
    pivot = df.pivot_table(values="ber_mean", index="delta", columns="noise_sigma", aggfunc="mean")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{s:.0f}" for s in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{d:.0f}" for d in pivot.index])
        for i_row in range(len(pivot.index)):
            for j_col in range(len(pivot.columns)):
                val = pivot.values[i_row, j_col]
                if not np.isnan(val):
                    ax.text(j_col, i_row, f"{val:.4f}", ha="center", va="center", fontsize=8)
        ax.set_xlabel("Sigma rumore")
        ax.set_ylabel("Delta (δ)")
        ax.set_title("Heatmap BER — Delta × Sigma")
        fig.colorbar(im, ax=ax, label="BER")
        _savefig(fig, "noise_heatmap_ber.png")

    # --- 3c. Correct% vs Sigma per delta ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, delta in enumerate(deltas):
        sub = df[df["delta"] == delta].sort_values("noise_sigma")
        ax.bar(
            sub["noise_sigma"] + i * 0.8,
            sub["correct_pct"],
            width=0.7,
            color=PALETTE[i % len(PALETTE)],
            alpha=0.8,
            label=f"δ={delta:.0f}",
        )
    ax.set_xlabel("Sigma rumore gaussiano")
    ax.set_ylabel("Correct (%)")
    ax.set_title("Accuratezza vs Rumore Gaussiano")
    ax.set_ylim(0, 110)
    ax.legend()
    _savefig(fig, "noise_correct_pct.png")


# ============================================================================
# 4. ROBUSTEZZA AL BLUR (agg_blur.csv)
# ============================================================================

def plot_blur(df: pd.DataFrame):
    """Genera i grafici dal test 4 – robustezza al blur."""

    deltas = sorted(df["delta"].unique())

    # --- 4a. BER vs Raggio blur per delta ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, delta in enumerate(deltas):
        sub = df[df["delta"] == delta].sort_values("blur_radius")
        ax.plot(
            sub["blur_radius"], sub["ber_mean"], "o-",
            color=PALETTE[i % len(PALETTE)], linewidth=2, markersize=5,
            label=f"δ={delta:.0f}",
        )
        std = sub["ber_std"].fillna(0).values
        ax.fill_between(sub["blur_radius"], sub["ber_mean"] - std, sub["ber_mean"] + std,
                         alpha=0.15, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Raggio blur gaussiano")
    ax.set_ylabel("BER medio")
    ax.set_title("BER vs Blur Gaussiano (±1σ)")
    ax.legend()
    _savefig(fig, "blur_ber_vs_radius.png")

    # --- 4b. Correct% vs Raggio blur ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    radii = sorted(df["blur_radius"].unique())
    x = np.arange(len(radii))
    width = 0.8 / max(len(deltas), 1)
    for i, delta in enumerate(deltas):
        sub = df[df["delta"] == delta].sort_values("blur_radius")
        ax.bar(x + i * width, sub["correct_pct"].values, width,
               color=PALETTE[i % len(PALETTE)], label=f"δ={delta:.0f}")
    ax.set_xticks(x + width * (len(deltas) - 1) / 2)
    ax.set_xticklabels([f"{r:.0f}" for r in radii])
    ax.set_xlabel("Raggio blur")
    ax.set_ylabel("Correct (%)")
    ax.set_title("Accuratezza vs Blur Gaussiano")
    ax.set_ylim(0, 110)
    ax.legend()
    _savefig(fig, "blur_correct_pct.png")


# ============================================================================
# 5. ROBUSTEZZA JPEG (agg_jpeg.csv)
# ============================================================================

def plot_jpeg(df: pd.DataFrame):
    """Genera i grafici dal test 5 – robustezza JPEG."""

    deltas = sorted(df["delta"].unique())

    # --- 5a. BER vs Qualità JPEG per delta ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, delta in enumerate(deltas):
        sub = df[df["delta"] == delta].sort_values("jpeg_quality")
        ax.plot(
            sub["jpeg_quality"], sub["ber_mean"], "o-",
            color=PALETTE[i % len(PALETTE)], linewidth=2, markersize=5,
            label=f"δ={delta:.0f}",
        )
        std = sub["ber_std"].fillna(0).values
        ax.fill_between(sub["jpeg_quality"], sub["ber_mean"] - std, sub["ber_mean"] + std,
                         alpha=0.15, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel("Qualità JPEG")
    ax.set_ylabel("BER medio")
    ax.set_title("BER vs Compressione JPEG (±1σ)")
    ax.invert_xaxis()  # qualità 100 → 30 (degradazione crescente)
    ax.legend()
    _savefig(fig, "jpeg_ber_vs_quality.png")

    # --- 5b. Correct% vs Qualità JPEG ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    qualities = sorted(df["jpeg_quality"].unique(), reverse=True)
    x = np.arange(len(qualities))
    width = 0.8 / max(len(deltas), 1)
    for i, delta in enumerate(deltas):
        sub = df[df["delta"] == delta].set_index("jpeg_quality").reindex(qualities)
        ax.bar(x + i * width, sub["correct_pct"].values, width,
               color=PALETTE[i % len(PALETTE)], label=f"δ={delta:.0f}")
    ax.set_xticks(x + width * (len(deltas) - 1) / 2)
    ax.set_xticklabels([f"{q:.0f}" for q in qualities])
    ax.set_xlabel("Qualità JPEG")
    ax.set_ylabel("Correct (%)")
    ax.set_title("Accuratezza vs Compressione JPEG")
    ax.set_ylim(0, 110)
    ax.legend()
    _savefig(fig, "jpeg_correct_pct.png")


# ============================================================================
# 6. ROI STRATEGIES (agg_roi.csv)
# ============================================================================

def plot_roi(df: pd.DataFrame):
    """Genera i grafici dal test 6 – confronto strategie ROI."""

    strategies = df["strategy"].values
    n = len(strategies)

    # --- 6a. Bar chart comparativo multi-metrica ---
    metrics = [
        ("psnr_mean", "PSNR (dB)"),
        ("ssim_mean", "SSIM"),
        ("ber_mean", "BER"),
        ("correct_pct", "Correct (%)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (col, label) in zip(axes, metrics):
        vals = df[col].fillna(0).values
        bars = ax.bar(range(n), vals, color=PALETTE[:n], edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        # Annota i valori sulle barre
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                fmt = f"{v:.2f}" if v < 10 else f"{v:.1f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt, ha="center", va="bottom", fontsize=8)
    fig.suptitle("Confronto Strategie ROI", fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, "roi_comparison_bar.png")

    # --- 6b. Radar / Spider chart ---
    # Normalizza le metriche tra 0 e 1 per il radar chart
    radar_metrics = ["psnr_mean", "ssim_mean", "correct_pct", "outside_intact_pct"]
    radar_labels = ["PSNR", "SSIM", "Correct%", "Outside OK%"]

    # Per BER, invertiamo (0 = perfetto)
    vals_matrix = []
    for _, row in df.iterrows():
        row_vals = []
        for col in radar_metrics:
            v = row.get(col, 0)
            row_vals.append(v if not np.isnan(v) else 0)
        vals_matrix.append(row_vals)
    vals_matrix = np.array(vals_matrix)

    # Normalizza ciascuna colonna
    col_max = np.nanmax(vals_matrix, axis=0)
    col_max[col_max == 0] = 1
    vals_norm = vals_matrix / col_max

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, strat in enumerate(strategies):
        values = np.concatenate([vals_norm[i], [vals_norm[i][0]]])
        ax.plot(angles, values, "o-", color=PALETTE[i % len(PALETTE)],
                linewidth=2, label=strat, markersize=4)
        ax.fill(angles, values, alpha=0.1, color=PALETTE[i % len(PALETTE)])
    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("Radar Chart — Strategie ROI", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0))
    _savefig(fig, "roi_radar.png")


# ============================================================================
# 7. GRAFICI RIASSUNTIVI CROSS-TEST
# ============================================================================

def plot_cross_test(sweep_df: pd.DataFrame | None, delta_df: pd.DataFrame | None):
    """Genera grafici che incrociano dati da più test."""

    # --- 7a. Box plot PSNR per block_size (dal sweep) ---
    if sweep_df is not None:
        valid = sweep_df.dropna(subset=["psnr_mean"])
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            block_sizes = sorted(valid["block_size"].unique())
            data = [valid[valid["block_size"] == bs]["psnr_mean"].values for bs in block_sizes]
            bp = ax.boxplot(data, labels=[str(int(bs)) for bs in block_sizes], patch_artist=True)
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xlabel("Block Size")
            ax.set_ylabel("PSNR medio (dB)")
            ax.set_title("Distribuzione PSNR per Block Size")
            _savefig(fig, "cross_boxplot_psnr_blocksize.png")

    # --- 7b. Box plot PSNR per sv_range (dal sweep) ---
    if sweep_df is not None:
        valid = sweep_df.dropna(subset=["psnr_mean"])
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            sv_ranges = sorted(valid["sv_range"].unique())
            data = [valid[valid["sv_range"] == sv]["psnr_mean"].values for sv in sv_ranges]
            bp = ax.boxplot(data, labels=sv_ranges, patch_artist=True)
            for patch, color in zip(bp["boxes"], PALETTE):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_xlabel("SV Range")
            ax.set_ylabel("PSNR medio (dB)")
            ax.set_title("Distribuzione PSNR per SV Range")
            _savefig(fig, "cross_boxplot_psnr_svrange.png")

    # --- 7c. Matrice di correlazione parametri-metriche (dal sweep) ---
    if sweep_df is not None:
        corr_cols = ["delta", "block_size", "psnr_mean", "ssim_mean", "ber_mean", "correct_pct"]
        valid = sweep_df[corr_cols].dropna()
        if len(valid) > 5:
            fig, ax = plt.subplots(figsize=(6, 5))
            corr = valid.corr()
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
            labels = ["δ", "block", "PSNR", "SSIM", "BER", "Correct%"]
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            for i_row in range(len(labels)):
                for j_col in range(len(labels)):
                    val = corr.values[i_row, j_col]
                    ax.text(j_col, i_row, f"{val:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="Correlazione di Pearson")
            ax.set_title("Matrice di Correlazione Parametri–Metriche")
            _savefig(fig, "cross_correlation_matrix.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"{'=' * 60}")
    print("GENERAZIONE GRAFICI — STEGANOGRAFIA SVD")
    print(f"{'=' * 60}")
    print(f"CSV dir:  {CSV_DIR}")
    print(f"Plot dir: {PLOT_DIR}")
    print()

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Carica tutti i CSV
    sweep_df = _load_csv("agg_sweep.csv")
    delta_df = _load_csv("agg_delta.csv")
    noise_df = _load_csv("agg_noise.csv")
    blur_df = _load_csv("agg_blur.csv")
    jpeg_df = _load_csv("agg_jpeg.csv")
    roi_df = _load_csv("agg_roi.csv")

    # --- Test 1: Sweep ---
    if sweep_df is not None:
        print("\n📊 Test 1 — Sweep parametri")
        plot_sweep(sweep_df)

    # --- Test 2: Delta ---
    if delta_df is not None:
        print("\n📊 Test 2 — Impatto del delta")
        plot_delta(delta_df)

    # --- Test 3: Rumore ---
    if noise_df is not None:
        print("\n📊 Test 3 — Robustezza al rumore")
        plot_noise(noise_df)

    # --- Test 4: Blur ---
    if blur_df is not None:
        print("\n📊 Test 4 — Robustezza al blur")
        plot_blur(blur_df)

    # --- Test 5: JPEG ---
    if jpeg_df is not None:
        print("\n📊 Test 5 — Robustezza JPEG")
        plot_jpeg(jpeg_df)

    # --- Test 6: ROI ---
    if roi_df is not None:
        print("\n📊 Test 6 — ROI Strategies")
        plot_roi(roi_df)

    # --- Grafici cross-test ---
    if sweep_df is not None or delta_df is not None:
        print("\n📊 Grafici riassuntivi cross-test")
        plot_cross_test(sweep_df, delta_df)

    print(f"\n{'=' * 60}")
    print("GRAFICI COMPLETATI")
    print(f"{'=' * 60}")
    print(f"Output: {PLOT_DIR}")
    print()


if __name__ == "__main__":
    main()
