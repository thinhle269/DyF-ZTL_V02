import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

# v3 utils (recovered from update_project_v3.py / utils.cpython-313.pyc) plus the
# two functions that only survived in bytecode (plot_efficiency,
# save_confusion_matrix_csv), plus one metrics extension: the classical IDS
# false-alarm rate (share of benign samples flagged as any attack), reported
# alongside the original macro one-vs-rest FPR that the paper calls "FAR".

sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'

NORMAL_CLASS_INDEX = 3  # LabelEncoder alphabetical order: ddos, dos, injection, normal, ...

def calculate_extended_metrics(y_true, y_pred, method_name, normal_idx=NORMAL_CLASS_INDEX):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    with np.errstate(divide='ignore', invalid='ignore'):
        far_per_class = FP / (FP + TN)
        far_per_class = np.nan_to_num(far_per_class)

    avg_far = np.mean(far_per_class)

    # Classical IDS FAR: fraction of benign (normal) samples predicted as any attack
    if normal_idx is not None and normal_idx < cm.shape[0]:
        normal_row = cm[normal_idx]
        far_binary = 1.0 - (normal_row[normal_idx] / normal_row.sum()) if normal_row.sum() > 0 else 0.0
    else:
        far_binary = np.nan

    metrics_dict = {
        'Method': method_name,
        'Accuracy': round(accuracy * 100, 2),
        'Detection Rate (Recall)': round(recall * 100, 2),
        'Precision': round(precision * 100, 2),
        'F1-Score': round(f1 * 100, 2),
        'False Alarm Rate (FAR)': round(avg_far * 100, 2),
        'FAR Benign->Attack': round(far_binary * 100, 2)
    }
    return metrics_dict, cm

def save_results(results_dict, filename="experiment_results.csv"):
    df = pd.DataFrame(results_dict)
    df.to_csv(f"results/{filename}", index=False)
    print(f"[INFO] Results saved to results/{filename}")

def save_confusion_matrix_csv(cm, classes, method_name, out_dir="results"):
    df = pd.DataFrame(cm, index=classes, columns=classes)
    df.to_csv(f"{out_dir}/Confusion_Matrix_{method_name}.csv")

# --- PLOTTING (v3, unchanged behavior; out_dir made configurable) ---

def plot_confusion_matrix(cm, classes, method_name, out_dir="results"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {method_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Confusion_Matrix_{method_name}.png", dpi=300)
    plt.close()

def plot_performance_metrics(df, out_dir="results"):
    if df is None: return
    metrics = ['Accuracy', 'Detection Rate (Recall)', 'Precision', 'F1-Score']
    df_melted = df.melt(id_vars=['Method'], value_vars=[m for m in metrics if m in df.columns],
                        var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Metric', y='Score', hue='Method', data=df_melted, palette="viridis")
    plt.title("Comparative Performance Analysis", fontsize=16, fontweight='bold')
    plt.ylim(80, 105)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, title='')

    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Fig2_Performance_Metrics.png", dpi=300)
    plt.close()

def plot_efficiency(df, out_dir="results"):
    # Recovered from utils.cpython-313.pyc: two-panel FAR + training-time chart
    if df is None: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(x='Method', y='False Alarm Rate (FAR)', data=df, palette="rocket", ax=axes[0])
    axes[0].set_title("False Alarm Rate Comparison", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("FAR (%)")
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=11, xytext=(0, 3), textcoords='offset points')

    if 'Training Time (s)' in df.columns:
        sns.barplot(x='Method', y='Training Time (s)', data=df, palette="mako", ax=axes[1])
        axes[1].set_title("Training Time Comparison", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Time (seconds)")
        for p in axes[1].patches:
            axes[1].annotate(f'{p.get_height():.1f}s', (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', fontsize=11, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f"{out_dir}/Fig3_Efficiency_Analysis.png", dpi=300)
    plt.close()

def plot_trust_evolution(df, out_dir="results"):
    if df is None: return
    plt.figure(figsize=(12, 7))
    rounds = df['Round']
    threshold = df['Threshold']

    plt.plot(rounds, threshold, label='Adaptive Threshold', color='black', linestyle='--', linewidth=2.5, alpha=0.8)

    client_cols = [c for c in df.columns if 'Client_' in c]
    labeled_good, labeled_bad, labeled_suspicious = False, False, False

    for client in client_cols:
        score_trend = df[client]
        final_score = score_trend.iloc[-1]

        if final_score < 0.2:
            label = 'Malicious Clients (Blocked)' if not labeled_bad else ""
            plt.plot(rounds, score_trend, color='red', alpha=0.8, linewidth=2, label=label)
            labeled_bad = True
        elif final_score > 0.8:
            label = 'Trusted Clients' if not labeled_good else ""
            plt.plot(rounds, score_trend, color='green', alpha=0.3, linewidth=1, label=label)
            labeled_good = True
        else:
            label = 'Low-Quality Clients' if not labeled_suspicious else ""
            plt.plot(rounds, score_trend, color='orange', linestyle=':', linewidth=2, label=label)
            labeled_suspicious = True

    plt.title("Zero Trust Dynamics: Isolation of Malicious Nodes", fontsize=16, fontweight='bold')
    plt.xlabel("Communication Rounds")
    plt.ylabel("Trust Score")
    plt.axhline(y=0.4, color='gray', linestyle='-', alpha=0.3)
    plt.legend(loc='center right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Fig4_Trust_Evolution.png", dpi=300)
    plt.close()

def plot_robustness(df, out_dir="results"):
    if df is None: return
    df_melted = df.melt(id_vars=['Poison_Ratio'], var_name='Method', value_name='Accuracy')
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Poison_Ratio', y='Accuracy', hue='Method', data=df_melted,
                 style='Method', markers=True, dashes=False, linewidth=2.5, markersize=9)
    plt.title("Robustness against Poisoning Attacks (0% to 90%)", fontsize=14, fontweight='bold')
    plt.xlabel("Poisoning Ratio")
    plt.ylabel("Global Accuracy (%)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/Fig5_Robustness_Total.png", dpi=300)
    plt.close()

def plot_convergence(results_dict, title="Convergence Analysis", out_dir="results"):
    plt.figure(figsize=(10, 6))
    for method, acc_list in results_dict.items():
        plt.plot(acc_list, label=method, linewidth=2)
    plt.title(title)
    plt.xlabel('Communication Rounds')
    plt.ylabel('Global Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_dir}/convergence_{title.replace(' ', '_')}.png", dpi=300)
    plt.close()
