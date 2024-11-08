import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns
from scipy.stats import mannwhitneyu
import os
from pathlib import Path

def calculate_rank_biserial_correlation(data1, data2, u_stat):

    n1 = len(data1)
    n2 = len(data2)
    rank_biserial_correlation = (2 * u_stat) / (n1 * n2) - 1

    return rank_biserial_correlation

module_directory = Path(__file__).resolve()
module_directory = module_directory.parents[1]
data_dir = os.path.join(module_directory,'resource','evaluations')

seq2seq_path = os.path.join(data_dir,"individual_fits","seq2seq_behavior_curve_rmses.json")
cmr_path = os.path.join(data_dir,"individual_fits","cmr_behavior_curve_rmses.json")

bigfont = 26
smallfont = 20

with open(seq2seq_path, 'r') as f:
    seq2seq_results = json.load(f)
with open(cmr_path, 'r') as f:
    cmr_results = json.load(f)

# Create figure
fig = plt.figure(figsize=(22, 14))
plt.rcParams['font.family'] = 'Nimbus Roman'
gs = GridSpec(nrows=4, ncols=12, figure=fig, width_ratios=[1 for _ in range(12)], height_ratios=[1,1,1,2])

# Plot individual SPC curves
human_data = cmr_results['human_spcs']
cognitive_model_data = cmr_results['cmr_spcs']
encoder_decoder_model_data = seq2seq_results['seq2seq_spcs']
serial_positions = np.arange(1, 17)
for i in range(12):
    ax = fig.add_subplot(gs[0, i])

    ax.plot(serial_positions, human_data[i], label='Human', color='#96031A', linestyle='-')
    ax.plot(serial_positions, cognitive_model_data[i], label='CMR', color="#FAA916", linestyle='--')
    ax.plot(serial_positions, encoder_decoder_model_data[i], label='Seq2Seq', color="#04A777", linestyle='-.')
    ax.set_xticks([1, 8, 16])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(['1', '8', '16'],fontsize=smallfont)
    ax.set_yticklabels(['0', '0.5', '1'],fontsize=smallfont)

    if i != 0:  # Hide y-axis labels except on the first plot
        ax.set_yticklabels([])
    if i == 0:
        ax.set_ylabel('Recall\n Prob.', ha='center', fontsize=bigfont)
        ax.text(-0.73, 1.35, 'A', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')
    if i == 7:
        ax.legend(['Human', 'CMR', 'Seq2Seq'],bbox_to_anchor=(10,1.1), fontsize=smallfont)
stat, p_value = mannwhitneyu(cmr_results['spc_rmses'], seq2seq_results['spc_rmses'])
rank_corr = calculate_rank_biserial_correlation(cmr_results['spc_rmses'], seq2seq_results['spc_rmses'], stat)
print(f"SPC Stat: {stat}, P-Value: {p_value}, Rank Biserial Correlation: {rank_corr}")

# Plot PFR Curves
human_data = cmr_results['human_pfrs']
cognitive_model_data = cmr_results['cmr_pfrs']
encoder_decoder_model_data = seq2seq_results['seq2seq_frps']
for i in range(12):
    ax = fig.add_subplot(gs[1, i])
    ax.plot(serial_positions, human_data[i], label='Human', color='#96031A', linestyle='-')
    ax.plot(serial_positions, cognitive_model_data[i], label='CMR', color="#FAA916", linestyle='--')
    ax.plot(serial_positions, encoder_decoder_model_data[i], label='Seq2Seq Model', color="#04A777", linestyle='-.')
    ax.set_xticks([1, 8, 16])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(['1', '8', '16'],fontsize=smallfont)
    ax.set_yticklabels(['0', '0.5', '1'],fontsize=smallfont)
    if i != 0:  # Hide y-axis labels except on the first plot
        ax.set_yticklabels([])
    if i == 0:
        ax.set_ylabel('Prob. of\n First Recall', ha='center', fontsize=bigfont)
        ax.text(-0.73, 1.35, 'B', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

stat, p_value = mannwhitneyu(cmr_results['crp_rmses'], seq2seq_results['crp_rmses'])
rank_corr = calculate_rank_biserial_correlation(cmr_results['crp_rmses'], seq2seq_results['crp_rmses'], stat)
print(f"PFR Stat: {stat}, P-Value: {p_value}, Rank Biserial Correlation: {rank_corr}")

# Plot CRP curves
human_data = cmr_results['human_crps']
cognitive_model_data = cmr_results['cmr_crps']
seq2seq_neg = seq2seq_results['seq2seq_neg_crps']
seq2seq_pos =seq2seq_results['seq2seq_pos_crps']
for i in range(12):
    ax = fig.add_subplot(gs[2, i])
    zero_indx = cognitive_model_data[i].index(0.0)
    neg_cmr = cognitive_model_data[i][:zero_indx]
    pos_cmr = cognitive_model_data[i][zero_indx+1:]
    neg_human = human_data[i][:zero_indx]
    pos_hum = human_data[i][zero_indx+1:]

    ax.plot(np.arange(-4,0), neg_human, label='Human', color='#96031A', linestyle='-')
    ax.plot(np.arange(1,5), pos_hum, color='#96031A', linestyle='-')
    ax.plot(np.arange(-4,0), neg_cmr, label='CMR', color="#FAA916", linestyle='--')
    ax.plot(np.arange(1,5), pos_cmr, color="#FAA916", linestyle='--')

    ax.plot(np.arange(-4,0), seq2seq_neg[i][-4:], label='Seq2Seq', color="#04A777", linestyle='-.')
    ax.plot(np.arange(1,5), seq2seq_pos[i][:4], color="#04A777", linestyle='-.')

    ax.set_xticks([-4, 4])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['-4', '4'], fontsize=smallfont)
    ax.set_yticklabels(['0', '1'], fontsize=smallfont)

    if i == 0:
        ax.set_ylabel('Conditional\n Response Prob.', ha='center', fontsize=bigfont)
        ax.text(-0.73, 1.35, 'C', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

    if i != 0:  # Hide y-axis labels except on the first plot
        ax.set_yticklabels([])

stat, p_value = mannwhitneyu(cmr_results['pfr_rmses'], seq2seq_results['pfr_rmses'])
rank_corr = calculate_rank_biserial_correlation(cmr_results['pfr_rmses'], seq2seq_results['pfr_rmses'], stat)
print(f"CRP Stat: {stat}, P-Value: {p_value}, Rank Biserial Correlation: {rank_corr}")

# Plot 4th row with 2 histograms
for i in range(1):
    ax = fig.add_subplot(gs[3, 0:4])
    ax = sns.histplot(cmr_results['spc_rmses'], label='CMR', stat='density', alpha=0.5)
    ax = sns.kdeplot(cmr_results['spc_rmses'], linewidth=3)
    ax = sns.histplot(seq2seq_results['spc_rmses'], label='Seq2Seq', stat='density', alpha=0.5)
    ax = sns.kdeplot(seq2seq_results['spc_rmses'], linewidth=3)
    ax.set_xlabel('RMSE', fontsize=bigfont)
    ax.set_ylabel('Count', fontsize=bigfont)
    ax.tick_params(axis='x', labelsize=smallfont)
    ax.tick_params(axis='y', labelsize=smallfont)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Serial Position Curve', fontsize=bigfont)
    ax.set_ylabel('')
    ax.text(-0.13, 1.05, 'D', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

# Plot 5th row with 2 wider plots
for i in range(2):
    ax = fig.add_subplot(gs[3, 4*(i+1):4*(i+2)])
    ax.set_xlabel('Count')
    if i ==0:

        ax = sns.histplot(cmr_results['crp_rmses'], label='CMR', stat='density', alpha=0.5)
        ax = sns.kdeplot(cmr_results['crp_rmses'], linewidth=3)
        ax = sns.histplot(seq2seq_results['crp_rmses'], label='Seq2Seq', stat='density', alpha=0.5)
        ax = sns.kdeplot(seq2seq_results['crp_rmses'], linewidth=3)
        ax.set_xlabel('RMSE', fontsize=bigfont)
        ax.tick_params(axis='x', labelsize=smallfont)
        ax.tick_params(axis='y', labelsize=smallfont)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        ax.set_title('Probability of First Recall', fontsize=bigfont)
        ax.text(-0.13, 1.05, 'E', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')
    else:
        ax = sns.histplot(cmr_results['pfr_rmses'], label='CMR', stat='density', alpha=0.5)
        ax = sns.kdeplot(cmr_results['pfr_rmses'], linewidth=3)
        ax = sns.histplot(seq2seq_results['pfr_rmses'], label='Seq2Seq', stat='density', alpha=0.5)
        ax = sns.kdeplot(seq2seq_results['pfr_rmses'], linewidth=3)
        ax.set_xlabel('RMSE', fontsize=bigfont)
        ax.tick_params(axis='x', labelsize=smallfont)
        ax.tick_params(axis='y', labelsize=smallfont)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Conditional Response Probability ', fontsize=bigfont)
        ax.text(-0.13, 1.05, 'F', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')
        ax.legend(fontsize=smallfont)
    ax.set_ylabel('')
    
# Adjust spacing between plots
plt.subplots_adjust(hspace=0.5, wspace=0.55)

# Display plot
plt.savefig("humanfit_comparison.png", dpi=300, bbox_inches='tight')

