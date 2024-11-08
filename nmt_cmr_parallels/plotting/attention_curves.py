import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import seaborn as sns
import random
import os
import re
from pathlib import Path

from nmt_cmr_parallels.utils.evaluation_utils import (calculate_conditional_recall_probability, 
                                                  calculate_recall_probability,
                                                  calculate_first_recall_probability)

def extract_number(label):
    match = re.search(r'Epoch-(\d+)', label)
    return int(match.group(1)) if match else float('inf')

def extract_sort_keys(file_path):
    attention_type_match = re.search(r'/(Attention|NoAttention)-', file_path)
    dimension_match = re.search(r'-(\d+)Dim', file_path)

    attention_type = attention_type_match.group(1) if attention_type_match else ''
    dimension_number = int(dimension_match.group(1)) if dimension_match else 0

    return attention_type, dimension_number


bigfont = 26
smallfont = 20

# Create figure
fig = plt.figure(figsize=(22, 14))
plt.rcParams['font.family'] = 'Nimbus Roman'
gs = GridSpec(nrows=3, ncols=3, figure=fig)

module_directory = Path(__file__).resolve()
module_directory = module_directory.parents[1]
data_dir = os.path.join(module_directory,'resource','evaluations')

# First Row - Optimal CMR Comparison
end_position = 11
omit_first_k = 1
filepath = os.path.join(data_dir, "attention", "Epoch-9.json")

ax = fig.add_subplot(gs[0, 0])
ax.set_xlabel('Serial Position', fontsize=bigfont)
ax.set_ylabel('Recall Prob.', fontsize=bigfont)
ax.set_ylim((0.7,1.0))
ax.tick_params(axis='both', which='major', labelsize=smallfont)
with open(filepath,'r') as f:
    data = json.load(f)

if omit_first_k is not None:
    data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
    data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

recall_probabilities = calculate_recall_probability(data['original_sequences'], data['predicted_sequences'])
label = os.path.splitext(os.path.basename(filepath))[0]

if end_position is not None:
    positions = list(range(1, end_position))
    min_length = min(len(recall_probabilities), end_position - 1)
    ax.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', color = "#00441b",linewidth=3, markersize=6)

else:
    positions = list(range(1, len(recall_probabilities) + 1))
    ax.plot(positions, recall_probabilities, marker='o', label=label,linewidth=3, markersize=6)

ax.plot(positions, [0.95,0.935,0.91,0.89,0.90,0.91,0.92,0.978,0.98,1.0], marker='o', color="#FF7700",label="Optimal CMR", linewidth=3, markersize=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=smallfont)
ax.tick_params(axis='both', which='minor', labelsize=smallfont)
ax.text(-0.1, 1.25, 'A', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

##################### FRP #################

ax = fig.add_subplot(gs[0, 1])
# Plot probability of first recall vs serial position
ax.set_xlabel('Serial Position', fontsize=bigfont)
ax.set_ylabel('Prob. of \nFirst Recall', fontsize=bigfont)
ax.set_ylim((-0.04,1.0))
ax.tick_params(axis='both', which='major', labelsize=smallfont)
#ax.grid(True, which='both', linestyle='-', linewidth=1.5)
#plt.title('FRP for NMT Model and Optimal CMR', fontsize=23)
#ax.set_title('FRP for Amnesiac and Control Patients', fontsize=22)
with open(filepath,'r') as f:
    data = json.load(f)

if omit_first_k is not None:
    data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
    data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

recall_probabilities = calculate_first_recall_probability(data['original_sequences'], data['predicted_sequences'])
label = os.path.splitext(os.path.basename(filepath))[0]

if end_position is not None:
    positions = list(range(1, end_position))
    min_length = min(len(recall_probabilities), end_position - 1)
    ax.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', color="#00441b",linewidth=3, markersize=6)

else:
    positions = list(range(1, len(recall_probabilities) + 1))
    ax.plot(positions, recall_probabilities, marker='o', label=label,linewidth=3, color="#00441b", markersize=10)

ax.plot(positions, [0.97,0.02,0.03,0.01,0.02,0.02,0.03,0.02,0.01,0.0], marker='o', color="#FF7700",label="Optimal CMR",linewidth=3, markersize=6)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#ax.set_title('Free Recall Behavior for Attention Model and Optimal CMR', fontsize=24)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=smallfont)
ax.tick_params(axis='both', which='minor', labelsize=smallfont)
ax.text(-0.1, 1.25, 'B', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

######################## CRP #############################3
ax = fig.add_subplot(gs[0, 2])

# Plot crp vs lag
ax.set_xlabel('Lag', fontsize=bigfont)
ax.set_ylabel('Conditional \nResponse Prob.', fontsize=bigfont)
ax.legend()
ax.set_ylim((-0.04,1.0))
#ax.grid(True, which='both', linestyle='-', linewidth=1.5)
ax.tick_params(axis='both', which='major', labelsize=smallfont)
#plt.title('CRP for NMT Model and Optimal CMR', fontsize=27)
#ax.set_title('CRP for Amnesiac and Control Patients', fontsize=22)
with open(filepath,'r') as f:
    data = json.load(f)

if omit_first_k is not None:
    data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
    data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

probabilities, lags = calculate_conditional_recall_probability(data['original_sequences'], data['predicted_sequences'])
positive_lags, pos_idxs = [lag for lag in lags if lag > 0], [i for i, lag in enumerate(lags) if lag > 0]
positive_probs = [probabilities[idx] for idx in pos_idxs]
negative_lags, neg_idxs = [lag for lag in lags if lag < 0], [i for i, lag in enumerate(lags) if lag < 0]
negative_probs = [probabilities[idx] for idx in neg_idxs]

label = os.path.splitext(os.path.basename(filepath))[0]
random_color = (random.random(), random.random(), random.random())

# #Plotting negative lags
ax.plot(negative_lags, negative_probs, marker='o', color="#00441b", label='Seq2seq',linewidth=3, markersize=6)
# # Plotting positive lags
ax.plot(positive_lags, positive_probs, marker='o', color='#00441b',linewidth=3, markersize=6)

ax.plot([-4,-3,-2,-1], [0.03,0.01,0.06,0.08], marker='o', color='#FF7700', label='Rational CMR',linewidth=3, markersize=6)
ax.plot([1,2,3,4], [0.98,0.0,0.0,0.0], marker='o', color='#FF7700',linewidth=3, markersize=6)


ax.tick_params(axis='both', which='major', labelsize=smallfont)
ax.tick_params(axis='both', which='minor', labelsize=smallfont)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=smallfont)
ax.text(-0.1, 1.25, 'C', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')


###################################33
# Intermediate curves
end_position = 14
omit_first_k = 1
reverse_colors = False


results_dir = os.path.join(data_dir,"attention")
results_files = [os.path.join(results_dir, x) for x in os.listdir(results_dir) if x.endswith('.json')]
sorted_file_paths = sorted(results_files, key=extract_number)

ax = fig.add_subplot(gs[1, 0])
ax.set_xlabel('Serial Position', fontsize=bigfont)
ax.set_ylabel('Recall Prob.', fontsize=bigfont)
ax.set_ylim((-0.02,1.0))
#ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#plt.title('Recall Prob. vs. Serial Position', fontsize=23)
ax.tick_params(axis='both', which='major', labelsize=smallfont)
swap_transparency = True
alpha_counter = 1
for i, filepath in enumerate(sorted_file_paths):
    with open(filepath,'r') as f:
        data = json.load(f)

    if omit_first_k is not None:
        data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
        data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

    recall_probabilities = calculate_recall_probability(data['original_sequences'], data['predicted_sequences'])
    label = os.path.splitext(os.path.basename(filepath))[0]

    if end_position is not None:
        positions = list(range(1, end_position))
        min_length = min(len(recall_probabilities), end_position - 1)

        if label != 'Human Subject':
            if 'NoAttention' in filepath:
                no_attention_colors = ['#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f']
                if reverse_colors:
                    no_attention_colors = list(reversed(no_attention_colors))
                ax.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', color=no_attention_colors[i], label=label,linewidth=3, markersize=6)
            else:
                attention_colors = ['#ccece6','#99d8c9','#66c2a4','#41ae76','#238b45','#006d2c','#00441b']
                if reverse_colors:
                    attention_colors = list(reversed(attention_colors))
                ax.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', color=attention_colors[i], label=label,linewidth=3, markersize=6)
                
        else:
            ax.plot(positions[:min_length], recall_probabilities[:min_length],alpha=0.5, marker='o', color='black',linestyle='--', label=label,linewidth=3, markersize=2)
    else:
        positions = list(range(1, len(recall_probabilities) + 1))
        ax.plot(positions, recall_probabilities, marker='o',linewidth=3, markersize=6)

handles, labels = ax.get_legend_handles_labels()
sorted_handles_labels = sorted(zip(handles, labels), key=lambda t: extract_number(t[1]))
sorted_handles, sorted_labels = zip(*sorted_handles_labels)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.1, 1.25, 'D', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

############# FRP CURVE #############################
ax = fig.add_subplot(gs[1, 1])

# Plot probability of first recall vs serial position
ax.set_xlabel('Serial Position',fontsize=bigfont)
ax.set_ylabel('Prob. of \nFirst Recall',fontsize=bigfont)
ax.set_ylim((-0.02,1.0))
#ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=smallfont)
#plt.title('First Recall Prob. vs. Serial Position',fontsize=23)
swap_transparency = True
alpha_counter = 1
for i, filepath in enumerate(sorted_file_paths):
    with open(filepath,'r') as f:
        data = json.load(f)

    if omit_first_k is not None:
        data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
        data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

    recall_probabilities = calculate_first_recall_probability(data['original_sequences'], data['predicted_sequences'])
    label = os.path.splitext(os.path.basename(filepath))[0]

    if end_position is not None:
        positions = list(range(1, end_position))
        min_length = min(len(recall_probabilities), end_position - 1)
        if label != 'Human Subject':
            if 'NoAttention' in filepath:
                no_attention_colors = ['#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f']
                if reverse_colors:
                    no_attention_colors = list(reversed(no_attention_colors))
                ax.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', color=no_attention_colors[i], label=label,linewidth=3, markersize=6)
            else:
                attention_colors = ['#ccece6','#99d8c9','#66c2a4','#41ae76','#238b45','#006d2c','#00441b']
                if reverse_colors:
                    attention_colors = list(reversed(attention_colors))
                ax.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', color=attention_colors[i], label=label,linewidth=3, markersize=6)
        else:
            ax.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', alpha=0.5, color='black', linestyle='--',label=label,linewidth=3, markersize=2)

    else:
        positions = list(range(1, len(recall_probabilities) + 1))
        ax.plot(positions, recall_probabilities, alpha=0.5, marker='o', label=label)


handles, labels = ax.get_legend_handles_labels()
sorted_handles_labels = sorted(zip(handles, labels), key=lambda t: extract_number(t[1]))
sorted_handles, sorted_labels = zip(*sorted_handles_labels)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.1, 1.25, 'E', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

############# CRP CURVE #############################
ax = fig.add_subplot(gs[1, 2])

# Plot crp vs lag
ax.set_xlabel('Lag',fontsize=bigfont)
ax.set_ylabel('Conditional \nResponse Prob.',fontsize=bigfont)
ax.legend()
#ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=smallfont)
#plt.title('Conditional Recall Prob. vs. Lag',fontsize=23)
swap_transparency = True
alpha_counter = 1
for i,filepath in enumerate(sorted_file_paths):
    with open(filepath,'r') as f:
        data = json.load(f)

    if omit_first_k is not None:
        data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
        data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

    probabilities, lags = calculate_conditional_recall_probability(data['original_sequences'], data['predicted_sequences'])
    positive_lags, pos_idxs = [lag for lag in lags if lag > 0], [i for i, lag in enumerate(lags) if lag > 0]
    positive_probs = [probabilities[idx] for idx in pos_idxs]
    negative_lags, neg_idxs = [lag for lag in lags if lag < 0], [i for i, lag in enumerate(lags) if lag < 0]
    negative_probs = [probabilities[idx] for idx in neg_idxs]

    label = os.path.splitext(os.path.basename(filepath))[0]
    random_color = (random.random(), random.random(), random.random())

    # Plotting negative lags
    if label != 'Human Subject':
        if 'NoAttention' in filepath:
            no_attention_colors = ['#d4b9da','#c994c7','#df65b0','#e7298a','#ce1256','#980043','#67001f']
            if reverse_colors:
                    no_attention_colors = list(reversed(no_attention_colors))
            negative_probs = [0.0 for x in negative_probs]
            ax.plot(negative_lags, negative_probs, marker='o', color=no_attention_colors[i], label=label,linewidth=3, markersize=6)
            ax.plot(positive_lags, positive_probs, marker='o', color=no_attention_colors[i],linewidth=3, markersize=6)
        else:
            attention_colors = ['#ccece6','#99d8c9','#66c2a4','#41ae76','#238b45','#006d2c','#00441b']
            if reverse_colors:
                    attention_colors = list(reversed(attention_colors))
            ax.plot(negative_lags, negative_probs, marker='o', color=attention_colors[i], label=label,linewidth=3, markersize=6)
            ax.plot(positive_lags, positive_probs, marker='o', color=attention_colors[i],linewidth=3, markersize=6)
    else:
        ax.plot(positive_lags, positive_probs, marker='o', color='black',linestyle='--',linewidth=3, alpha=0.5, markersize=2)
        ax.plot(negative_lags, negative_probs, marker='o', color='black', linestyle='--',label=label, alpha=0.5,linewidth=3, markersize=2)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='upper left', bbox_to_anchor=(1, 1.2),fontsize=smallfont)
ax.text(-0.1, 1.25, 'F', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

####################################################################

# Attention model behavior

# no_attention_dir = "/media/ouranos/bulkstorage/phd_work/seq_recall_runs/final_trials/IntermediateNoAttention/evaluations/"
# attention_dir = "/media/ouranos/bulkstorage/phd_work/seq_recall_runs/final_trials/attention_intermediate_checkpoints/selected_evaluations/"
# hidden_dim_dir = "/media/ouranos/bulkstorage/phd_work/seq_recall_runs/final_trials/hiddendim-attention-noattencomparison/evaluations/"
no_attention_dir = os.path.join(data_dir,'no_attention')
attention_dir = os.path.join(data_dir,'attention')
hidden_dim_dir = os.path.join(data_dir,'varying_hidden_dim')
no_attention_files = [os.path.join(no_attention_dir, x) for x in os.listdir(no_attention_dir) if x.endswith('.json')]
attention_files = [os.path.join(attention_dir, x) for x in os.listdir(attention_dir) if x.endswith('.json') and 'Human' not in x]
hidden_dim_files = [os.path.join(hidden_dim_dir, x) for x in os.listdir(hidden_dim_dir) if x.endswith('.json')]

ax = fig.add_subplot(gs[2, 0])

# Collect data
omit_first_k = 1
recency_metric = []
for filepath in attention_files:
    with open(filepath, 'r') as f:
        results = json.load(f)
    if omit_first_k > 0:
        results['original_sequences'] = [x[omit_first_k:] for x in results['original_sequences']]
        results['predicted_sequences'] = [x[omit_first_k:] for x in results['predicted_sequences']]
    frp = calculate_first_recall_probability(results['original_sequences'], results['predicted_sequences'])
    recency_metric.append((extract_number(filepath), np.mean(frp[-3:])))
recency_metric = sorted(recency_metric, key=lambda x: x[0])
epochs = [x[0] for x in recency_metric]
recency_metric = [x[1] for x in recency_metric]

ax.set_ylabel('Recency', fontsize=bigfont)
ax.set_ylim((-0.01,0.2))
ax.tick_params(axis='both', which='major', labelsize=smallfont)
ax.plot(epochs, recency_metric, marker='o', color="#08519c",linewidth=3, markersize=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Training Epoch',fontsize=bigfont)

ax.text(-0.1, 1.25, 'G', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')
# 2b - ATTENTION PRIMACY PLOT
# Collect data
ax = fig.add_subplot(gs[2, 1])
omit_first_k = 1
primacy_metric = []
for filepath in attention_files:
    with open(filepath, 'r') as f:
        results = json.load(f)
    if omit_first_k > 0:
        results['original_sequences'] = [x[omit_first_k:] for x in results['original_sequences']]
        results['predicted_sequences'] = [x[omit_first_k:] for x in results['predicted_sequences']]
    frp = calculate_first_recall_probability(results['original_sequences'], results['predicted_sequences'])
    primacy_metric.append((extract_number(filepath), np.mean(frp[:1])))
primacy_metric = sorted(primacy_metric, key=lambda x: x[0])
epochs = [x[0] for x in primacy_metric]
primacy_metric = [x[1] for x in primacy_metric]

ax.set_ylabel('Primacy', fontsize=bigfont)
ax.set_ylim((-0.01,1.0))
ax.tick_params(axis='both', which='major', labelsize=smallfont)
ax.plot(epochs, primacy_metric, marker='o', color="#08519c",linewidth=3, markersize=6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Training Epoch',fontsize=bigfont)
ax.text(-0.1, 1.25, 'H', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

# 2c - ATTENTION CRP(-1) PLOT
ax = fig.add_subplot(gs[2, 2])

# Collect data
omit_first_k = 1
backcont = []
for filepath in attention_files:
    with open(filepath, 'r') as f:
        results = json.load(f)
    if omit_first_k > 0:
        results['original_sequences'] = [x[omit_first_k:] for x in results['original_sequences']]
        results['predicted_sequences'] = [x[omit_first_k:] for x in results['predicted_sequences']]
    probs, lags = calculate_conditional_recall_probability(results['original_sequences'], results['predicted_sequences'])
    neg1_index = lags.index(-1)
    backcont.append((extract_number(filepath), probs[neg1_index]))
backcont = sorted(backcont, key=lambda x: x[0])
epochs = [x[0] for x in backcont]
backcont = [x[1] for x in backcont]

#ax.set_xlabel('Training Epoch', fontsize=22)
ax.set_ylabel('Backward Contiguity', fontsize=bigfont)
ax.set_ylim((-0.01,0.2))
ax.tick_params(axis='both', which='major', labelsize=smallfont)
ax.plot(epochs, backcont, marker='o', color="#08519c",linewidth=3, markersize=6)

#fig.suptitle('Evolution of Attention Model Behavior', fontsize=38)
#fig.text(0.5, -0.06, 'Training Epoch', ha='center', fontsize=38)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Training Epoch',fontsize=bigfont)
ax.text(-0.1, 1.25, 'I', transform=ax.transAxes, 
            fontsize=bigfont, fontweight='bold', va='top', ha='right')

# Adjust spacing between plots
plt.subplots_adjust(hspace=0.7, wspace=0.5)

# Display plot
plt.savefig("attentionbehaviorcurves.png", dpi=300, bbox_inches='tight')
