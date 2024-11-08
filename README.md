## Parallels between NMT and CMR

This package provides scripts and tools for training and evaluating a seq2seq model of free recall, fitting models to individual subject data, and analyzing results using visualizations for the paper *Uncovering Parallels between Neural Machine Translation and Human Memory Search through Cognitive Modeling*. Model Checkpoints/Data used to produce all plots in the paper can be found in the "resource" folder.

---

## Table of Contents
- [Installation](#installation)
- [Script Overview and Usage Examples](#scripts-overview)
  - [train_free_recall.py](#train_free_recallpy)
  - [train_rl_recall.py](#train_rl_recallpy)
  - [eval_free_recall.py](#eval_free_recallpy)
  - [individual_subject_fitting.py](#individual_subject_fittingpy)
  - [eval_individual_fits.py](#eval_individual_fitspy)
  - [plot_analysis.py](#plot_analysispy)
  - [plotting/ablationstudy_comparison.py](#plotting/ablationstudy_comparisonpy)
  - [plotting/attention_curves.py](#plotting/attention_curvespy)
  - [plotting/attention_heatmaps.py](#plotting/attention_heatmapspy) 
  - [plotting/plot_individual_fits.py](#plotting/plot_individual_fitspy)
---

## Installation

Clone the repository and install the dependencies using the following commands (a Conda environment is recommended):

```bash
git clone https://github.com/nds113/nmt_cmr_parallels.git
cd nmt_cmr_parallels
conda env create -f nmtcmr.yml
conda activate nmtcmr
pip install -r requirements.txt
pip install .
```

## Script Overview and Usage Examples

### train_free_recall.py
Trains a seq2seq model on a free recall task.  '--peers_vocab' indicates that sequences should only be generated using words from the PEERS dataset. '--seq_tokens' indicates that special start and end of sequence tokens should be added to the model's vocabulary.

**Usage:**
```bash
python -m nmt_cmr_parallels.train_seq_recall --epochs 100 --rnn_mode GRU --attention_type luong --hidden_dim 128 --lr 0.001 --log_dir attention_128dim --checkpoint_path attention_128dim.pt --sequence_length 14 --peers_vocab --dropout 0.1 -v --num_sequences 1000 --seq_tokens

```

### train_rl_recall.py
RL training script using a PPO approach to fine-tune pre-trained seq2seq models. '--actor_load_path' should point to a seq2seq model previously trained using the 'train_free_recall.py' script.

**Usage:**
```bash
python -m nmt_cmr_parallels.train_rl_recall --rnn_mode GRU --attention_type luong --exp-name testrl --num-envs 4 --seq_tokens -v --actor_load_path attention_128dim/attention_128dim.pt 

```

### eval_free_recall.py
Evaluates a trained model on free recall.

**Usage:**
```bash
python -m nmt_cmr_parallels.eval_free_recall --checkpoint_path attention_128dim/attention_128dim.pt  --results_path attention_128dim/evaluation.json --num_sequences 1000 --sequence_length 14 --peers_vocab

```

### individual_subject_fitting.py
Script that fits a seq2seq model to trials for each individual subject found in the PEERS dataset. '--augmentation_factor' indicates the multiplier by which the trial data should be augmented (i.e. 10 = 10x the number of trials), which is accomplished using random substitution of semantically similar words in the given trials.

**Usage:**
```bash
python -m nmt_cmr_parallels.individual_subject_fitting --logging_path individual_fits --augmentation_factor 10 --epochs 200 --lr 0.01
```

### eval_individual_fits.py
Script to evaluate all trained models fit to individual subjects. Free recall behavior plots are automatically generated per subject as well as group-wide plots for comparing subject fitting.

**Usage:**
```bash
python -m nmt_cmr_parallels.eval_individual_fits --root_dir individual_fits/

```

### plot_analysis.py
Plot free recall behavior curves using previously generated model evaluation files.

**Usage:**
```bash
python -m nmt_cmr_parallels.plot_analysis --results_files attention_128dim/evaluation.json --output_log attention_freerecall_behavior/

```

### plotting/ablationstudy_comparison.py
Script for reproducing ablation study plots from collected data.

**Usage:**
```bash
python -m nmt_cmr_parallels.plotting.ablationstudy_comparison
```

### plotting/attention_curves.py
Script for reproducing attention curve plots from collected data.

**Usage:**
```bash
python -m nmt_cmr_parallels.plotting.attention_curves.py
```

### plotting/attention_heatmaps.py
Script for reproducing attention heatmap plots from collected data.

**Usage:**
```bash
python -m nmt_cmr_parallels.plotting.attention_heatmaps.py
```

### plotting/plot_individual_fits.py
Script for reproducing behavior curves for individual subjects from collected data.

**Usage:**
```bash
python -m nmt_cmr_parallels.plotting.plot_individual_fits.py
```