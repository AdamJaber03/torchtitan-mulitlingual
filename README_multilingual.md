# Multilingual Pretraining Extensions for TorchTitan

This fork of TorchTitan has been modified to support **extensive multilingual pretraining experiments**, including:

* curriculum and multi-stage training
* dataset mixing
* code-switching augmentations
* scheduled augmentation parameters
* contrastive objectives
* fictional entity injection for recall evaluation

---

# Getting Started

## 1. Download the Dataset

Start by downloading the multilingual `fineweb-edu-ar` dataset using:

```bash
python ./data_download/fineweb-edu-ar_ar&en_download.py
```

---

## 2. Update Dataset Paths

Update dataset paths inside:

```text
./torchtitan/hf_datasets/text_datasets.py
```

Specifically, modify the `_load_dataset()` function to point to your local dataset locations.

---

## 3. Run the First Training

Recommended first run:

```bash
CONFIG="smollm2_360m_flex_curriculum_barebones" \
NGPU=8 \
MODULE="llama3" \
./run_train.sh
```

To understand the overall pipeline and code structure, follow the execution flow from:

```text
train.py -> trainer.py
```

This should give you a good high-level overview of:

* training orchestration
* dataset flow
* curriculum stages
* augmentation pipeline
* distributed setup
* loss handling

---

# Model Architecture Notes

Most experiments in this repository use **Llama3-style architectures**, mainly in terms of the **attention mechanism and overall design philosophy** (not necessarily exact layer counts or hidden dimensions).

Relevant configs can be found at:

```text
./torchtitan/models/llama3/config_registry.py
```

---

# Recommended Experiments

## English1 / English2 Experiment

A second useful config to explore is:

```text
smollm2_360m_en1_en2
```

This experiment trains a model on **two copies of the same language**.

### Training Structure

#### Stage 1

* Includes **code-switching**
* Encourages shared representations and generalization

#### Stage 2

* Uses **disjoint training**

### Interesting Failure Case

Try:

* commenting out Stage 1
* increasing Stage 2 duration

You should observe the model struggling to generalize properly.

---

# Major Additions to the Codebase

---

## 1. Multi-Source and Multi-Stage Training

The codebase supports:

### Curriculum Training

Stages run sequentially.

### Dataset Mixing

Multiple datasets can be sampled in parallel within a stage.

Example:

```text
smollm2_360m_en1_en2
```

### Important Rules

* Mixed dataset weights per stage must sum to `1`
* Total stage steps must equal `num_training_steps` as defined in the config.

---

## 2. Augmentations & Post-Tokenization Augmentations

Located in:

```text
./torchtitan/hf_datasets
```

### Design

Augmentations are implemented as classes.

Each augmentation:

* defines parameters in the constructor
* implements a `forward()` method

The `forward()` method receives a dictionary containing:

* text/tokens depending if its a pre/post tokenization augmentation
* metadata
* outputs from previous augmentations

### Execution Order

Augmentations are applied sequentially in the exact order defined in the config.

### Scope

Augmentations can be defined:

* per training stage
* per dataset source

---

## 3. Scheduled Augmentation Parameters

Available schedulers:

```text
./torchtitan/hf_datasets/value_schedualers.py
```

Example usage:

* `WordwiseUnigramCodeSwitching.replace_prob`

This allows augmentation parameters to change dynamically during training.

### Current Limitation

Due to distributed training constraints, schedulable attributes currently need to be predefined manually.

(Planned to improve in future revisions.)

---

## 4. Fictional Entity Data Pipeline

Located in:

```text
./fictional_entity_data
```

This pipeline creates fictional entities that can be injected into training data for:

* knowledge recall evaluation
* memorization experiments
* probing studies

### Available Datasets

#### General Fictional Entities

* 2080 entities
* English + Arabic
* includes MCQs (multiple-choice questions)

```text
https://drive.google.com/file/d/1vAngjliVnosUjurqK3h2avDAFaCfRkRs/view
```

#### Human Fictional Entities

* 2080 entities
* English only
* includes MCQs

```text
https://drive.google.com/file/d/1pA_Y8yizW9l-I2UUC9uQ658wNXVm0GYB/view
```

---

## 5. Dictionary Creation / Code-Switching Tools

Located in:

```text
./code_switching_tools
```

Contains tooling for:

* word-level mixing
* translation
* bilingual dictionary generation
* code-switching dataset creation

---

## 6. Multi-Loss Support

Loss handling has been extended to support:

* multiple simultaneous objectives
* weighted losses

Example config:

```text
smollm2_360m_flex_curriculum_contrastive
```

---

## 7. Contrastive Loss Support

This is currently somewhat experimental and could use cleanup/refactoring.

### Overview

The dataset can provide:

* two sequences
* a mapping mask

The mapping mask specifies corresponding subsequences between `seq1` and `seq2`

The pipeline then:

1. pools subsequence representations
2. applies an MLP projection
3. sends paired representations to the loss function

This enables contrastive learning between aligned subsequences.

### Example Config

```text
smollm2_360m_flex_curriculum_contrastive
```

If you plan to use or extend this functionality, expect several moving parts. A cleaner abstraction is planned in future revisions.
