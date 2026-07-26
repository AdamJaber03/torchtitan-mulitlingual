# Sentence-Parallel en1/en2 Experiments

This experiment family tests whether learning an explicit translation relation
between two disjoint token spaces improves factual knowledge transfer.

## Controlled setup

The tokenizer vocabulary is duplicated:

- en1 uses token IDs `0..65535`;
- en2 uses the corresponding IDs shifted by `65536`;
- both token spaces decode to the same English surface strings.

The model flavor is `smollm2_360m_2xvocab`. Input/output embeddings contain
131,072 rows; the transformer body is shared.

The three configs are:

```text
smollm2_360m_en1_en2_sentence_wise_code_switching
smollm2_360m_en1_en2_sentence_parallel_doc_order
smollm2_360m_en1_en2_sentence_parallel_sentence_order
```

`sentence_wise_code_switching` emits each sentence once in a randomly chosen
token space. Both parallel modes emit every sentence twice, once in each token
space. Document order chooses one direction for the document; sentence order
chooses the direction independently for each sentence. The resulting text is
tokenized and packed into ordinary 2,048-token training sequences.

## Curriculum and fictional facts

The default run contains:

1. 3,000 intervention steps;
2. 1,000 clean steps.

`EN1_EN2_MIXED_DATA_FRACTION` is the stage-1 source weight for the augmented
source. The remaining stage-1 weight is split equally between clean en1 and en2
sources. Stage 2 contains only the two clean sources with weight 0.5 each.

The tested stage-1 fractions are `0.01`, `0.10`, and `0.60`. Their full-run
mixed shares are 0.75%, 7.5%, and 45%, respectively.

By default, training uses Gemini entity IDs `0..2079`, shuffled with
`random.Random(43)`. The ordered entities form the complete 4×4 grid of en1/en2
target counts `[0, 20, 100, 1000]`.

Injection probabilities are calculated separately for each stage from:

- the stage’s per-language clean-source token budget;
- measured base/injection document lengths;
- the target exposure count assigned to the stage;
- the empirical `1/1.04` probability readjustment.

For stage 1, one language source receives
`steps × global_batch × seq_len × (1-m)/2` tokens. For stage 2 it receives
`steps × global_batch × seq_len × 0.5` tokens. Target counts are divided between
the stages in proportion to their lengths.

Optional human-entity support uses a separate NumPy RNG with seed 48. It is
disabled unless `EN1_EN2_INCLUDE_HUMAN_ENTITIES=1`.

## Setup

From the repository root:

```bash
uv sync
python -m nltk.downloader \
  -d ../nltk_data \
  punkt punkt_tab
```

The default workspace layout is:

```text
../data/fineweb_translated/en-original/
../fictional_entity_data/gemini_seeds/<entity-id>/en_data.jsonl
../assets/65k_paired/
../nltk_data/tokenizers/
../outputs/torchtitan/
../evals/knowledge_sharing/
```

All roots can be overridden:

```text
MULTILINGUAL_PRETRAINING_ROOT
FINEWEB_TRANSLATED_ROOT
FICTIONAL_ENTITY_DATA_ROOT
EN1_EN2_HF_ASSETS_PATH
EN1_EN2_OUTPUT_ROOT
NLTK_DATA
HF_HOME
TORCH_HOME
```

Knowledge evaluation additionally requires `lm-evaluation-harness` with its
`lm_eval` executable available in the active environment.

## Preflight and training

Validate one effective config without starting training:

```bash
EN1_EN2_MIXED_DATA_FRACTION=0.10 \
EN1_EN2_RUN_TAG=injcountfix_gemini_s43 \
.venv/bin/python scripts/inspect_en1_en2_sentence_config.py \
  --config smollm2_360m_en1_en2_sentence_parallel_sentence_order
```

Run the launcher’s filesystem precheck:

```bash
VARIANT=sentence_parallel_sentence_order \
MIXED_DATA_FRACTION=0.10 \
EN1_EN2_RUN_TAG=injcountfix_gemini_s43 \
PRECHECK_ONLY=1 \
bash en1_en2_sentence_train.slurm
```

Submit one fraction’s three training modes:

```bash
MIXED_DATA_FRACTION=0.10 \
EN1_EN2_RUN_TAG=injcountfix_gemini_s43 \
bash submit_en1_en2_sentence_train.sh
```

Submit the complete 3×3 matrix and its dependent evaluation jobs:

```bash
RUN_TAG=injcountfix_gemini_s43 \
GPU_PARTITION=p_b200_schwartz \
GPU_ACCOUNT=ug_schwartz \
bash submit_en1_en2_sentence_pipeline.sh
```

For the RTX private partition:

```bash
RUN_TAG=injcountfix_gemini_s43 \
GPU_PARTITION=p_rtx_schwartz \
GPU_ACCOUNT=ug_schwartz \
GPU_NODE=mm-lab02 \
TRAIN_TIME=48:00:00 \
TRANSLATION_TIME=48:00:00 \
bash submit_en1_en2_sentence_pipeline.sh
```

The pipeline refuses a dirty working tree, runs both prechecks, assigns stable
W&B IDs, submits dependencies with `afterok`, and writes a TSV job manifest
under `logs/en1en2_sentence/submissions/`.

## Translation validation

Held-out translation validation operates on individual Punkt sentences. For a
sentence `s`, it constructs `s + " " + s`, shifts one side into en2, and scores
only the target sentence. Source, separator, padding, and EOS labels use
`IGNORE_INDEX`.

The two metrics are:

```text
validation_metrics/translation_en1_to_en2/loss
validation_metrics/translation_en2_to_en1/loss
```

`run_en1_en2_translation_validation.slurm` loads model weights only and
evaluates requested checkpoints. Local CSV/JSONL output is retry-safe:
complete steps are skipped, partial steps fail, and explicit overwrite is
required to recompute existing rows.

## Knowledge-sharing evaluation

Generate the 16 lm-eval tasks from the same canonical entity order:

```bash
.venv/bin/python fictional_entity_data/eval_gen.py
```

Convert a checkpoint into the two 65,536-token HF views:

```bash
sbatch \
  --export=ALL,EXPERIMENT_NAME=<experiment>,STEP=4000 \
  convert_en1_en2_final_to_hf.slurm
```

Evaluate one view:

```bash
sbatch \
  --export=ALL,EXPERIMENT_NAME=<experiment>,STEP=4000,LANG_VIEW=en1 \
  run_en1_en2_knowledge_eval.slurm
```

The plotting script fits `acc_norm` across the 16 rate cells and reports the
own-language/other-language slope ratio. When the other-language slope is near
zero, signed or extreme ratios mean no detectable transfer and should not be
interpreted literally.

## Outputs

```text
../outputs/torchtitan/<experiment>/step-<step>/
../outputs/torchtitan/translation_validation/<experiment>/
../evals/knowledge_sharing/hf_models/<experiment>/step-<step>/<en1|en2>/
../evals/knowledge_sharing/results/<experiment>/step-<step>/<en1|en2>/
```

Use a new `EN1_EN2_RUN_TAG` or explicit `EN1_EN2_EXPERIMENT_NAME` for every
protocol revision. Existing checkpoints, translation rows, HF exports, and
evaluation results are not overwritten by default.
