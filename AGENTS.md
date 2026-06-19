# Repository Context

This repository is a TorchTitan fork for multilingual pretraining experiments,
with an emphasis on bilingual transfer and synthetic control. The main local
orientation document is `README_multilingual.md`.

## Research Goal

The core research question is whether pretraining can induce knowledge sharing
between two languages. The most controlled setup is the `en1/en2` experiment:
training text is English, but the tokenizer vocabulary is duplicated so that a
token can appear either in the base ID range (`en1`) or shifted by the base
vocabulary size (`en2`). The two "languages" are therefore the same language
surface form with disjoint token IDs. Fictional facts are injected into
pretraining data and later probed with MCQ evaluations to measure recall and
cross-language transfer.

## Important Code Paths

- `torchtitan/models/llama3/config_registry.py`: main experiment configs,
  including `smollm2_360m_en1_en2` and the sentence-level en1/en2 variants.
- `torchtitan/hf_datasets/text_datasets.py`: HuggingFace streaming dataset,
  staged curriculum, source mixing, injection, tokenization, and augmentation
  application.
- `torchtitan/hf_datasets/augmentations.py`: pre-tokenization text
  augmentations.
- `torchtitan/hf_datasets/post_tokenization_augmentations.py`:
  token-ID-shifting augmentations for synthetic language tagging.
- `fictional_entity_data/`: generation and lm-eval task tooling for synthetic
  fact recall.
- `run_train.sh`: local TorchTitan launcher.

## Existing en1/en2 Pattern

The 2x vocabulary model flavor is `smollm2_360m_2xvocab`. Synthetic en2 is
created post-tokenization by adding `vocab_size` to non-special token IDs.
Existing wordwise code-switching uses `stochastic_word_tagging` with
`symmetric=True` for mixed documents and `prob=1.0` for clean en2 documents.

The typical curriculum has two stages:

1. Intervention stage: mixed or code-switched documents plus clean injected
   en1/en2 sources.
2. Final clean stage: no mixed-language documents; en1 and en2 appear in
   separate clean documents so generation does not keep mixing languages.

## Sentence-Level Experiment Family

The new sentence-level configs are:

- `smollm2_360m_en1_en2_sentence_wise_code_switching`
- `smollm2_360m_en1_en2_sentence_parallel_doc_order`
- `smollm2_360m_en1_en2_sentence_parallel_sentence_order`

`EN1_EN2_PARALLEL_DOC_FRACTION` controls X, the fraction of stage-1 training
documents receiving the intervention. Use fractions only: `0.01` means 1%,
`0.6` means 60%, and `1.0` means all documents. The intervention source gets
weight X; clean injected en1/en2 sources split `1-X`. Stage-1 injection
probabilities are scaled to keep absolute exposure comparable when X changes.

The sentence-level implementation records character spans before tokenization
with `synthetic_sentence_language_mixing`, then shifts the tokens whose offsets
overlap en2 spans using `language_span_token_tagging`. Sentence experiments use
NLTK Punkt span tokenization so the transform gets abbreviation-aware
`(start, end)` spans.

## Common Environment Variables

- `EN1_EN2_PARALLEL_DOC_PERCENT`: X for the sentence experiment.
- `FICTIONAL_ENTITY_DATA_ROOT`: root containing `0/en_data.jsonl`, ...,
  `2079/en_data.jsonl`.
- `EN1_EN2_HF_ASSETS_PATH`: tokenizer/HF assets path, defaulting to the
  existing 65k paired tokenizer path.
- `EN1_EN2_OUTPUT_ROOT`: checkpoint output root.
- `EN1_EN2_STAGE1_STEPS`, `EN1_EN2_CLEAN_STEPS`: curriculum lengths.
- `EN1_EN2_LOCAL_BATCH_SIZE`, `EN1_EN2_GLOBAL_BATCH_SIZE`, `EN1_EN2_SEQ_LEN`:
  training shape overrides.

Before running sentence experiments in a fresh environment, install project
dependencies and download Punkt data once:

```bash
uv pip install -e .
python -m nltk.downloader -d "${HOME}/nltk_data" punkt punkt_tab
```

## Evaluation Notes

The intended MCQ recall evaluation uses `lm-evaluation-harness` cloned next to
this repo and installed into the active venv with `uv pip install -e`. Run
`fictional_entity_data/eval_gen.py` after updating its lm-eval task output path,
then run `lm_eval` against the HF-exported checkpoint and generated task group.
Exact cluster command may vary; keep Slurm scripts parameterized rather than
baking in personal lm-eval paths.
