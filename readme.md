# PaRC: Pair-Referenced Contrastive Learning

Implementation accompanying **Contrastive Learning with In-Batch Translation-Pair References for Multilingual Supervised Fine-Tuning**.

PaRC alternates multilingual supervised fine-tuning (SFT) with a contrastive objective whose reference is the mean translation-pair distance in the current batch. This release contains the training, data preparation, generation, retrieval, and COMET scoring code for **MASSIVE slot generation** and **ALMA-based translation evaluated on WMT23**, using Llama-3.2-1B-Instruct and Qwen3.5-2B.

The implementation name of PaRC is `gap_distance_infonce`. Select this value when reproducing the proposed method.

## Contents

- [Method](#method)
- [Installation](#installation)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results and metric conventions](#results-and-metric-conventions)
- [Reproducibility notes](#reproducibility-notes)
- [Code map](#code-map)

## Method

### Intuition

Cosine contrastive alignment rewards translation pairs for pointing in the same direction. PaRC instead asks whether a candidate's Euclidean distance is consistent with the distances of known translation pairs in the current batch. A candidate receives a high score when its distance is close to this batch reference; distances on either side of the reference receive lower scores. The contrastive loss must still distinguish the correct translation from the other candidates.

The reference adapts to the representations during training. It is neither a fixed target distance nor a dataset-wide estimate, and it does not explicitly enforce preservation of a language gap. Its gradient is retained, so the reference also contributes to the representation updates. At generation time, the model uses its ordinary language-model head; no reference computation or additional inference module is needed.

### Objective

An alignment batch contains $B$ translation pairs from one language pair $(a,b)$. Each sentence is tokenized independently, without the downstream task prompt or a chat template. Let $h_i^{(a)}$ and $h_i^{(b)}$ be their last non-padding token representations. The main configuration uses the final decoder output after final RMSNorm, without additional L2 normalization.

Define the candidate distances and the batch reference as

$$
d_{ij}=\left\|h_i^{(a)}-h_j^{(b)}\right\|_2,
\qquad
\mu_{\mathcal B}=\frac{1}{B}\sum_{k=1}^{B}d_{kk}.
$$

The PaRC logits are

$$
z_{ij}=-\frac{(d_{ij}-\mu_{\mathcal B})^2}{\kappa}.
$$

The diagonal entries are the positive translation pairs. All $B$ candidates, including the positive, participate in each softmax:

$$
\mathcal L_{\mathrm{PaRC}}=-\frac{1}{2B}\sum_{i=1}^{B}
\left[
\log\frac{\exp(z_{ii})}{\sum_{j=1}^{B}\exp(z_{ij})}
+\log\frac{\exp(z_{ii})}{\sum_{j=1}^{B}\exp(z_{ji})}
\right].
$$

In [models.py](models.py), the logit is computed as `-((distance - mean_distance) / alignment_gap_scale)**2 / alignment_temperature`. Thus

$$
\kappa=\texttt{alignment\_gap\_scale}^{\,2}\,
\texttt{alignment\_temperature}=1^2\times0.05=0.05.
$$

The reference is recomputed in each alignment forward pass and remains differentiable. Distance and PaRC loss calculations use FP32 with autocast disabled. The cosine baseline uses L2-normalized representations and cosine logits divided by temperature $0.05$.

SFT uses the model's chat template and supervises the assistant target tokens:

$$
\mathcal L_{\mathrm{SFT}}=-\frac{1}{N_{\mathrm{target}}}
\sum_{t\in\mathrm{target}}\log p_\theta(y_t\mid x,y_{<t}).
$$

Training alternates **one alignment optimizer update, then one SFT optimizer update**, with the same LoRA parameters and optimizer. The main budget is 100,000 updates: 50,000 alignment and 50,000 SFT updates. The two losses are computed on separate batches and alternate by update.

## Installation

Clone the repository, then run the remaining commands from its root directory, which contains `main.py` and this README:

```bash
git clone https://github.com/SoeonPark/ARR_PaRC.git
cd ARR_PaRC
```

Use Bash and Python 3.11. Training requires one CUDA GPU with BF16 support and a compatible NVIDIA driver. The reference experiments used one RTX 4090 per run. The `same_pair` sampler supports one process and one visible GPU.

### Training, data preparation, and generation environment

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

export PYTHON_BIN="$PWD/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

Choose an available physical GPU index for `CUDA_VISIBLE_DEVICES`. Keep only one index visible. These exports must also be set when opening a new shell. Always set `PYTHON_BIN`: some preserved shell launchers otherwise use an environment-specific absolute path.

[requirements.txt](requirements.txt) specifies the main environment, including:

| Package | Version |
| --- | --- |
| PyTorch | 2.14.0 |
| Transformers | 5.17.0 |
| Accelerate | 1.15.0 |
| PEFT | 0.20.0 |
| bitsandbytes | 0.50.2 |
| Datasets | 3.6.0 |
| NumPy | 2.4.6 |
| SacreBLEU, with Japanese tokenizer dependencies | 2.6.0 |

The requirements file also includes the data preparation and logging dependencies. Some auxiliary packages are unpinned; the file is not a complete transitive dependency lockfile.

Check the installed GPU environment:

```bash
"$PYTHON_BIN" -c 'import torch; print("torch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported()'
"$PYTHON_BIN" -m pip check
```

Model and dataset files are downloaded when absent from the Hugging Face cache. For Llama, obtain access to the model repository with your Hugging Face account and authenticate locally:

```bash
.venv/bin/hf auth login
```

W&B is optional. All training commands below disable it; no W&B account or API key is required. To use W&B, authenticate locally and change `--wandb_mode` or the launcher's `WANDB_MODE` to `online`.

### Separate COMET-22 environment

COMET scoring uses its own environment. Install [requirements-comet22.txt](requirements-comet22.txt) separately from the main requirements:

```bash
python3.11 -m venv .venv-comet22
.venv-comet22/bin/python -m pip install --upgrade pip
.venv-comet22/bin/python -m pip install -r requirements-comet22.txt
export COMET22_PYTHON="$PWD/.venv-comet22/bin/python"
"$COMET22_PYTHON" -m pip check
```

This environment pins `unbabel-comet==2.2.7`, `torch==2.5.1`, `transformers==4.44.2`, `pytorch-lightning==2.4.0`, and `numpy==1.26.4`. Calling its Python executable directly keeps training and scoring dependencies separate; activating this environment is unnecessary.

The scorer uses `Unbabel/wmt22-comet-da` at revision `2760a223ac957f30acfb18c8aa649b01cf1d75f2` and verifies the checkpoint and hyperparameter file SHA256 values. It downloads these files on first use. GPU scoring requires a CUDA-enabled PyTorch installation in this environment as well; `--gpus 0` selects CPU scoring.

## Data

Datasets and model weights are obtained from their providers and remain subject to their respective access conditions and licenses. They are not bundled with this code.

### MASSIVE and OPUS-100

No separate preparation command is needed for MASSIVE experiments. Training loads `AmazonScience/massive` and `Helsinki-NLP/opus-100` through Hugging Face Datasets.

| Component | Configuration |
| --- | --- |
| SFT languages / primary evaluation | English, Korean, Japanese, Spanish (`en`, `ko`, `ja`, `es`) |
| Additional out-language evaluation | French, German, Italian (`fr`, `de`, `it`) |
| MASSIVE training examples | 46,056 across the four SFT languages |
| MASSIVE test examples | 2,974 per language |
| Alignment pairs | English paired with Korean, Japanese, and Spanish |
| OPUS alignment training | 10,000 translation pairs per language pair |
| OPUS retrieval test | 2,000 translation pairs per language pair |

MASSIVE targets are semicolon-separated `slot_name: value` entries, or `None` when no slots are present. The English system instruction is shared across languages; the user message is the utterance in its original language.

### Translation: ALMA + Japanese extension / WMT23

Prepare the `alma_ja_opus` profile once, using a nonexistent or empty output directory:

```bash
"$PYTHON_BIN" scripts/prepare_wmt23.py \
  --output_dir data/wmt23_alma_ja_opus \
  --corpus_profile alma_ja_opus \
  --seed 42

export WMT23_DATA_DIR="$PWD/data/wmt23_alma_ja_opus"
export WMT23_MANIFEST_SHA256="$(sha256sum "$WMT23_DATA_DIR/manifest.json" | cut -d ' ' -f 1)"
```

The preparation command prints `MANIFEST_SHA256`; the exported value must match it. If the data have already been prepared, skip preparation and set the two exports. When transferring prepared data, copy the entire directory, including the input files recorded in the manifest.

The profile contains:

- German and Czech human-parallel data from `haoranxu/ALMA-Human-Parallel`: 14,211 and 12,076 training pairs, respectively, with its supplied validation split.
- A Japanese extension of 4,002 pairs from WMT20 and FLORES-200 dev/devtest. Japanese validation uses WMT21.
- Both translation directions for every training pair, yielding **60,578 SFT examples** over `en-de`, `de-en`, `en-cs`, `cs-en`, `en-ja`, and `ja-en`.
- **9,320 test examples** across those six directions from `haoranxu/WMT23-Test`. The `cs-en` test direction reverses the available `en-cs` pairs.
- Separate OPUS-100 alignment data: 10,000 training pairs and 2,000 retrieval test pairs for each of the three language pairs. Additional out-language evaluation data cover Chinese, Russian, and Ukrainian.

The preparation code pins ALMA, WMT23, FLORES, and OPUS revisions and records the resulting files and checksums in `manifest.json`. It preserves the SFT examples without length filtering or cross-corpus deduplication, rejects exact bilingual SFT/held-out overlaps, and excludes OPUS training rows whose English sentence occurs in the designated held-out pools before alignment sampling.

Training and evaluation validate the manifest, language partition, and recorded file sizes and SHA256 values. A newly prepared manifest can have a different hash because it records preparation metadata; use the hash for your prepared directory. Setting the hash does not replace inspection of the recorded data provenance.

SFT tokenization does not truncate examples. OPUS tokenization uses the tokenizer's default maximum length unless `--alignment_max_length` is set. The commands below retain the original length policy.

## Training

### Main settings

| Setting | Value |
| --- | --- |
| Models | `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3.5-2B` |
| Schedule | `alternative`: alignment, SFT, alignment, SFT, ... |
| Optimizer updates | 100,000 total; 50,000 per objective |
| Batch size / Trainer gradient accumulation | 16 / 1 |
| Alignment batch composition | 16 translation pairs from one language pair (`same_pair`) |
| Qwen translation SFT | Two internal microbatches of 8 per update; alignment remains 16 |
| Optimizer / peak learning rate | AdamW / `1e-4` |
| Learning-rate schedule / warmup | Linear / 10% of total updates |
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | BF16 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.1 |
| Representation | Last non-padding token, final layer (`-1`) |
| PaRC scale | `alignment_gap_scale=1`, `alignment_temperature=0.05` |
| Training seed / alignment sampling seed | 42 / 42 |
| Validation | Before training and every 2,500 updates |
| Reported checkpoint | Final checkpoint |

LoRA targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Qwen3.5 additionally targets the linear-attention projections `in_proj_qkv`, `in_proj_z`, and `out_proj`. The final representation corresponds to layer 16 for Llama and layer 24 for Qwen. Qwen uses text inputs with thinking disabled in its chat template.

### Reproduce the main PaRC training configuration

Define the common arguments in the same Bash session as the environment exports above:

```bash
common_args=(
  --training_type alternative
  --alignment_loss gap_distance_infonce
  --alignment_batching same_pair
  --alignment_num_samples_per_lang 10000
  --alignment_sampling_seed 42
  --training_anchor_langs en
  --training_seed 42
  --batch_size 16
  --accumulative_steps 1
  --num_steps 100000
  --learning_rate 0.0001
  --lr_scheduler_type linear
  --warmup_ratio 0.1
  --alignment_hidden_state_layer -1
  --alignment_hidden_state_position last_token
  --alignment_gap_scale 1.0
  --alignment_temperature 0.05
  --peft_lora_r 16
  --peft_lora_alpha 32
  --peft_lora_dropout 0.1
  --quantization_load_in_4bit
  --quantization_use_double_quant
  --quantization_type nf4
  --quantization_compute_dtype bfloat16
  --eval_batch_size 16
  --eval_steps 2500
  --eval_sample_log_limit 64
  --logging_steps 10
  --train_sample_log_interval 1000
  --train_sample_log_limit 8
  --wandb_mode disabled
)
```

**MASSIVE.** The following loop trains Llama and then Qwen on the selected GPU. Keep only one model in the list to run a single experiment. These commands preserve the recorded main runs' 500-update checkpoint interval and `both` validation scope. Out-language validation is diagnostic; evaluation uses the final checkpoint.

```bash
for model in meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen3.5-2B; do
  "$PYTHON_BIN" -u main.py "${common_args[@]}" \
    --model_name "$model" \
    --downstream_task massive \
    --training_lang ko ja es \
    --out_inference_lang fr de it \
    --downstream_micro_batch_size 0 \
    --save_steps 500 \
    --eval_language_scope both \
    --output_root "$PWD/results" || break
done
```

**Translation.** Prepare the data and export its path and manifest hash first. This loop also trains the two models sequentially. Qwen's SFT batch is split internally into two microbatches of eight; the optimizer still updates once per 16-example task batch.

```bash
for model in meta-llama/Llama-3.2-1B-Instruct Qwen/Qwen3.5-2B; do
  micro_batch=0
  if [[ "$model" == Qwen/Qwen3.5-2B ]]; then
    micro_batch=8
  fi
  "$PYTHON_BIN" -u main.py "${common_args[@]}" \
    --model_name "$model" \
    --downstream_task wmt23 \
    --wmt23_data_dir "$WMT23_DATA_DIR" \
    --wmt23_manifest_sha256 "$WMT23_MANIFEST_SHA256" \
    --wmt23_corpus_profile alma_ja_opus \
    --wmt23_downstream_sampling balanced_mixed \
    --training_lang de cs ja \
    --out_inference_lang zh ru uk \
    --downstream_micro_batch_size "$micro_batch" \
    --save_steps 1000 \
    --eval_language_scope in \
    --wandb_project_name Oct_ARR_WMT23 \
    --output_root "$PWD/results_wmt23" || break
done
```

`balanced_mixed` equalizes exposure across the seen translation-language pools while producing mixed-language SFT batches. Run the MASSIVE and translation loops sequentially when using the same GPU. Reference training times were approximately 8-15 hours for MASSIVE and 12-21 hours for translation per model on an RTX 4090, excluding final evaluation and downloads.

### Included shell launchers

The shell launchers provide an alternative to the direct commands above, with per-run logs and a per-GPU training lock. Despite their filenames, passing `alternative` selects the alternating schedule. Set the model, loss, and seed explicitly:

```bash
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct \
TRAINING_SEED=42 ALIGNMENT_LOSS=gap_distance_infonce \
WANDB_MODE=disabled TRAIN_EVAL_LANGUAGE_SCOPE=both \
bash scripts/massive_contrastive_only.sh alternative

# Requires WMT23_DATA_DIR and WMT23_MANIFEST_SHA256 from data preparation.
MODEL_NAME=Qwen/Qwen3.5-2B \
TRAINING_SEED=42 ALIGNMENT_LOSS=gap_distance_infonce \
WANDB_MODE=disabled TRAIN_EVAL_LANGUAGE_SCOPE=in \
bash scripts/wmt23_contrastive_only.sh alternative
```

Replace `MODEL_NAME` to select the other backbone. The MASSIVE launcher saves every **1,000** updates; use the direct command above for the recorded **500**-update cadence. Both launchers default to `in` validation unless overridden. Without the explicit overrides, the MASSIVE launcher uses seed 41 and queues both cosine and PaRC; the translation launcher queues both backbones. `OUTPUT_ROOT` overrides the output directory. Prefix a training launcher command with `DRY_RUN=1` to print its expanded command without training.

### Baselines and controls

The Python entry point also supports the following configurations. Apply these values in place of the corresponding entries in `common_args`; retain the task-specific arguments.

| Configuration | `training_type` | `alignment_loss` | Total updates |
| --- | --- | --- | --- |
| Task-only SFT | `transfer_only` | `infonce` (unused for training) | 50,000 |
| Cosine alignment followed by SFT | `contrastive_then_transfer` | `infonce` | 100,000 |
| Alternating cosine alignment and SFT | `alternative` | `infonce` | 100,000 |
| PaRC | `alternative` | `gap_distance_infonce` | 100,000 |
| Alignment only | `contrastive_only` | Choose the alignment loss | 50,000 |

For middle-layer cosine alignment, set `--alignment_hidden_state_layer 8` for Llama or `12` for Qwen. The recorded Llama MASSIVE middle-layer baseline used FP16 compute; the main PaRC commands above use BF16.

Available controls include `gap_distance_detach` (detach the batch reference), `gap_distance_rms` (divide distances and their reference by the live all-candidate RMS), `centered_infonce`, and `gap_direction_infonce`. `gap_consistency` minimizes positive-distance variance and is a separate objective. This release does not include every supplementary analysis or control from the manuscript, including the zero-reference and calibrated linear-distance training variants.

### Saved outputs

Training prints the run directory. Its structure is:

```text
results/ or results_wmt23/
  <organization>__<model>/
    <run-name>/
      experiment_config.json
      run_metadata.json
      adapter_config.json
      adapter_model.safetensors
      tokenizer files
      trainer_state.json
      training_args.bin
      checkpoint-<update>/
      eval_samples/
```

The run directory contains the final adapter after training completes. Intermediate checkpoint directories contain saved adapters as well. Base model weights are loaded separately. `experiment_config.json` records the resolved configuration; `run_metadata.json` records environment, dataset, and run information. The shell launchers write console logs under `logs/`.

All checkpoints are retained by default, so budget disk space for model downloads, adapters, and sample logs. The Python option `--save_total_limit` can limit checkpoint retention. Checkpoints use `save_only_model=True`; they do not provide optimizer/scheduler/RNG state for an exact interrupted-training continuation. Although `--resume_from_checkpoint` exists, model-only checkpoints do not guarantee trajectory-preserving resume.

## Evaluation

Use the **main environment** for representation extraction and text generation. Use the **COMET environment** only after generation finishes. The commands below evaluate final adapters and use the paper's generation batch size and token limits.

Replace the checkpoint placeholders with the run directories printed by training. An intermediate `checkpoint-N` directory can also be passed explicitly. Repeat evaluation for each model's run. Evaluation does not automatically select the best validation checkpoint.

### MASSIVE: slot generation and retrieval

```bash
export MASSIVE_CHECKPOINT="$PWD/results/<model-folder>/<run-name>"
export MASSIVE_EVAL="$MASSIVE_CHECKPOINT/evaluations/main"

EVAL_OUTPUT_DIR="$MASSIVE_EVAL" \
EVAL_SPLIT=test EVAL_LANGUAGE_SCOPE=in \
EVAL_TASKS="alignment massive" \
EVAL_ALIGNMENT_BATCH_SIZE=16 EVAL_MASSIVE_BATCH_SIZE=16 \
EVAL_MAX_NEW_TOKENS=128 \
bash scripts/eval.sh "$MASSIVE_CHECKPOINT"
```

This evaluates task generation in English, Korean, Japanese, and Spanish, plus bidirectional OPUS retrieval for the three alignment pairs. Greedy decoding uses at most 128 new tokens. The shell's unoverridden batch-size default is 32; the explicit values above select 16.

For additional out-language slot evaluation, run a separate invocation with `EVAL_LANGUAGE_SCOPE=out` and `EVAL_TASKS=massive`. `EVAL_LANGUAGE_SCOPE=both` evaluates both scopes and writes them into separate directories. To inspect an evaluation command without loading the model, insert `--dry-run` immediately after `scripts/eval.sh`; the checkpoint files must already exist.

### Translation: retrieval, generation, and BLEU

Set the prepared data path when evaluating on another machine or after moving the data. The checkpoint's saved manifest hash must match that directory.

```bash
export MT_CHECKPOINT="$PWD/results_wmt23/<model-folder>/<run-name>"
export MT_EVAL="$MT_CHECKPOINT/evaluations/main"
export MT_RETRIEVAL="$MT_CHECKPOINT/evaluations/retrieval"
export WMT23_DATA_DIR="$PWD/data/wmt23_alma_ja_opus"

# Bidirectional OPUS retrieval.
EVAL_OUTPUT_DIR="$MT_RETRIEVAL" \
EVAL_SPLIT=test EVAL_LANGUAGE_SCOPE=in \
EVAL_TASKS=alignment EVAL_ALIGNMENT_BATCH_SIZE=16 \
EVAL_COMET22=false \
bash scripts/wmt23_eval.sh "$MT_CHECKPOINT"

# WMT23 generation and SacreBLEU in the six seen directions.
EVAL_OUTPUT_DIR="$MT_EVAL" \
EVAL_SPLIT=test EVAL_LANGUAGE_SCOPE=in \
EVAL_TASKS=wmt23 EVAL_WMT23_BATCH_SIZE=16 \
EVAL_WMT23_MAX_NEW_TOKENS=16384 EVAL_WMT23_METRIC=sacrebleu \
EVAL_COMET22=false \
bash scripts/wmt23_eval.sh "$MT_CHECKPOINT"
```

Translation uses greedy decoding with at most **16,384 new tokens**. Generation is saved before COMET scoring. BLEU uses SacreBLEU 2.6.0, with MeCab for Japanese targets and `13a` for the other primary target languages. Lowering the token limit or changing decoding settings changes the evaluation protocol.

The two invocations use separate output roots to retain each task's `test/evaluation_metadata.json` and diagnostic sample records. Retrieval metrics are under `$MT_RETRIEVAL/test/in`; translation predictions and scores are under `$MT_EVAL/test/in`. `scripts/wmt23_eval.sh --dry-run "$MT_CHECKPOINT"` also supports command inspection with an existing checkpoint.

### COMET-22 on saved translations

After generation finishes, score its existing prediction files without loading the language model:

```bash
"$COMET22_PYTHON" scripts/score_comet22.py \
  --prediction_dir "$MT_EVAL/test/in" \
  --batch_size 16 \
  --gpus 1
```

Here `--gpus 1` means one GPU, selected by `CUDA_VISIBLE_DEVICES`; it is not a physical GPU index. Use `--gpus 0` for CPU scoring. The scorer writes `wmt23_comet22_metrics.json` and `wmt23_comet22_scores.jsonl` beside the predictions. Use `macro_average.comet22_x100` for the paper's equally weighted direction average.

Alternatively, set `EVAL_COMET22=true COMET22_GPUS=1 COMET22_BATCH_SIZE=16` on the translation-generation shell command. The wrapper then starts the separate COMET process after `evaluate.py` exits and releases the language model. Its default COMET device is CPU unless `COMET22_GPUS=1` is set.

### chrF++ on the same saved translations

The paper uses character order 6, word order 2, and beta 2. The evaluator's `chrf` option uses word order 0, so calculate **chrF++** from the saved predictions with the following command in the main environment. This prints each direction's corpus score and their equal-weight macro average without generating translations again:

```bash
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
from sacrebleu.metrics import CHRF

prediction_dir = Path(os.environ["MT_EVAL"]) / "test" / "in"
directions = ("cs-en", "de-en", "en-cs", "en-de", "en-ja", "ja-en")
scorer = CHRF(char_order=6, word_order=2, beta=2)
scores = []
for direction in directions:
    path = prediction_dir / f"wmt23_predictions.{direction}.jsonl"
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No predictions in {path}")
    score = scorer.corpus_score(
        [row["prediction"] for row in rows],
        [[row["target"] for row in rows]],
    ).score
    scores.append(score)
    print(f"{direction}: chrF++={score:.4f} (n={len(rows)})")
print(f"Macro chrF++: {sum(scores) / len(scores):.4f}")
print(f"Signature: {scorer.get_signature()}")
PY
```

Keep all generated hypotheses, including empty ones, when scoring. The command performs no additional case conversion, normalization, or hypothesis filtering.

## Results and metric conventions

Under the evaluation root, results are written to `<split>/<scope>/`, for example `evaluations/main/test/in/`:

| File | Contents / reporting convention |
| --- | --- |
| `massive_metrics.json` | Use `language_macro_slot_f1` and `language_macro_exact_match`, multiplied by 100 |
| `massive_predictions.jsonl` | Generated slot strings and reference targets |
| `alignment_metrics.json` | Bidirectional cosine retrieval; use `language_pair_macro` and multiply R@1, R@5, and MRR by 100 |
| `wmt23_metrics.json` | Direction scores and macro BLEU, already on the 0-100 scale |
| `wmt23_predictions.<direction>.jsonl` | Source, generated translation, reference, and generation diagnostics |
| `wmt23_comet22_metrics.json` | Direction scores and COMET macro, with explicit raw and x100 fields |
| `wmt23_comet22_scores.jsonl` | Segment-level COMET scores |

MASSIVE scoring parses slot-name/value **multisets**, normalizes case and whitespace, and counts duplicates. It computes micro slot F1 within each language and then averages languages equally. EM requires the entire parsed slot multiset to match. These are generation-based slot metrics, distinct from token-tagging F1.

Retrieval uses cosine similarity at the trained representation layer, including for PaRC models. Each direction ranks the correct translation against all **2,000 candidates** for that language pair; extraction batch size and query chunk size do not reduce this candidate pool. The two directions are averaged within a pair, followed by an equal-weight average over the three pairs. Training-time alignment loss is a separate, batch-dependent diagnostic.

Translation metrics are computed per direction and averaged equally across the six directions. COMET's `mean_score` is the example-weighted mean and differs from the direction macro used in the paper. The chrF++ command above likewise computes a direction macro.

The manuscript reports the following final-checkpoint PaRC scores:

| Backbone | MASSIVE F1 | MASSIVE EM | WMT23 BLEU | COMET x100 | chrF++ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.2-1B-Instruct | 77.12 | 74.98 | 15.52 | 71.15 | 36.34 |
| Qwen3.5-2B | 78.29 | 76.07 | 20.60 | 77.30 | 42.20 |

These are reference results from the manuscript. Compare a new run's saved metrics using the same scope, checkpoint, aggregation, and generation settings, taking the reproducibility notes below into account.

## Reproducibility notes

- **Configuration and provenance:** Preserve each run's `experiment_config.json`, `run_metadata.json`, tokenizer files, data manifest, and evaluation outputs. The training seed and OPUS sampling seed are separate settings. Model loading and the direct MASSIVE/OPUS download path do not pass explicit upstream revisions; the prepared MT profile does pin its dataset revisions.
- **Chat templates:** The released code preserves the original template behavior. Llama's template can insert a date-dependent `Today Date` field, and the code does not set `date_string`. A run on another date can therefore receive different task input tokens even with the same seed. Matching seeds alone does not establish identical inputs or a bitwise-identical trajectory.
- **Batch composition:** PaRC's reference and negatives depend on the alignment microbatch. Reducing the alignment batch size and increasing gradient accumulation changes the objective. Qwen MT's internal SFT microbatching leaves the 16-pair alignment batch intact.
- **Generation and precision:** Preserve the tokenizer, chat template, dtype, generation batch size, EOS behavior, and token limits for numerical comparisons. CUDA kernels and package/hardware differences can also affect exact numerical agreement.
- **Scope of this release:** This directory contains executable source and dependency specifications. It does not bundle trained adapters, prepared datasets, manuscript PDFs, historical predictions, bootstrap analysis, or figure-generation code. The commands above cover the core method and its primary task/retrieval metrics.

## Code map

| File | Role |
| --- | --- |
| [main.py](main.py) | Model/LoRA setup, dataset construction, training, and run metadata |
| [config.py](config.py) | CLI options, defaults, and configuration validation |
| [models.py](models.py) | Representation extraction, cosine contrastive loss, PaRC, and controls |
| [custom_trainer.py](custom_trainer.py) | Objective routing, alternating updates, SFT microbatching, and validation |
| [samplers.py](samplers.py) | Language-pair alignment batches and downstream sampling |
| [data_utils.py](data_utils.py) | Datasets, prompts, target masking, tokenization, and manifest validation |
| [utils.py](utils.py) | MASSIVE language/prompt definitions and model-specific LoRA targets |
| [alignment_logging.py](alignment_logging.py) | Alignment diagnostics and sample records |
| [evaluate.py](evaluate.py) | Adapter loading, generation, slot scoring, BLEU, and retrieval |
| [comet22_scoring.py](comet22_scoring.py) | Pinned COMET model verification and scoring |
| [scripts/prepare_wmt23.py](scripts/prepare_wmt23.py) | ALMA/Japanese, WMT23, and OPUS preparation with provenance |
| [scripts/score_comet22.py](scripts/score_comet22.py) | Standalone COMET scoring of saved predictions |
| [scripts/massive_contrastive_only.sh](scripts/massive_contrastive_only.sh) | MASSIVE training launcher |
| [scripts/wmt23_contrastive_only.sh](scripts/wmt23_contrastive_only.sh) | Translation training launcher |
| [scripts/eval.sh](scripts/eval.sh) | MASSIVE/retrieval evaluation launcher |
| [scripts/wmt23_eval.sh](scripts/wmt23_eval.sh) | Translation/retrieval evaluation and optional separate COMET scoring |

Inspect further options with `"$PYTHON_BIN" main.py --help`, `"$PYTHON_BIN" evaluate.py --help`, `"$PYTHON_BIN" scripts/prepare_wmt23.py --help`, and `"$COMET22_PYTHON" scripts/score_comet22.py --help` in their respective environments.
