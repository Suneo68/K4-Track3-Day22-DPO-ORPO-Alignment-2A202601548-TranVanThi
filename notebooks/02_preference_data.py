# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB2 — Preference Data
#
# **Stack:** `argilla/ultrafeedback-binarized-preferences-cleaned` (English) +
# `bkai-foundation-models/vi-alpaca` (native VN, rejection-sampled against SFT-mini)
# + tokenizer `apply_chat_template`.
# Maps to deck §5.1 (preference data formats) + §5.4 (VN landscape — what exists vs not).
#
# > **Mục tiêu:** load 2 nguồn preference, format thành `{prompt, chosen, rejected}` với
# > chat template Qwen2.5, lưu Parquet vào `data/pref/`.
# >
# > Deck §5.4 lists VN preference data realities:
# > - **VinaLLaMA / PhoGPT / Vistral**: SFT-only, no published DPO data.
# > - **SeaLLM / Sailor2**: DPO-aligned, Sailor2 has `Sailor2-translated-ultrafeedback-vi`.
# > - **Native VN preference**: gap. This notebook closes part of that gap directly —
# >   see §2b below — instead of leaving it as a bonus-only provocation.

# %% [markdown]
# ## 0. Setup

# %%
import os
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()

# Total preference pairs, and what fraction of them are native-VN (constructed in
# §2b) vs English UltraFeedback (§2). Defaults: 1800 EN + 200 VN (T4) / 4500 EN +
# 500 VN (BIGGPU) — a 90/10 mix. English stays the majority so headline numbers
# (reward gap, NB4 win-rate) remain comparable to the deck demo (§7.1: "2k
# UltraFeedback, 3.2 -> 4.1"); the VN slice adds a real Vietnamese alignment signal
# on top instead of aligning purely on translated/foreign-language preferences.
if COMPUTE_TIER == "T4":
    PREF_SLICE = int(os.environ.get("PREF_SLICE", "2000"))
    MAX_LEN = 512
    MAX_PROMPT_LEN = 256
else:
    PREF_SLICE = int(os.environ.get("PREF_SLICE", "5000"))
    MAX_LEN = 1024
    MAX_PROMPT_LEN = 512

VN_PREF_FRACTION = float(os.environ.get("VN_PREF_FRACTION", "0.1"))
VN_POOL_SIZE = int(PREF_SLICE * VN_PREF_FRACTION)
ENGLISH_SLICE = PREF_SLICE - VN_POOL_SIZE

# NB1's SFT-mini was trained on vi-alpaca rows [:SFT_SLICE]. The native-VN pool
# below must start AFTER that so `chosen` answers were never seen during SFT —
# otherwise the DPO reward gap partly measures memorisation, not preference.
SFT_SLICE = int(os.environ.get("SFT_SLICE", "1000"))

PREF_DATASET = os.environ.get(
    "PREF_DATASET", "argilla/ultrafeedback-binarized-preferences-cleaned"
)
VN_DATASET = os.environ.get("VN_PREF_DATASET", "bkai-foundation-models/vi-alpaca")

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ADAPTER_DIR = REPO_ROOT / "adapters" / "sft-mini"
PREF_OUT = REPO_ROOT / "data" / "pref"
PREF_OUT.mkdir(parents=True, exist_ok=True)

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"PREF_DATASET:    {PREF_DATASET}  (slice: {ENGLISH_SLICE})")
print(f"VN_DATASET:      {VN_DATASET}  (pool: {VN_POOL_SIZE}, starting at row {SFT_SLICE})")
print(f"Total pairs:     {PREF_SLICE}  ({ENGLISH_SLICE} EN + {VN_POOL_SIZE} VN native)")
print(f"MAX_LEN:         {MAX_LEN}")
print(f"MAX_PROMPT_LEN:  {MAX_PROMPT_LEN}")
print(f"output:          {PREF_OUT}")

# %%
import torch

assert torch.cuda.is_available(), "Building native-VN pairs needs a GPU to generate rejected samples. See HARDWARE-GUIDE.md."

# %% [markdown]
# ## 1. Load tokenizer (matches NB1 base model)

# %%
from transformers import AutoTokenizer

assert ADAPTER_DIR.exists(), f"NB1 must run first — {ADAPTER_DIR} missing"
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer: {tokenizer.__class__.__name__}  vocab={tokenizer.vocab_size:,}")

# %% [markdown]
# ## 2. Load UltraFeedback (English baseline)
#
# **Why English?** UltraFeedback was the canonical preference dataset of the deck
# demo (§7.1: "2k UltraFeedback pairs, 30 min A100, 3.2 → 4.1 helpfulness"). Using
# the same dataset keeps the headline reward-gap number comparable to the deck.
#
# **Why not 100% Vietnamese?** Native VN preference data is a gap (deck §5.4), and
# what we can build ourselves in a 30-min lab (§2b) is small and rejection-sampled,
# not human-curated at UltraFeedback's scale. Blending keeps the bulk of the signal
# on a well-understood dataset while still giving the model real VN preference
# examples instead of zero.

# %%
from datasets import load_dataset

ds = load_dataset(PREF_DATASET, split=f"train[:{ENGLISH_SLICE}]")
print(f"Loaded {len(ds)} pairs. Columns: {ds.column_names}")

# %% [markdown]
# ## 3. Format with chat template
#
# DPO Trainer expects `prompt / chosen / rejected` columns. Each must already
# include the chat template tokens — Trainer doesn't apply template internally.

# %%
def format_pref(row):
    prompt_msgs = [{"role": "user", "content": row["prompt"]}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    # `chosen` and `rejected` in this dataset are list-of-dicts with role/content.
    # Take just the assistant turn text (last message).
    chosen_text = row["chosen"][-1]["content"] if isinstance(row["chosen"], list) else row["chosen"]
    rejected_text = row["rejected"][-1]["content"] if isinstance(row["rejected"], list) else row["rejected"]
    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "source": "ultrafeedback_en",
    }


pref_en = ds.map(format_pref, remove_columns=ds.column_names)
print(f"Formatted: {len(pref_en)} pairs · cols: {pref_en.column_names}")

# %% [markdown]
# ## 2b. Build native-VN preference pairs (rejection sampling)
#
# **Idea (deck §5.3 option 2 — "generate native"):** for a disjoint slice of
# `vi-alpaca` that NB1 never trained on, treat the dataset's own human-written
# `output` as `chosen`, and greedily generate a `rejected` answer from the
# **SFT-mini model itself** for the same instruction. SFT-mini has seen only 1
# epoch on 1k examples, so its own greedy completion is a plausible-but-weaker
# answer versus the reference — a cheap, judge-free way to get a real VN
# preference signal without an API key. This is the same family of trick as
# SPIN / self-rewarding rejection sampling, simplified to "one greedy sample."
#
# This is the part of the notebook that actually needs the GPU — budget an extra
# ~10-15 min on T4 for `VN_POOL_SIZE` generations. Lower `VN_PREF_FRACTION` (env
# var) if you're time-constrained; 0 disables this section entirely and NB2
# behaves exactly like the English-only baseline.

# %%
if VN_POOL_SIZE > 0:
    vn_ds = load_dataset(VN_DATASET, split=f"train[{SFT_SLICE}:{SFT_SLICE + VN_POOL_SIZE}]")
    vn_ds = vn_ds.filter(
        lambda r: bool((r.get("instruction") or "").strip())
        and bool((r.get("output") or "").strip())
    )
    print(f"Loaded {len(vn_ds)} native-VN rows from {VN_DATASET}[{SFT_SLICE}:{SFT_SLICE + VN_POOL_SIZE}]")
else:
    vn_ds = None
    print("VN_PREF_FRACTION=0 — skipping native-VN pair construction.")

# %%
def vn_alpaca_prompt(row) -> str:
    text = row["instruction"]
    if row.get("input"):
        text += "\n\n" + row["input"]
    return text


if vn_ds is not None:
    from unsloth import FastLanguageModel
    from peft import PeftModel
    import gc

    BASE_MODEL = (
        "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4" else "unsloth/Qwen2.5-7B-bnb-4bit"
    )
    gen_model, gen_tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_LEN, dtype=None, load_in_4bit=True,
    )
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_model = PeftModel.from_pretrained(gen_model, str(ADAPTER_DIR))
    FastLanguageModel.for_inference(gen_model)

    vn_rows = []
    for i, row in enumerate(vn_ds):
        prompt = vn_alpaca_prompt(row)
        messages = [{"role": "user", "content": prompt}]
        inputs = gen_tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to("cuda")
        with torch.no_grad():
            out = gen_model.generate(
                input_ids=inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=gen_tokenizer.eos_token_id,
            )
        rejected = gen_tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        chosen = row["output"].strip()

        if rejected and rejected != chosen:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            vn_rows.append({
                "prompt": prompt_text, "chosen": chosen, "rejected": rejected,
                "source": "vn_native",
            })
        if (i + 1) % 25 == 0:
            print(f"  generated {i + 1}/{len(vn_ds)} native-VN rejected samples")

    del gen_model, gen_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    from datasets import Dataset

    pref_vn = Dataset.from_list(vn_rows) if vn_rows else None
    n_built = len(pref_vn) if pref_vn is not None else 0
    print(f"\nBuilt {n_built} native-VN pairs "
          f"({len(vn_ds) - n_built} skipped — SFT-mini's greedy output matched the reference exactly)")
else:
    pref_vn = None

# %% [markdown]
# ## 4. Combine + shuffle

# %%
from datasets import concatenate_datasets

if pref_vn is not None and len(pref_vn) > 0:
    pref = concatenate_datasets([pref_en, pref_vn]).shuffle(seed=42)
else:
    pref = pref_en.shuffle(seed=42)

print(f"Combined: {len(pref)} pairs · cols: {pref.column_names}")

from collections import Counter
print("Composition:", dict(Counter(pref["source"])))

# %% [markdown]
# ### 4a. Inspect 3 examples + token counts (deliverable: NB2 rubric §2)
#
# Deliberately shows at least 1 example from EACH source (not just 3 random rows,
# which at a 90/10 mix would likely show zero native-VN examples).

# %%
import textwrap

vn_idx = next((i for i in range(len(pref)) if pref[i]["source"] == "vn_native"), None)
en_idx = [i for i in range(len(pref)) if pref[i]["source"] == "ultrafeedback_en"][:2]
show_idx = ([vn_idx] if vn_idx is not None else []) + en_idx
show_idx = show_idx[:3] if show_idx else list(range(3))

for n, i in enumerate(show_idx, start=1):
    row = pref[i]
    n_prompt = len(tokenizer(row["prompt"]).input_ids)
    n_chosen = len(tokenizer(row["chosen"]).input_ids)
    n_rejected = len(tokenizer(row["rejected"]).input_ids)
    print(f"\n────── Example {n} · source={row['source']} ──────")
    print(f"PROMPT ({n_prompt} tok):\n{textwrap.shorten(row['prompt'], 200)}")
    print(f"\nCHOSEN ({n_chosen} tok):\n{textwrap.shorten(row['chosen'], 250)}")
    print(f"\nREJECTED ({n_rejected} tok):\n{textwrap.shorten(row['rejected'], 250)}")
    assert row["chosen"] != row["rejected"], "chosen == rejected — dataset is corrupt!"

# %% [markdown]
# ### 4b. Length distribution check
#
# Pairs longer than `MAX_LEN` will be truncated by the trainer. If too many are
# clipped, DPO loses signal. Aim for ≥ 80% of pairs fitting.

# %%
import numpy as np

prompt_lens = np.array([len(tokenizer(p).input_ids) for p in pref["prompt"]])
chosen_lens = np.array([len(tokenizer(c).input_ids) for c in pref["chosen"]])
rejected_lens = np.array([len(tokenizer(r).input_ids) for r in pref["rejected"]])

total_len = prompt_lens + np.maximum(chosen_lens, rejected_lens)
fit_pct = (total_len <= MAX_LEN).mean() * 100

print(f"Prompt:   median={np.median(prompt_lens):.0f}  P95={np.percentile(prompt_lens, 95):.0f}")
print(f"Chosen:   median={np.median(chosen_lens):.0f}  P95={np.percentile(chosen_lens, 95):.0f}")
print(f"Rejected: median={np.median(rejected_lens):.0f}  P95={np.percentile(rejected_lens, 95):.0f}")
print(f"\n{fit_pct:.1f}% of pairs fit in MAX_LEN={MAX_LEN}")
if fit_pct < 80:
    print("⚠  Less than 80% fit. Consider increasing MAX_LEN or filtering long pairs.")

# %% [markdown]
# ## 5. Save Parquet

# %%
pref.to_parquet(str(PREF_OUT / "train.parquet"))
print(f"Saved {len(pref)} pairs to {PREF_OUT / 'train.parquet'}  {dict(Counter(pref['source']))}")

# Also save a small eval slice (last 50 pairs) for NB4 use.
eval_slice = pref.select(range(len(pref) - 50, len(pref)))
eval_slice.to_parquet(str(PREF_OUT / "eval.parquet"))
print(f"Saved 50 eval pairs to {PREF_OUT / 'eval.parquet'}")

# %% [markdown]
# ## 6. Vibe-coding callout
#
# Bạn vừa build 1 preference set **hybrid**: phần lớn English UltraFeedback (fair
# so sánh với deck demo) + 1 phần native-VN pairs tự sinh bằng rejection sampling
# từ chính SFT-mini (deck §5.3 option 2, `BONUS-CHALLENGE.md` provocation 1).
#
# 3 câu hỏi think-hard-zone để đưa vào `submission/REFLECTION.md` § 6:
#
# 1. **Chất lượng rejected tự sinh**: SFT-mini's rejected output có thực sự "tệ
#    hơn" chosen, hay đôi khi ngang bằng/tốt hơn (nghĩa là preference signal bị
#    đảo)? Đọc lại vài cặp `source=vn_native` ở mục 4a — bạn có tin tưởng nhãn
#    chosen/rejected này không?
# 2. **Tỷ lệ trộn**: thử tăng `VN_PREF_FRACTION` từ 0.1 → 0.3, re-run NB2 → NB3.
#    Reward gap có đổi không? Output tiếng Việt có "tự nhiên" hơn không?
# 3. **So sánh với dịch máy**: nếu có thời gian, thử thay `vn_ds` bằng bản dịch
#    NLLB của UltraFeedback (`Sailor2-translated-ultrafeedback-vi`) và so sánh
#    fluency của output DPO cuối cùng — dịch máy vs native VN construction.
#
# **Next:** NB3 — train DPO trainer với reward curves.
