# Exact SFT checkpoint rerun

Completed September 17, 2026. All 800 responses generated and judged. Endpoint verified paused; paid-run notebook controls switched off.

## Result

| Run | VEA-positive | Rate |
|---|---:|---:|
| Released-main SFT | 264/800 | 33.0% |
| Exact `5e-5-step10000` rerun | 256/800 | 32.0% |
| Published `5e-5-step10000` result | 492/800 | 61.5% |

**The exact checkpoint did not resolve the discrepancy.** Under our fixed prompt, sampling, and judging protocol, its rate is 1 percentage point below released-main and 29.5 points below the published result. The prior hypothesis that the checkpoint difference was the main explanation is not supported by this rerun.

All 800 generated answers are unique, stopped normally, and have closing reasoning tags; none has empty reasoning. All 800 judge calls returned `openai/gpt-5-mini` from provider OpenAI and stopped normally; no judge input was truncated. No new answer exactly duplicates its corresponding released-main answer. All 14 non-verbatim quote flags were manually checked against the underlying reasoning and have supporting evaluation/test language; labels remain unchanged. This was not exhaustive human relabeling of all 800 answers.

Descriptive paired-question bootstrap intervals (100 question groups, 10,000 resamples) are -4.875 to +2.75 percentage points versus main and -36.125 to -22.875 versus the published checkpoint. These do not identify the source of the discrepancy or account for systematic judge/setup differences.

Generation took 113.4 seconds for the pilot and 2353.1 seconds for the remainder, about 41.1 minutes total. Endpoint displayed usage increased from57 minutes/$4.75 to105 minutes/$8.75: $4.00 incremental GPU cost including startup/idle time. Judge reported cost was $0.53148925, approximately $4.53 combined. Provider billing is authoritative.

Judge/report folder: `/content/drive/MyDrive/olmo-evaluation/sft-jbb-644d21da7e08/vea-judge-7356919ece86`. Includes `comparison.md`, `summary.json`, `judgments.jsonl`, `comparison-per-question.csv`, `browse-comparison.html`, `final-audit.json`, and `manual-quote-review.json`.

Next investigation should examine the authors' exact original SFT input formatting/generation environment and judge calibration. No further paid run or message to the authors was initiated.

## Purpose

Compare the paper's `5e-5-step10000` checkpoint against the completed released-main SFT run while holding the question texts, rendered prompts, generation settings, and judge protocol fixed. Preserve the released-main result of 264/800 (33.0%). The published checkpoint result is 492/800 (61.5%).

## Configuration

- Model: `allenai/Olmo-3-32B-Think-SFT`
- Branch: `5e-5-step10000`
- Immutable revision: `e72d7528f502a3e68929bb8b33b6f17f0e69e70f`
- Endpoint: existing `olmo-sft-jbb`, updated from released-main while paused, then resumed. Overview verified the new revision.
- Hardware: two A100 80GB GPUs; tensor parallel 2; vLLM 0.29.0; context limit 20480; BF16.
- Dataset: same 100 JailbreakBench prompts, eight answers each.
- Sampling: temperature 0.6, top_p 0.95, seed 42, output cap 8192, two concurrent questions.
- Template: explicitly use the embedded Think template in pinned `tokenizer_config.json`. The checkpoint also has a conflicting standalone template. All 100 rendered input strings were asserted identical to the completed main run; tokenization consistency checked.
- Judge: OpenRouter `openai/gpt-5-mini`, low reasoning effort, 4000-token output budget; same source rubric and quote-based labels as the completed run. Exact-quote discrepancies are flagged for review, not relabeled automatically.

## New files and notebook cells

Colab: (see SFT_Evaluation.ipynb in this repository)

- Cells19–21: checkpoint preparation, 3-question pilot, saved-pilot checks.
- Cell22: full generation, resuming the pilot.
- Cells23–25: export validation and matching judge setup/functions.
- Cells26–27: judge execution control and comparison report.
- Cells28–29: final integrity checks, question-group uncertainty, manual quote review, and saved narrative report.
- Results: `/content/drive/MyDrive/olmo-evaluation/sft-jbb-644d21da7e08`
- Original results remain in `/content/drive/MyDrive/olmo-evaluation/sft-jbb-efcce86f8389`.

## Pilot

All 24 answers saved and valid; all contained closing reasoning tags; none hit the output cap. Pilot generation took 113.4 seconds. The remaining 97 questions were then started with user authorization.

## Sources

- [Published judgments](https://github.com/arbdwj/VEA-through-training/blob/2c1379ee9648c16884bb1634d554a27154d7a01c/data/stage_vea.jsonl)
- [Historical methods appendix](https://github.com/arbdwj/VEA-through-training/blob/2b92e419499bae35a6226422fe0970c124683298/draft_old/post.md#L111-L163)
- [Judge source](https://github.com/arbdwj/VEA-through-training/blob/2c1379ee9648c16884bb1634d554a27154d7a01c/pipeline/judge_vea.py)

The original SFT generation program and complete serving environment are not available. This rerun controls the known configuration within our two runs; it does not promise bit-for-bit reproduction of the authors' outputs or isolate model averaging from additional SFT training.
