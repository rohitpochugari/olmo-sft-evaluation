# SFT / RLVR reproduction investigation

Investigated September 17, 2026. No Colab edits, paid inference, endpoint changes, or judge requests were made during this investigation.

## Main finding

The missing generation scripts were not recovered, but an earlier write-up contains explicit prompt formats and sampling settings. This substantially narrows the reproduction gap. Earlier claims that these settings could only be inferred from results were incomplete.

## Repository history

The original local clone was shallow. After downloading the full public history, all 12 commits were inspected. The SFT/RLVR generation scripts were absent from the initial commit and were not added in the inspected history. Public GitHub metadata lists two branches, one merged data-only pull request, and no forks. The second branch is the merged PR's source.

The original README describes pipeline scripts as reference material rather than a turnkey rerun. This documents the limited release scope but does not explain why the authors omitted particular scripts.

- [Original README](https://github.com/arbdwj/VEA-through-training/blob/c2023dc6f6fbd3aed087afc50840b846a0d12498/README.md)
- [Historical appendix](https://github.com/arbdwj/VEA-through-training/blob/2b92e419499bae35a6226422fe0970c124683298/draft_old/post.md#L111-L163)
- [Merged data PR](https://github.com/arbdwj/VEA-through-training/pull/1)

The draft was moved under draft_old and subsequently removed by commit 9dbc731, whose message is “rm stale files.” The current LessWrong comments specifically correct the draft's old judge-model account. Treat the appendix as historical evidence, not an unquestionable final specification.

## Recovered generation settings

Appendix B, line 157, explicitly describes vLLM, tensor parallel size 2, H100 GPUs, eight responses per prompt, temperature 0.6, top_p 0.95, and seed 42.

| Runs | Output-token cap | Engine context limit |
|---|---:|---:|
| Post-training and released base main | 8192 | 20480 |
| Intermediate initial-pretraining checkpoints | 2048 | 8192 |

The context limit is the combined prompt/output budget configured in the engine; it is not the output cap. Current SFT model config advertises native max_position_embeddings=65536, so the historical draft's description of 20480 as the models' native extension should not be taken literally.

Cross-check against current published result files:

- SFT JBB across sampled checkpoints has responses up to 5960 tokens. At step10000, the maximum is 2243. Other SFT benchmarks have length-capped 8192-token responses.
- RLVR JBB has length-capped 8192-token responses.
- RLVR Fortress has length-capped 16384-token responses. Thus the draft's universal post-training cap is not valid for every currently published benchmark; the evidence supports 8192 for our JBB selection.
- SFT JBB step10000 has 800 rows, all lr=5e-5, with 492 positive VEA judgments.

## Prompt format and actual cause of the hiccup

Appendix A explicitly gives identical base/SFT prompts: the default helpful-assistant system message, the benchmark user message, then an assistant prefix ending in <think>. RLVR uses the longer Ai2/Olmo identity-and-date system message, also ending its generation prefix in <think>.

At SFT revision e72d7528f502a3e68929bb8b33b6f17f0e69e70f (5e-5-step10000):

- tokenizer_config.json contains the full Think template, including the default system message and opening <think>.
- chat_template.jinja contains a generic template lacking both additions.
- Transformers 4.57.1 explicitly gives standalone template files priority over the template inside tokenizer_config.json. This explains the generic template observed in Colab.
- The released main revision a6d7f3cf497c7049712c13a664c65c7992f2da0c has the same embedded Think template and no standalone chat_template.jinja. Our explicit template selection therefore matches the documented base/SFT format for our single-user-message prompts.

Sources:

- [Step tokenizer configuration](https://huggingface.co/allenai/Olmo-3-32B-Think-SFT/blob/e72d7528f502a3e68929bb8b33b6f17f0e69e70f/tokenizer_config.json)
- [Conflicting standalone template](https://huggingface.co/allenai/Olmo-3-32B-Think-SFT/blob/e72d7528f502a3e68929bb8b33b6f17f0e69e70f/chat_template.jinja)
- [Transformers precedence, lines 2167–2184](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/tokenization_utils_base.py#L2167-L2184)

## Checkpoint choice

The selected SFT checkpoint is the final sampled point in the research's 5e-5 SFT sweep, not merely the repository's default main branch. Public Hugging Face metadata shows 14 identically named weight shards on main and step10000, with zero matching SHA256 hashes. These are different weight-file releases, not just different documentation commits. Metadata alone does not establish the numerical relationship of every tensor.

This justifies explicitly choosing the paper's sampled branch for a comparison to its step10000 results. It does not establish that step10000 and the default released SFT model are interchangeable or that step10000 was the end of every SFT training run.

## Practical outcome and remaining uncertainty

The notebook's eight samples, temperature, top_p, seed, 8192-token cap, and SFT prompt shape now have historical author documentation supporting them. The provenance comments should be updated on the next code-edit request. Set the future endpoint's engine context limit to 20480 to follow the documented configuration; the current notebook only checks that at least 8245 tokens are supported.

Exact original SFT/RLVR generation scripts, model/template commit pins at experiment time, and complete serving-library versions remain unavailable. The original implementation's handling of template conflicts is unknown. The 16384-token Fortress results are a concrete reason not to treat the stale appendix as universally final.

No need to abandon the JBB reproduction. Describe it as following the documented method, with hardware and unavailable implementation details recorded, rather than claiming a byte-for-byte recreation of the original run.
