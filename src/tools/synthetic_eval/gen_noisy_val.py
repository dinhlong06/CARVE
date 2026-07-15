"""Generate noisy val queries by LLM style-rewriting clean captions, with a DIFFICULTY GRADIENT
(composition) so the val covers easy->hard, not just one style. Qwen, vLLM, chat format.

Run inside the ssdc-vllm container (vLLM image), cwd /workspace/SSDC:
  docker exec -e CUDA_VISIBLE_DEVICES=7 ssdc-vllm \
    python tools/synthetic_eval/gen_noisy_val.py --limit 50 --gpu-mem 0.6

Pipeline (this script = stage A; meaning-gate = stage B in ssdc-eval):
  A. read val_queries_clean.jsonl {caption, image_id}; for each query emit `variants` rewrites,
     each at a difficulty LEVEL (easy/medium/hard). Writes {caption, image_id, style, level}.
  B. tools/synthetic_eval/filter_noisy.py applies the meaning-gate (CMP-text cos >= 0.85) and
     drops drifted rewrites that would corrupt the gt. (Different container: CMP needs transformers
     4.x, vLLM needs 5.x.)

Difficulty levels (composition = stack transformations, the spec's coverage-by-breadth):
  easy   = paraphrase only (closest to clean, high cos)
  medium = one style rewrite (casual / formal / concise / narrative)
  hard   = composite rewrite ('all'): several style transformations stacked in ONE prompt
           (casual+reorder+condense), mirroring the test's 'all' change.

NOTE: word-level EDA (synonym / swap / delete) is NOT done here -- it is already applied during
CMP training (dataset/eda.py via search_dataset, eda_p). The val therefore focuses on the
style-rewrite axis (the HARD residual that EDA does not cover); this gen is LLM-rewrites only.

Uses the model CHAT TEMPLATE (llm.chat) so Qwen follows the instruction instead of continuing it.
"""
import json
import random
import argparse

# Generic rephrasing AXES from NLP augmentation literature (register / length / discourse /
# lexical / composition). Named by their linguistic axis -- NOT by any benchmark's change labels.
# Each instruction is constrained to PRESERVE meaning and NOT invent details (drift -> wrong gt).
PROMPTS = {
    "lexical_paraphrase": "Paraphrase the description below. Preserve its exact meaning and every "
                   "concrete detail; do NOT add, remove, or invent anything. Output only the paraphrase.\n\n{t}",
    "register_informal": "Rewrite the description below in a casual, conversational tone. Keep the same "
                   "meaning and all the same details; do NOT add or invent anything. Output only the "
                   "rewritten description.\n\n{t}",
    "register_formal": "Rewrite the description below in a formal, precise tone. Keep the same meaning and "
                   "all the same details; do NOT add or invent anything. Output only the rewritten "
                   "description.\n\n{t}",
    "length_compressed": "Rewrite the description below as a single concise sentence (at most two), keeping "
               "the most important concrete details: the person/people, their key clothing colours and items, "
               "the main action, and the setting. Keep it a natural sentence; do NOT add or invent "
               "anything. Output only the concise sentence.\n\n{t}",
    "discourse_narrative": "Retell the description below as one short narrative sentence-or-two. Keep the same "
                   "meaning and all the same details; do NOT add or invent anything. Output only the "
                   "narrative.\n\n{t}",
    "composite_multiop": "Rewrite the description below applying several changes AT ONCE: use a casual, "
                   "spoken tone, REORDER the information, and drop filler words. Keep the same meaning "
                   "and every concrete detail (the people, their clothing colours and items, the main "
                   "action, the setting); do NOT add, remove, or invent anything. Output only the "
                   "rewritten description.\n\n{t}",
}
# style pool per difficulty level (composition lives in the multi-op prompt, not in EDA)
LEVEL_STYLES = {
    "easy":   ["lexical_paraphrase"],
    "medium": ["register_informal", "register_formal", "length_compressed", "discourse_narrative"],
    "hard":   ["composite_multiop"],   # several rephrasing ops stacked in one prompt
}
LEVELS = ["easy", "medium", "hard"]


def clean_out(text):
    """Strip preambles/quotes the model sometimes adds, keep the (possibly multi-sentence) body."""
    t = text.strip()
    # drop a leading "Here's ...:" / "Okay ...:" preamble line if a blank line separates it from the body
    if "\n\n" in t:
        head, rest = t.split("\n\n", 1)
        low = head.lower()
        if len(head) < 60 and (low.startswith(("here", "okay", "sure", "rewritten", "paraphrase")) or low.endswith(":")):
            t = rest.strip()
    return t.strip().strip('"').strip()


def build_tasks(recs, variants, level_weights, seed):
    """Return list of (rec, level, style). variants=1 -> one random-level variant per query
    (levels spread across the dataset); variants>=2 -> one per level (same query, graded)."""
    rng = random.Random(seed)
    tasks = []
    for r in recs:
        if variants <= 1:
            lvls = [rng.choices(LEVELS, weights=level_weights)[0]]
        else:
            lvls = LEVELS[:variants] if variants <= len(LEVELS) else LEVELS
        for lvl in lvls:
            style = rng.choice(LEVEL_STYLES[lvl])
            tasks.append((r, lvl, style))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='checkpoint/Qwen3-4B-Instruct-2507-AWQ-4bit')
    ap.add_argument('--clean', default='data/synthetic_eval/val_queries_clean.jsonl')
    ap.add_argument('--out', default='data/synthetic_eval/val_queries_noisy_candidates.jsonl',
                    help="candidates (pre meaning-gate). Run filter_noisy.py -> val_queries_noisy.jsonl")
    ap.add_argument('--variants', type=int, default=1,
                    help="rewrites per query. 1 = one random-level each (spread). 2-3 = one per level.")
    ap.add_argument('--level-weights', default='1,1,1', help="easy,medium,hard weights when variants=1")
    ap.add_argument('--limit', type=int, default=0, help="0 = all")
    ap.add_argument('--gpu-mem', type=float, default=0.6)
    ap.add_argument('--tp', type=int, default=1,
                    help="tensor_parallel_size: split the model across N GPUs (set CUDA_VISIBLE_DEVICES "
                         "to N ids). Halves per-GPU memory -> fits when no single card has enough free.")
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=20260622)
    ap.add_argument('--dry-run', action='store_true',
                    help="skip vLLM; use clean caption as identity rewrite (validate format/EDA on CPU)")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.clean) if l.strip()]
    if a.limit:
        recs = recs[:a.limit]
    weights = [float(x) for x in a.level_weights.split(',')]
    tasks = build_tasks(recs, a.variants, weights, a.seed)
    convs = [[{"role": "user", "content": PROMPTS[style].format(t=r["caption"])}]
             for (r, lvl, style) in tasks]
    print(f"### {len(recs)} queries x variants={a.variants} -> {len(tasks)} rewrites", flush=True)

    if a.dry_run:
        texts = [r["caption"] for (r, lvl, style) in tasks]   # identity rewrite
    else:
        from vllm import LLM, SamplingParams
        # small max_num_seqs + modest max_model_len keep the KV cache tiny so the engine fits the
        # little free VRAM on the shared cluster (default max_num_seqs=256 OOMs at warmup).
        llm = LLM(model=a.model, dtype='float16', attention_backend='TRITON_ATTN',
                  tensor_parallel_size=a.tp,
                  gpu_memory_utilization=a.gpu_mem, max_model_len=384, max_num_seqs=32,
                  enforce_eager=True)
        sp = SamplingParams(temperature=a.temperature, max_tokens=a.max_tokens)
        outs = llm.chat(convs, sp)
        texts = [clean_out(o.outputs[0].text) for o in outs]

    n_fallback = 0
    with open(a.out, 'w') as f:
        for (r, lvl, style), txt in zip(tasks, texts):
            noisy = txt
            if not noisy.strip():
                noisy = r["caption"]
                n_fallback += 1
            f.write(json.dumps({"caption": noisy, "image_id": r["image_id"],
                                "style": style, "level": lvl}) + "\n")
    print(f"### wrote {len(tasks)} candidates -> {a.out}  (fallback-to-clean: {n_fallback})", flush=True)
    import collections
    by = collections.Counter(lvl for (_, lvl, _) in tasks)
    print(f"### level mix: {dict(by)}", flush=True)
    for (r, lvl, style), txt in list(zip(tasks, texts))[:6]:
        print(f"\n[{lvl}/{style}] CLEAN: {r['caption'][:130]}")
        print(f"[{lvl}/{style}] NOISY: {txt[:130]}")


if __name__ == '__main__':
    main()
