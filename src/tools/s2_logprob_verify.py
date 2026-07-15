"""Stage-2 verify with CONTINUOUS P(Yes) from first-token logprobs (not binary Yes/No).

Fixes two failure modes measured on fuse_gated_best_s2:
  - 87 rank-2..5 failures were Yes/Yes TIES  -> continuous p_yes breaks them
  - 150 were No/No (verifier blind)          -> soft scores still carry signal below 0.5
Also verifies an EXTENDED candidate pool (approach 3): gated top-R  UNION  recall-max
top-K extras, lifting the reachable GT ceiling 94.34% -> 97.4% (K=15).

Writes an IMAGE-KEYED cache {query: {image: p_yes}} to submissions/<name>/verify_logprob.json
(checkpointed every --save_every batches, resumable). Blending/sweeping is offline in
tools/s2_logprob_blend.py -- this script only spends GPU.

Run INSIDE ssdc-stage2, cwd /workspace/SSDC:
  CUDA_VISIBLE_DEVICES set via docker exec -e (NOT in-script).
  python3 tools/s2_logprob_verify.py --run fuse_gated_best \
      --ext_run fuse_cmp512_vith_siglip_recallmax_w13 --ext_k 15 \
      --model_dir /workspace/SSDC/checkpoint/Qwen3-VL-8B-Instruct-AWQ-4bit \
      --name s2cache_logprob_gr15
"""
import os
import gc
import json
import argparse
import datetime

import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rerank_stage2 import VERIFY_PROMPT, load_jsonl  # noqa


class QwenLogprobVerifier:
    """Same Turing-safe vLLM config as rerank_stage2.QwenReranker, but returns P(Yes)
    from the first generated token's top-logprobs instead of parsing generated text."""

    def __init__(self, model_dir):
        from vllm import LLM
        from transformers import AutoProcessor
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        tp = int(os.environ.get("SSDC_TP_SIZE", "2"))
        util = float(os.environ.get("SSDC_GPU_UTIL", "0.45"))
        max_len = int(os.environ.get("SSDC_MAX_LEN", "1536"))
        attn = os.environ.get("SSDC_ATTN_BACKEND", "TRITON_ATTN")
        quant = os.environ.get("SSDC_QUANT") or None
        self.llm = LLM(model=model_dir, tensor_parallel_size=tp, gpu_memory_utilization=util,
                       max_num_seqs=int(os.environ.get("SSDC_MAX_SEQS", "64")),
                       max_model_len=max_len, enforce_eager=True, disable_log_stats=True,
                       trust_remote_code=True, limit_mm_per_prompt={"image": 1},
                       enable_chunked_prefill=True,
                       max_num_batched_tokens=int(os.environ.get("SSDC_MAX_BATCHED_TOKENS", "2048")),
                       dtype=torch.float16, attention_config={"backend": attn},
                       quantization=quant,
                       mm_processor_kwargs={"max_pixels": 50176})
        self.processor = AutoProcessor.from_pretrained(model_dir)

    @torch.no_grad()
    def verify(self, caps, image_paths, bs=16, n_logprobs=10):
        """-> list of p_yes in [0,1]: p(Yes) / (p(Yes) + p(No)) over first-token logprobs."""
        from vllm import SamplingParams
        from qwen_vl_utils import process_vision_info
        out = []
        # n_logprobs kept small: gather_logprobs allocates per-seq full-vocab tensors and
        # OOMs on 11GB Turing cards when many seqs finish in the same step
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=n_logprobs)
        def load_rgb(p):
            # staged gallery may contain a truncated file from an interrupted copy;
            # a grey placeholder keeps batch shapes while scoring ~0 for that pair
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                return Image.new("RGB", (224, 224), (128, 128, 128))
        for s in range(0, len(caps), bs):
            cap_b, img_b = caps[s:s + bs], image_paths[s:s + bs]
            messages = [[{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": [
                             {"type": "image", "image": load_rgb(p),
                              "min_pixels": 50176, "max_pixels": 50176},
                             {"type": "text", "text": VERIFY_PROMPT.format(cap=c)}]}]
                        for c, p in zip(cap_b, img_b)]
            prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages]
            inputs = [{"prompt": pr, "multi_modal_data": {"image": process_vision_info(m)[0]}}
                      for pr, m in zip(prompts, messages)]
            res = self.llm.generate(inputs, sampling_params=sp)
            for o in res:
                out.append(self._p_yes(o.outputs[0].logprobs))
            del inputs, res
            torch.cuda.empty_cache(); gc.collect()
        return out

    @staticmethod
    def _p_yes(logprobs):
        import math
        if not logprobs:
            return 0.0
        py = pn = 0.0
        for lp in logprobs[0].values():
            tok = (lp.decoded_token or '').strip().lower()
            if tok in ('yes', 'y'):
                py += math.exp(lp.logprob)
            elif tok in ('no', 'n'):
                pn += math.exp(lp.logprob)
        return py / (py + pn) if (py + pn) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='fuse_gated_best')
    ap.add_argument('--ext_run', default='fuse_cmp512_vith_siglip_recallmax_w13')
    ap.add_argument('--ext_k', type=int, default=15, help='extras = ext_run top-K not in run top-`round`')
    ap.add_argument('--round', type=int, default=10, help='verify the run top-`round`')
    ap.add_argument('--out_root', default='submissions')
    ap.add_argument('--queries', default='data/localeval_dedup/queries_competition.jsonl')
    ap.add_argument('--cap_field', default='caption')
    ap.add_argument('--gallery_dir', default='data/COMPETITION/test')
    ap.add_argument('--model_dir', default='/workspace/SSDC/checkpoint/Qwen3-VL-8B-Instruct-AWQ-4bit')
    ap.add_argument('--name', default='s2cache_logprob_gr15')
    ap.add_argument('--save_every', type=int, default=20, help='checkpoint cache every N batches')
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--logprobs', type=int, default=10)
    ap.add_argument('--pairs_file', default=None,
                    help='JSON {query: [gallery_file,...]} — verify exactly these pairs instead of the pool logic')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    scores = json.load(open(os.path.join(a.out_root, a.run, 'scores.json')))
    ext = json.load(open(os.path.join(a.out_root, a.ext_run, 'scores.json')))
    cap_of = {r['query_index_comp']: r[a.cap_field] for r in load_jsonl(a.queries)}
    qids = list(scores.keys())
    if a.limit:
        qids = qids[:a.limit]

    run_dir = os.path.join(a.out_root, a.name)
    os.makedirs(run_dir, exist_ok=True)
    cache_path = os.path.join(run_dir, 'verify_logprob.json')
    sem = {}
    if os.path.exists(cache_path):
        sem = json.load(open(cache_path)).get('sem', {})
        print(f"### resuming: {sum(len(v) for v in sem.values())} pairs already cached", flush=True)

    explicit = json.load(open(a.pairs_file)) if a.pairs_file else None
    pair_caps, pair_imgs, pair_key = [], [], []
    n_missing = 0
    for q in qids:
        if explicit is not None:
            pool = explicit.get(q, [])
        else:
            pool = [n for n, _ in scores[q][:a.round]]
            seen = set(pool)
            pool += [n for n, _ in ext.get(q, [])[:a.ext_k] if n not in seen]
        done = sem.get(q, {})
        for name in pool:
            if name in done:
                continue
            path = os.path.join(a.gallery_dir, name)
            # gallery staged from a flaky NFS: a few images may be absent -> skip the
            # pair (blend treats missing sem as neutral) instead of crashing the run
            if not os.path.isfile(path):
                n_missing += 1
                continue
            pair_caps.append(cap_of[q])
            pair_imgs.append(path)
            pair_key.append((q, name))
    print(f"### {len(qids)} queries -> {len(pair_key)} pairs to verify "
          f"(round={a.round} + ext top-{a.ext_k}, {n_missing} skipped: image absent)", flush=True)
    if not pair_key:
        print("### nothing to do"); return

    verifier = QwenLogprobVerifier(a.model_dir)

    def flush_cache():
        json.dump({"run": a.run, "ext_run": a.ext_run, "ext_k": a.ext_k, "round": a.round,
                   "cap_field": a.cap_field, "model_dir": a.model_dir, "scoring": "p_yes_logprob",
                   "updated": datetime.datetime.now().isoformat(timespec='seconds'), "sem": sem},
                  open(cache_path + '.tmp', 'w'))
        os.replace(cache_path + '.tmp', cache_path)

    chunk = a.bs * a.save_every
    done_n = 0
    for s in range(0, len(pair_key), chunk):
        keys = pair_key[s:s + chunk]
        ps = verifier.verify(pair_caps[s:s + chunk], pair_imgs[s:s + chunk], bs=a.bs,
                             n_logprobs=a.logprobs)
        for (q, name), p in zip(keys, ps):
            sem.setdefault(q, {})[name] = round(float(p), 4)
        done_n += len(keys)
        flush_cache()
        print(f"### progress {done_n}/{len(pair_key)}", flush=True)

    flush_cache()
    print(f"### wrote {cache_path} ({sum(len(v) for v in sem.values())} pairs)", flush=True)


if __name__ == '__main__':
    main()
