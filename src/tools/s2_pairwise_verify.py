"""L2 — PAIRWISE tie-break verify: for tied candidate pairs (both verifiers rate both
images plausible), show Qwen BOTH images and ask which matches the caption better.
Pointwise p_yes saturates >0.9 on near-duplicate scenes; a forced A/B choice does not.

Reads  submissions/l2_pairwise.json   [[query, imgA, imgB], ...]  (imgA = current top1)
Writes <out>/pairwise_cache.json      {query: {"imgA|imgB": p_first_wins}}
Each pair is asked in BOTH orders and averaged to cancel position bias
(p_first_wins is stored per ORDER; the consumer averages p(A first) and 1-p(B first)).

Run inside ssdc-stage2 (same Turing-safe config as s2_logprob_verify):
  python3 tools/s2_pairwise_verify.py --pairs submissions/l2_pairwise.json \
      --gallery_dir /dev/shm/s2/test --model_dir /dev/shm/s2/model --name s2cache_pairwise
"""
import os
import gc
import json
import math
import argparse
import datetime

import torch
from PIL import Image

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rerank_stage2 import load_jsonl  # noqa

PAIR_PROMPT = """You are an expert in Person Re-identification and Anomaly Detection.
You are shown TWO images: the FIRST image is option A, the SECOND image is option B.
Text: {cap}

Which image does this text describe more accurately? Consider the primary person's
appearance (gender, clothing colors/type, accessories) and action (e.g., falling,
fighting, walking). Ignore background/lighting differences.

Answer STRICTLY with a single letter: "A" or "B"."""


class QwenPairwise:
    def __init__(self, model_dir):
        from vllm import LLM
        from transformers import AutoProcessor
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        tp = int(os.environ.get("SSDC_TP_SIZE", "1"))
        util = float(os.environ.get("SSDC_GPU_UTIL", "0.62"))
        max_len = int(os.environ.get("SSDC_MAX_LEN", "1536"))
        attn = os.environ.get("SSDC_ATTN_BACKEND", "TRITON_ATTN")
        self.llm = LLM(model=model_dir, tensor_parallel_size=tp, gpu_memory_utilization=util,
                       max_num_seqs=int(os.environ.get("SSDC_MAX_SEQS", "8")),
                       max_model_len=max_len, enforce_eager=True, disable_log_stats=True,
                       trust_remote_code=True, limit_mm_per_prompt={"image": 2},
                       enable_chunked_prefill=True,
                       max_num_batched_tokens=int(os.environ.get("SSDC_MAX_BATCHED_TOKENS", "1024")),
                       dtype=torch.float16, attention_config={"backend": attn},
                       quantization=os.environ.get("SSDC_QUANT") or None,
                       mm_processor_kwargs={"max_pixels": 50176})
        self.processor = AutoProcessor.from_pretrained(model_dir)

    @torch.no_grad()
    def choose(self, caps, img_firsts, img_seconds, bs=8, n_logprobs=10):
        """-> p(first image wins) per (caption, imgA, imgB)."""
        from vllm import SamplingParams
        from qwen_vl_utils import process_vision_info
        out = []
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=n_logprobs)

        def load_rgb(p):
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                return Image.new("RGB", (224, 224), (128, 128, 128))

        for s in range(0, len(caps), bs):
            messages = [[{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": [
                             {"type": "image", "image": load_rgb(pa),
                              "min_pixels": 50176, "max_pixels": 50176},
                             {"type": "image", "image": load_rgb(pb),
                              "min_pixels": 50176, "max_pixels": 50176},
                             {"type": "text", "text": PAIR_PROMPT.format(cap=c)}]}]
                        for c, pa, pb in zip(caps[s:s+bs], img_firsts[s:s+bs], img_seconds[s:s+bs])]
            prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                       for m in messages]
            inputs = [{"prompt": pr, "multi_modal_data": {"image": process_vision_info(m)[0]}}
                      for pr, m in zip(prompts, messages)]
            res = self.llm.generate(inputs, sampling_params=sp)
            for o in res:
                out.append(self._p_first(o.outputs[0].logprobs))
            del inputs, res
            torch.cuda.empty_cache(); gc.collect()
        return out

    @staticmethod
    def _p_first(logprobs):
        if not logprobs:
            return 0.5
        pa = pb = 0.0
        for lp in logprobs[0].values():
            tok = (lp.decoded_token or '').strip().upper()
            if tok == 'A':
                pa += math.exp(lp.logprob)
            elif tok == 'B':
                pb += math.exp(lp.logprob)
        return pa / (pa + pb) if (pa + pb) > 0 else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='submissions/l2_pairwise.json')
    ap.add_argument('--name', default='s2cache_pairwise')
    ap.add_argument('--out_root', default='submissions')
    ap.add_argument('--queries', default='data/localeval_dedup/queries_competition.jsonl')
    ap.add_argument('--gallery_dir', default='data/COMPETITION/test')
    ap.add_argument('--model_dir', default='/workspace/SSDC/checkpoint/Qwen3-VL-8B-Instruct-AWQ-4bit')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--save_every', type=int, default=10, help='checkpoint every N batches')
    a = ap.parse_args()

    pairs = json.load(open(a.pairs))          # [[q, imgA, imgB], ...]
    cap_of = {r['query_index_comp']: r['caption'] for r in load_jsonl(a.queries)}
    run_dir = os.path.join(a.out_root, a.name)
    os.makedirs(run_dir, exist_ok=True)
    cache_path = os.path.join(run_dir, 'pairwise_cache.json')
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path)).get('pw', {})
        print(f"### resuming: {sum(len(v) for v in cache.values())} orders cached", flush=True)

    # each logical pair -> two ordered tasks (A,B) and (B,A); key "first|second"
    tasks = []
    for q, A, B in pairs:
        for x, y in ((A, B), (B, A)):
            if f"{x}|{y}" in cache.get(q, {}):
                continue
            fx, fy = os.path.join(a.gallery_dir, x), os.path.join(a.gallery_dir, y)
            if not (os.path.isfile(fx) and os.path.isfile(fy)):
                continue
            tasks.append((q, x, y, cap_of[q], fx, fy))
    print(f"### {len(pairs)} pairs -> {len(tasks)} ordered calls to run", flush=True)
    if not tasks:
        print("### nothing to do"); return

    model = QwenPairwise(a.model_dir)

    def flush():
        json.dump({"pairs_file": a.pairs, "model_dir": a.model_dir,
                   "updated": datetime.datetime.now().isoformat(timespec='seconds'), "pw": cache},
                  open(cache_path + '.tmp', 'w'))
        os.replace(cache_path + '.tmp', cache_path)

    chunk = a.bs * a.save_every
    for s in range(0, len(tasks), chunk):
        part = tasks[s:s + chunk]
        ps = model.choose([t[3] for t in part], [t[4] for t in part], [t[5] for t in part], bs=a.bs)
        for (q, x, y, _, _, _), p in zip(part, ps):
            cache.setdefault(q, {})[f"{x}|{y}"] = round(float(p), 4)
        flush()
        print(f"### progress {min(s+chunk, len(tasks))}/{len(tasks)}", flush=True)
    flush()
    print(f"### wrote {cache_path}", flush=True)


if __name__ == '__main__':
    main()
