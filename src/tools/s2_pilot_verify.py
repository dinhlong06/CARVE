"""L3 pilot — MODEL-AGNOSTIC verify over the pilot pair set, for auditioning non-Qwen
VLMs as a 4th verifier. Uses vLLM's llm.chat() so each model's own chat template and
multimodal pipeline are applied automatically (images passed as base64 data URLs).

Two answer-parsing modes:
  logprob : max_tokens=1, p_yes from first-token top-logprobs (Yes/No models)
  text    : max_tokens=160, parse the LAST standalone yes/no in the generated text
            (for thinking models like GLM-4.1V); p is 1.0/0.0

Writes {query: {image: p}} to <out_root>/<name>/verify_logprob.json (same schema as
s2_logprob_verify, so pilot_eval.py and the hybrid blends can consume it unchanged).

  python3 tools/s2_pilot_verify.py --model_dir /dev/shm/s2/models/phi35v \
      --name pilot_phi35v --pairs submissions/pilot_pairs.json --parse logprob
"""
import os
import gc
import json
import math
import base64
import argparse
import datetime

import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rerank_stage2 import VERIFY_PROMPT, load_jsonl  # noqa


def b64_url(path):
    with open(path, 'rb') as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


class GenericVerifier:
    def __init__(self, model_dir):
        if os.environ.get("SSDC_ALLOW_FP16_UNSTABLE") == "1":
            # gemma3 is soft-blocked from fp16 ("numerical instability") but Turing has no
            # bf16 and AWQ-Marlin has no fp32 -> the only path is fp16-at-own-risk; the
            # smoke degenerate-check downstream catches actual numeric garbage
            try:
                from vllm.config import model as _vcm
                for k in ("gemma3", "gemma3_text", "gemma2", "glm4"):
                    _vcm._FLOAT16_NOT_SUPPORTED_MODELS.pop(k, None)
                print("### fp16-unstable guard lifted (SSDC_ALLOW_FP16_UNSTABLE=1)", flush=True)
            except Exception as e:
                print(f"### fp16-unstable patch failed: {e}", flush=True)
        from vllm import LLM
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        tp = int(os.environ.get("SSDC_TP_SIZE", "1"))
        util = float(os.environ.get("SSDC_GPU_UTIL", "0.6"))
        max_len = int(os.environ.get("SSDC_MAX_LEN", "4096"))
        attn = os.environ.get("SSDC_ATTN_BACKEND", "TRITON_ATTN")
        self.llm = LLM(model=model_dir, tensor_parallel_size=tp, gpu_memory_utilization=util,
                       max_num_seqs=int(os.environ.get("SSDC_MAX_SEQS", "4")),
                       max_model_len=max_len, enforce_eager=True, disable_log_stats=True,
                       trust_remote_code=True, limit_mm_per_prompt={"image": 1},
                       enable_chunked_prefill=True,
                       max_num_batched_tokens=int(os.environ.get("SSDC_MAX_BATCHED_TOKENS", "2048")),
                       # gemma3 refuses fp16 (numerical instability) and Turing lacks bf16
                       # -> allow float32 via env (int4 weights stay int4; activations widen)
                       dtype=getattr(torch, os.environ.get("SSDC_DTYPE", "float16")),
                       attention_config={"backend": attn},
                       # ViT tower has its own attention selection; several archs (InternVL,
                       # SmolVLM) hardcode FA2 there, which Turing (SM75) lacks -> force SDPA
                       mm_encoder_attn_backend=os.environ.get("SSDC_MM_ATTN", "TORCH_SDPA"),
                       quantization=os.environ.get("SSDC_QUANT") or None)

    @torch.no_grad()
    def verify(self, caps, image_paths, parse='logprob', bs=8):
        from vllm import SamplingParams
        if parse == 'logprob':
            sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=10)
        else:
            sp = SamplingParams(temperature=0.0, max_tokens=160)
        out = []
        for s in range(0, len(caps), bs):
            convs = [[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": b64_url(p)}},
                        {"type": "text", "text": VERIFY_PROMPT.format(cap=c)}]}]
                     for c, p in zip(caps[s:s+bs], image_paths[s:s+bs])]
            res = self.llm.chat(convs, sampling_params=sp, use_tqdm=False)
            for o in res:
                if parse == 'logprob':
                    out.append(self._p_yes(o.outputs[0].logprobs))
                else:
                    out.append(self._parse_text(o.outputs[0].text))
            del convs, res
            torch.cuda.empty_cache(); gc.collect()
        return out

    @staticmethod
    def _p_yes(logprobs):
        if not logprobs:
            return None
        py = pn = 0.0
        for lp in logprobs[0].values():
            tok = (lp.decoded_token or '').strip().lower()
            if tok in ('yes', 'y'):
                py += math.exp(lp.logprob)
            elif tok in ('no', 'n'):
                pn += math.exp(lp.logprob)
        return py / (py + pn) if (py + pn) > 0 else None

    @staticmethod
    def _parse_text(text):
        import re
        t = text.lower()
        if '</think>' in t:
            t = t.split('</think>')[-1]
        hits = re.findall(r'\b(yes|no)\b', t)
        if not hits:
            return None
        return 1.0 if hits[-1] == 'yes' else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--pairs', default='submissions/pilot_pairs.json')
    ap.add_argument('--parse', choices=['logprob', 'text'], default='logprob')
    ap.add_argument('--out_root', default='submissions')
    ap.add_argument('--queries', default='data/localeval_dedup/queries_competition.jsonl')
    ap.add_argument('--gallery_dir', default='data/COMPETITION/test')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--save_every', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0, help='debug: only first N (query,img) pairs')
    a = ap.parse_args()

    pairs = json.load(open(a.pairs))
    cap_of = {r['query_index_comp']: r['caption'] for r in load_jsonl(a.queries)}
    run_dir = os.path.join(a.out_root, a.name)
    os.makedirs(run_dir, exist_ok=True)
    cache_path = os.path.join(run_dir, 'verify_logprob.json')
    sem = {}
    if os.path.exists(cache_path):
        sem = json.load(open(cache_path)).get('sem', {})
        print(f"### resuming {sum(len(v) for v in sem.values())} pairs", flush=True)

    tasks = []
    for q, imgs in pairs.items():
        for img in imgs:
            if img in sem.get(q, {}):
                continue
            p = os.path.join(a.gallery_dir, img)
            if os.path.isfile(p):
                tasks.append((q, img, cap_of[q], p))
    if a.limit:
        tasks = tasks[:a.limit]
    print(f"### {a.name}: {len(tasks)} pairs to verify (parse={a.parse})", flush=True)
    if not tasks:
        print("### nothing to do"); return

    v = GenericVerifier(a.model_dir)

    def flush():
        json.dump({"model_dir": a.model_dir, "parse": a.parse, "pairs_file": a.pairs,
                   "updated": datetime.datetime.now().isoformat(timespec='seconds'), "sem": sem},
                  open(cache_path + '.tmp', 'w'))
        os.replace(cache_path + '.tmp', cache_path)

    chunk = a.bs * a.save_every
    for s in range(0, len(tasks), chunk):
        part = tasks[s:s + chunk]
        ps = v.verify([t[2] for t in part], [t[3] for t in part], parse=a.parse, bs=a.bs)
        for (q, img, _, _), p in zip(part, ps):
            if p is not None:
                sem.setdefault(q, {})[img] = round(float(p), 4)
        flush()
        print(f"### progress {min(s+chunk, len(tasks))}/{len(tasks)}", flush=True)
    flush()
    n = sum(len(x) for x in sem.values())
    ys = sum(1 for x in sem.values() for p in x.values() if p >= 0.5)
    print(f"### wrote {cache_path} ({n} pairs, yes-rate {100*ys/max(n,1):.1f}%)", flush=True)


if __name__ == '__main__':
    main()
