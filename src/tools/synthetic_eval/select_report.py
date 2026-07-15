"""#3 Part B: contamination-aware selection view across configs. Reads results.csv (each config
run once on 'clean' and once on 'noisy'), pairs the two, and reports clean R@1, noisy R@1,
drop=(clean-noisy), and per-difficulty bins. Pure python (no torch) -> runs anywhere.

  python tools/synthetic_eval/select_report.py --csv data/synthetic_eval/results.csv

Why (spec 23-06 §5.1/§5.2/§6): cmp.pth trained on ALL 1M incl. the val sources -> it memorised
val images -> raw val numbers are inflated. Guards:
  - SELECT on noisy R@1 (the noisy strings are genuinely unseen -> real robustness).
  - drop = clean - noisy: image-memorisation inflates clean & noisy ~equally, so the drop largely
    cancels it -> a cleaner noise-robustness signal.
  - if the NOISY-R@1 ranking and the (low-)DROP ranking disagree at the TOP, contamination may be
    distorting the ranking -> verify with gt_local before trusting val (don't just pick top-noisy).
"""
import csv
import argparse


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/synthetic_eval/results.csv')
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    by = {}
    for r in rows:
        by.setdefault(r['config_tag'], {})[r.get('split', '?')] = r

    table = []
    for cfg, d in by.items():
        clean = _f(d.get('clean', {}).get('R@1'))
        noisy = _f(d.get('noisy', {}).get('R@1'))
        nz = d.get('noisy', {})
        table.append({'cfg': cfg, 'clean': clean, 'noisy': noisy,
                      'drop': (clean - noisy) if (clean is not None and noisy is not None) else None,
                      'easy': _f(nz.get('R@1_easy')), 'medium': _f(nz.get('R@1_medium')),
                      'hard': _f(nz.get('R@1_hard'))})

    def s(v, w=7, p=2):
        return (f"{v:.{p}f}".rjust(w)) if v is not None else '-'.rjust(w)

    print(f"\n{'config':28s} {'clean':>7s} {'noisy':>7s} {'drop':>6s} | {'easy':>6s} {'med':>6s} {'hard':>6s}")
    print('-' * 74)
    for t in sorted(table, key=lambda x: (x['noisy'] is None, -(x['noisy'] or 0))):
        print(f"{t['cfg']:28s} {s(t['clean'])} {s(t['noisy'])} {s(t['drop'], 6)} | "
              f"{s(t['easy'], 6)} {s(t['medium'], 6)} {s(t['hard'], 6)}")

    full = [t for t in table if t['drop'] is not None]
    if len(full) >= 2:
        by_noisy = sorted(full, key=lambda x: -x['noisy'])
        by_drop = sorted(full, key=lambda x: x['drop'])
        print(f"\nSELECT (best noisy R@1): {by_noisy[0]['cfg']}  (noisy {by_noisy[0]['noisy']:.2f})")
        if by_noisy[0]['cfg'] != by_drop[0]['cfg']:
            print(f"  WARNING: best-noisy ({by_noisy[0]['cfg']}) != lowest-drop ({by_drop[0]['cfg']}, "
                  f"drop {by_drop[0]['drop']:.2f}).")
            print("  Noisy-rank and drop-rank disagree -> contamination may be distorting val. "
                  "VERIFY with gt_local before picking.")
        else:
            print("  OK: best-noisy is also lowest-drop -> ranking consistent (lower contamination concern).")
    else:
        print("\n(need >=2 configs with BOTH clean & noisy runs for the drop / contradiction guard)")


if __name__ == '__main__':
    main()
