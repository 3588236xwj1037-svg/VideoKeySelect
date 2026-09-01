import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

MANIFEST = Path("data/nextqa/manifests/dev1000.jsonl")
QLORA = Path("results/dev1000_qlora_eval/answers.json")
BASE = Path("results/dev1000_qwen_eval/answers.json")
OUT = Path("results/dev1000_qlora_analysis/summary.json")
METHODS = ["uniform", "clip_topk", "qatss"]


def load_records(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{path} 顶层必须是 list")

    result = {}
    for row in rows:
        sid = row.get("sample_id")
        if sid is None:
            sid = f"{row['video_id']}_{row['qid']}"
        if sid in result:
            raise ValueError(f"{path} 存在重复 sample_id: {sid}")
        result[sid] = row
    return result


def exact_mcnemar_p(a_correct_b_wrong, a_wrong_b_correct):
    n = a_correct_b_wrong + a_wrong_b_correct
    if n == 0:
        return 1.0
    k = min(a_correct_b_wrong, a_wrong_b_correct)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def accuracy(records, method, ids):
    values = [records[sid]["methods"][method]["is_correct"] for sid in ids]
    return {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}


def compare(records_a, records_b, method_a, method_b, ids, label_a=None, label_b=None):
    a_only = b_only = both = neither = 0
    for sid in ids:
        a = records_a[sid]["methods"][method_a]["is_correct"]
        b = records_b[sid]["methods"][method_b]["is_correct"]
        if a and not b:
            a_only += 1
        elif b and not a:
            b_only += 1
        elif a and b:
            both += 1
        else:
            neither += 1

    return {
        "A": label_a or method_a,
        "B": label_b or method_b,
        "A_correct_B_wrong": a_only,
        "A_wrong_B_correct": b_only,
        "both_correct": both,
        "both_wrong": neither,
        "net_B_minus_A": b_only - a_only,
        "mcnemar_exact_p": exact_mcnemar_p(a_only, b_only),
    }

manifest_rows = [
    json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
types = {f"{row['video_id']}_{row['qid']}": row["type"] for row in manifest_rows}

qlora = load_records(QLORA)
base = load_records(BASE)
ids = list(types)

assert set(ids) == set(qlora), "QLoRA answers 与 dev1000 manifest 不一致"
assert set(ids) == set(base), "原始模型 answers 与 dev1000 manifest 不一致"

for sid in ids:
    assert qlora[sid]["correct_answer"] == base[sid]["correct_answer"]
    for method in METHODS:
        assert method in qlora[sid]["methods"]
        assert method in base[sid]["methods"]

by_type = defaultdict(list)
for sid in ids:
    by_type[types[sid]].append(sid)

summary = {
    "integrity": {"questions": len(ids), "types": {t: len(x) for t, x in by_type.items()}},
    "qlora_overall": {m: accuracy(qlora, m, ids) for m in METHODS},
    "base_overall": {m: accuracy(base, m, ids) for m in METHODS},
    "qlora_by_type": {
        t: {m: accuracy(qlora, m, type_ids) for m in METHODS}
        for t, type_ids in sorted(by_type.items())
    },
    "qlora_pairwise": {
        f"{a}_vs_{b}": compare(qlora, qlora, a, b, ids)
        for a, b in itertools.combinations(METHODS, 2)
    },
    "base_vs_qlora": {
        m: compare(
            base,
            qlora,
            m,
            m,
            ids,
            label_a=f"base_{m}",
            label_b=f"qlora_{m}",
        )
        for m in METHODS
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nQLoRA overall:")
for m, x in summary["qlora_overall"].items():
    print(f"{m:10s} {x['correct']}/{x['total']} = {x['accuracy']:.4f}")

print("\nQLoRA by type:")
print("type  n    uniform  clip_topk  qatss")
for t, values in summary["qlora_by_type"].items():
    print(
        f"{t:4s} {values['uniform']['total']:3d} "
        f"{values['uniform']['accuracy']:.4f}   "
        f"{values['clip_topk']['accuracy']:.4f}    "
        f"{values['qatss']['accuracy']:.4f}"
    )

print("\nQLoRA pairwise:")
for name, x in summary["qlora_pairwise"].items():
    print(
        f"{name}: A对B错={x['A_correct_B_wrong']}, "
        f"A错B对={x['A_wrong_B_correct']}, "
        f"净B-A={x['net_B_minus_A']}, p={x['mcnemar_exact_p']:.6f}"
    )

print("\nBase -> QLoRA:")
for method, x in summary["base_vs_qlora"].items():
    print(
        f"{method}: 原模型对/QLoRA错={x['A_correct_B_wrong']}, "
        f"原模型错/QLoRA对={x['A_wrong_B_correct']}, "
        f"净QLoRA-原模型={x['net_B_minus_A']}, "
        f"p={x['mcnemar_exact_p']:.6f}"
    )

print(f"\nSaved: {OUT}")