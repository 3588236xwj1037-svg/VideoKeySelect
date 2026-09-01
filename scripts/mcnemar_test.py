import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

METHODS = ["uniform", "clip_topk", "qatss", "qatss_v2"]


def load_records(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ("answers", "results", "data", "items"):
            if isinstance(obj.get(key), list):
                return obj[key]

        # 支持以题目 ID 为键的字典结构
        rows = []
        for key, value in obj.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("_key", key)
                rows.append(row)
        return rows

    raise ValueError(f"不支持的 JSON 结构: {type(obj)}")


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {
            "true", "1", "yes", "correct", "是", "正确"
        }
    return None


def get_correct(row, method):
    containers = []

    for key in ("methods", "method_results", "predictions", "evaluations"):
        value = row.get(key)
        if isinstance(value, dict) and method in value:
            containers.append(value[method])

    if method in row:
        containers.append(row[method])

    for item in containers:
        if isinstance(item, dict):
            for key in ("correct", "is_correct", "ok"):
                if key in item:
                    result = to_bool(item[key])
                    if result is not None:
                        return result
        else:
            result = to_bool(item)
            if result is not None:
                return result

    return None


def exact_mcnemar_p(b01, b10):
    n = b01 + b10
    if n == 0:
        return 1.0

    k = min(b01, b10)
    lower_tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * lower_tail)


def get_id(row, index):
    video = row.get("video_id", row.get("video"))
    qid = row.get("qid", row.get("question_id", row.get("id")))

    if video is None and qid is None:
        return str(row.get("_key", index))

    return f"{video}:{qid}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("answers", nargs="+")
    args = parser.parse_args()

    all_rows = []
    seen = set()

    for answer_path in args.answers:
        for i, row in enumerate(load_records(answer_path)):
            key = get_id(row, i)

            if key in seen:
                raise ValueError(f"发现重复题目: {key}")

            seen.add(key)
            all_rows.append(row)

    print(f"读取题目数: {len(all_rows)}")

    parsed = {}
    for i, row in enumerate(all_rows):
        key = get_id(row, i)
        parsed[key] = {}

        for method in METHODS:
            value = get_correct(row, method)
            if value is not None:
                parsed[key][method] = value

    for method in METHODS:
        count = sum(method in result for result in parsed.values())
        print(f"{method}: 找到 {count}/{len(parsed)} 条正确性结果")

    print("\nMcNemar exact test:")
    for method_a, method_b in combinations(METHODS, 2):
        both_correct = 0
        both_wrong = 0
        a_only = 0       # A 对，B 错
        b_only = 0       # A 错，B 对
        usable = 0

        for result in parsed.values():
            if method_a not in result or method_b not in result:
                continue

            usable += 1
            a = result[method_a]
            b = result[method_b]

            if a and b:
                both_correct += 1
            elif not a and not b:
                both_wrong += 1
            elif a and not b:
                a_only += 1
            else:
                b_only += 1

        p_value = exact_mcnemar_p(a_only, b_only)
        difference = b_only - a_only

        print(
            f"{method_a} vs {method_b}: "
            f"usable={usable}, "
            f"A对B错={a_only}, "
            f"A错B对={b_only}, "
            f"净变化={difference}, "
            f"p={p_value:.6f}"
        )


if __name__ == "__main__":
    main()