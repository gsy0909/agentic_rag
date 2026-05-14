import json
import sys
import os

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("storage", "compression_stats.jsonl")

records = []
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

if not records:
    print("No records found.")
    sys.exit(0)

total = len(records)
avg_ratio = sum(r["ratio"] for r in records) / total
skipped = sum(1 for r in records if r["ratio"] >= 0.99)
avg_orig = sum(r["orig_words"] for r in records) / total
avg_comp = sum(r["comp_words"] for r in records) / total

print(f"Total docs compressed : {total}")
print(f"Average orig words    : {avg_orig:.1f}")
print(f"Average comp words    : {avg_comp:.1f}")
print(f"Average ratio         : {avg_ratio:.2%}")
print(f"Pass-through (ratio≥0.99): {skipped} ({skipped/total:.1%})")

turns = {}
for r in records:
    t = r.get("turn", 1)
    turns.setdefault(t, []).append(r["ratio"])
if len(turns) > 1:
    print("\nRatio by turn:")
    for t in sorted(turns):
        avg = sum(turns[t]) / len(turns[t])
        print(f"  Turn {t}: {avg:.2%}  (n={len(turns[t])})")
