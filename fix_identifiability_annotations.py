"""One-time corpus-maintenance pass: recompute rel_crlb/log10_cond in
place on every existing shard under $PLANTFORGE_DATA/corpus/, using the
fixed identifiability() (2026-07-27 pinv-bug fix) -- WITHOUT
re-simulating. Reuses each shard's own u and theta exactly (identical to
what corpus.gen_cell used when the shard was first generated), so u, y,
theta, keys, dt, family, and excitation are byte-identical before and
after; only rel_crlb and log10_cond change.

    PLANTFORGE_DATA=... python -m plantforge.fix_identifiability_annotations
"""
from __future__ import annotations

import torch

from .corpus import OUT
from .identifiability import identifiability


def main():
    paths = sorted(OUT.glob("*.pt"))
    print(f"found {len(paths)} shards under {OUT}")
    total_rows = total_changed = total_zero_before = total_zero_after = 0
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        keys = shard["keys"]
        theta = shard["theta"]
        p = {k: theta[:, i] for i, k in enumerate(keys)}
        old_rel_crlb = shard["rel_crlb"]
        idn = identifiability(shard["family"], p, shard["u"], shard["dt"])
        new_rel_crlb, new_log10_cond = idn["rel_crlb"], idn["log10_cond"]

        changed = int((~torch.isclose(old_rel_crlb, new_rel_crlb, rtol=1e-4, atol=1e-6)).sum())
        zero_before = int((old_rel_crlb == 0).sum())
        zero_after = int((new_rel_crlb == 0).sum())
        total_rows += old_rel_crlb.numel()
        total_changed += changed
        total_zero_before += zero_before
        total_zero_after += zero_after
        print(f"  {path.name}: changed={changed} zero_entries {zero_before}->{zero_after}")

        shard["rel_crlb"] = new_rel_crlb
        shard["log10_cond"] = new_log10_cond
        torch.save(shard, path)

    print(f"SUMMARY: {len(paths)} shards, {total_rows} total (instance,param) entries, "
          f"{total_changed} changed, zero_entries {total_zero_before} -> {total_zero_after}")


if __name__ == "__main__":
    main()
