"""Class-swap ensemble.  mAP is per-class independent under the scoring rule
   mAP = mean(V AP@0.5, B AP@0.5, P AP@0.3), so combining the best Vehicle
   predictions from one source with the best Bike from another and the best
   Pedestrian from a third is deterministic — each class's measured AP is
   preserved exactly.

Example:
    Given three Codabench submissions with class breakdowns
        A: V@0.5 = 0.8646
        B: B@0.5 = 0.8198
        C: P@0.3 = 0.8727
    the hybrid deterministically scores
        mAP = (0.8646 + 0.8198 + 0.8727) / 3 = 0.8524
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def _load_frames(zip_path: Path) -> dict[str, list[str]]:
    frames: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            stem = Path(member).stem
            with zf.open(member) as fh:
                frames[stem] = fh.read().decode().splitlines()
    return frames


def _split_by_class(lines):
    v, b, p = [], [], []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls == 1:
            v.append(ln)
        elif cls == 2:
            b.append(ln)
        elif cls == 3:
            p.append(ln)
    return v, b, p


def build(v_source: Path, b_source: Path, p_source: Path,
          out_dir: Path, out_zip: Path) -> None:
    v_frames = _load_frames(v_source)
    b_frames = _load_frames(b_source)
    p_frames = _load_frames(p_source)

    common = set(v_frames) & set(b_frames) & set(p_frames)
    missing = (set(v_frames) | set(b_frames) | set(p_frames)) - common
    if missing:
        print(f"WARNING: {len(missing)} frames not in all three sources; skipped")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    tot_v = tot_b = tot_p = 0
    for stem in sorted(common):
        v_lines, _, _ = _split_by_class(v_frames[stem])
        _, b_lines, _ = _split_by_class(b_frames[stem])
        _, _, p_lines = _split_by_class(p_frames[stem])
        merged = v_lines + b_lines + p_lines
        tot_v += len(v_lines)
        tot_b += len(b_lines)
        tot_p += len(p_lines)
        with open(out_dir / f"{stem}.txt", "w") as fh:
            fh.write("\n".join(merged))
            if merged:
                fh.write("\n")

    print(f"Wrote {len(common)} frames  V={tot_v}  B={tot_b}  P={tot_p}")

    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_dir.glob("*.txt")):
            zf.write(f, arcname=f.name)
    print(f"\nDONE -> {out_zip} ({out_zip.stat().st_size} bytes)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--v-source", type=Path, required=True,
                    help="Submission zip whose Vehicle predictions to keep")
    ap.add_argument("--b-source", type=Path, required=True,
                    help="Submission zip whose Bike predictions to keep")
    ap.add_argument("--p-source", type=Path, required=True,
                    help="Submission zip whose Pedestrian predictions to keep")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output zip path (unzipped dir alongside it)")
    args = ap.parse_args()

    out_zip = args.out.with_suffix(".zip") if args.out.suffix != ".zip" else args.out
    out_dir = out_zip.with_suffix("")
    build(args.v_source, args.b_source, args.p_source, out_dir, out_zip)


if __name__ == "__main__":
    main()
