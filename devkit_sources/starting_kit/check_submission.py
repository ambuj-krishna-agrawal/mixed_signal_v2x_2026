#!/usr/bin/env python3
"""Validate a Mixed Signals submission before uploading to Codabench.

Usage:
    python3 check_submission.py <folder-or-zip> [--frames validation_frames.txt]

Checks every *.txt file for the 9-column `cls cx cy cz dx dy dz yaw score`
format, sane value ranges, and (optionally) that all required frame ids are
present. Exits 0 if the submission is valid, 1 otherwise. Pure standard library.
"""
import argparse
import os
import sys
import glob
import zipfile
import tempfile

VALID_CLASSES = {1, 2, 3}
RANGE = 51.2


def iter_txt(path):
    """Yield (display_name, list_of_lines) for each .txt in a folder or zip."""
    if path.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = tmp
    files = glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
    for f in sorted(files):
        with open(f, "r") as fh:
            yield os.path.splitext(os.path.basename(f))[0], f, fh.readlines()


def check_file(name, lines):
    errs, warns, ndet = [], [], 0
    for i, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        p = line.split()
        if len(p) != 9:
            errs.append(f"  line {i}: expected 9 columns, got {len(p)}: '{line}'")
            continue
        try:
            cls = int(float(p[0]))
            vals = [float(x) for x in p[1:]]
        except ValueError:
            errs.append(f"  line {i}: non-numeric value: '{line}'")
            continue
        cx, cy, cz, dx, dy, dz, yaw, score = vals
        ndet += 1
        if cls not in VALID_CLASSES:
            warns.append(f"  line {i}: class {cls} not in {{1,2,3}} (ignored by scorer)")
        if not (0.0 <= score <= 1.0):
            warns.append(f"  line {i}: score {score} outside [0,1]")
        if min(dx, dy, dz) <= 0:
            errs.append(f"  line {i}: non-positive box dimension: {dx},{dy},{dz}")
        if abs(cx) > RANGE or abs(cy) > RANGE:
            warns.append(f"  line {i}: centre outside +/-{RANGE} m (dropped by scorer)")
    return errs, warns, ndet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="folder or .zip of *.txt files")
    ap.add_argument("--frames", help="text file of required frame ids (one per line)")
    args = ap.parse_args()

    seen, total_errs, total_det, total_files = set(), 0, 0, 0
    for stem, fpath, lines in iter_txt(args.submission):
        seen.add(stem)
        total_files += 1
        errs, warns, ndet = check_file(stem, lines)
        total_det += ndet
        if errs or warns:
            print(f"[{stem}] ({os.path.basename(fpath)})")
            for e in errs:
                print("  ERROR" + e)
            for w in warns:
                print("  warn " + w)
        total_errs += len(errs)

    if total_files == 0:
        print("ERROR: no .txt files found in submission.")
        sys.exit(1)

    if args.frames and os.path.isfile(args.frames):
        required = {
            l.split("#", 1)[0].strip()
            for l in open(args.frames)
            if l.split("#", 1)[0].strip()
        }
        missing = sorted(required - seen)
        extra = sorted(seen - required)
        if missing:
            print(f"ERROR: {len(missing)} required frames missing, e.g. {missing[:5]}")
            total_errs += len(missing)
        if extra:
            print(f"warn: {len(extra)} files are not in the required frame list, "
                  f"e.g. {extra[:5]}")

    print("-" * 60)
    print(f"Files: {total_files}  detections: {total_det}  errors: {total_errs}")
    if total_errs:
        print("INVALID: fix the errors above before submitting.")
        sys.exit(1)
    print("OK: submission looks valid. Zip the *.txt files and upload.")


if __name__ == "__main__":
    main()
