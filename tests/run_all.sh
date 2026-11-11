#!/bin/sh
# Run this repo's tests the way they are written: as scripts.
#
# `pytest tests` reports "no tests ran" for this suite on purpose -- the files are
# module-level asserts with a print at the end, so pytest only executes them as an import
# side effect and collects zero items. A pass count from pytest therefore says nothing
# about this repo's checks; this loop proves each file ran and printed its own line.
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
# Running a file as a script puts tests/ on sys.path, not the repo root, so the repo's own
# packages and the scripts/ helpers only import if they are put there.
export PYTHONPATH="$PWD:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
for f in tests/test_*.py; do
  out=$("$PY" "$f" 2>&1) || { printf 'FAIL %s\n%s\n' "$f" "$out"; exit 1; }
  printf '%-34s %s\n' "$f" "$(printf '%s\n' "$out" | grep -i 'passed\|ok:' | tail -1)"
done
