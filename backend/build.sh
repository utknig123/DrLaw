#!/usr/bin/env bash
# exit on error
set -o errexit

echo "[build.sh] Running build script in $(pwd)"
echo "[build.sh] Python: $(python -V 2>&1)"

# ensure setuptools and wheel are available before anything else
python -m pip install --upgrade pip
python -m pip install setuptools==75.3.0 wheel==0.44.0 build

echo "[build.sh] Installing numpy compatible with faiss first (1.24.4)"
# Install a numpy version known to be compatible with faiss wheels before installing faiss
python -m pip install --no-cache-dir numpy==1.24.4 || echo "[build.sh] numpy install failed"

echo "[build.sh] Installing binary packages (faiss-cpu, torch, torchvision)"
# install binary packages first; continue on failure to show full logs
python -m pip install --no-cache-dir faiss-cpu==1.12.0 || echo "[build.sh] faiss-cpu install failed"
python -m pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 || echo "[build.sh] torch/vision install failed"

echo "[build.sh] Installing remaining requirements (using constraints.txt)"
if [ -f constraints.txt ]; then
	python -m pip install --no-cache-dir -r requirements.txt -c constraints.txt || echo "[build.sh] requirements install failed"
else
	python -m pip install --no-cache-dir -r requirements.txt || echo "[build.sh] requirements install failed (no constraints)"
fi

echo "[build.sh] Checking installed package compatibility with 'pip check'"
python -m pip check || echo "[build.sh] pip check reported issues (see above)"

echo "[build.sh] Verifying critical packages"
python - <<'PY'
import importlib
def check(pkg):
	try:
		m = importlib.import_module(pkg)
		v = getattr(m, '__version__', 'unknown')
		print(f"[build.sh] OK import {pkg} (version={v})")
	except Exception as e:
		print(f"[build.sh] MISSING {pkg}: {e}")

check('authlib')
check('faiss')
check('torch')
PY

echo "[build.sh] pip freeze (top 50 lines)"
python -m pip freeze | sed -n '1,50p' || true

echo "[build.sh] build script finished"








