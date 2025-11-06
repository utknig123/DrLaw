#!/usr/bin/env bash
# exit on error
set -o errexit

echo "[build.sh] Running build script in $(pwd)"
echo "[build.sh] Python: $(python -V 2>&1)"

# ensure setuptools and wheel are available before anything else
python -m pip install --upgrade pip
python -m pip install setuptools==75.3.0 wheel==0.44.0 build

echo "[build.sh] Cleaning any existing numpy/faiss/torch installs"
python -m pip uninstall -y numpy faiss faiss-cpu torch torchvision || true

echo "[build.sh] Force-installing numpy compatible with faiss first (1.24.4)"
# Force reinstall numpy so the C extensions match the expected ABI
python -m pip install --no-cache-dir --force-reinstall numpy==1.24.4 || echo "[build.sh] numpy install failed"

echo "[build.sh] Installing faiss-cpu (prefer wheel) and torch packages"
# Try to install faiss from a wheel first to avoid building from source or triggering incompatible numpy builds
python -m pip install --upgrade pip
echo "[build.sh] Attempting wheel install for faiss-cpu"
if ! python -m pip install --no-cache-dir --only-binary=:all: faiss-cpu==1.12.0; then
	echo "[build.sh] faiss-cpu wheel install failed; attempting normal install to get better error output"
	if ! python -m pip install --no-cache-dir faiss-cpu==1.12.0; then
		echo "[build.sh] faiss-cpu install ultimately failed. Showing diagnostics:" 
		python -m pip show faiss-cpu || true
		python -m pip index versions faiss-cpu || true
		python -m pip freeze | sed -n '1,200p' || true
		echo "[build.sh] Exiting build due to faiss-cpu install failure"
		exit 1
	fi
fi

echo "[build.sh] Installing torch and torchvision (wheel preferred)"
if ! python -m pip install --no-cache-dir --only-binary=:all: torch==2.0.1 torchvision==0.15.2; then
	echo "[build.sh] torch wheel install failed; attempting normal install"
	python -m pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 || echo "[build.sh] torch install failed"
fi

# Verify faiss can be imported now; if not, fail early with diagnostics
echo "[build.sh] Verifying faiss import"
python - <<'PY'
import sys,traceback
try:
		import faiss
		print('[build.sh] OK import faiss', getattr(faiss,'__version__','unknown'))
except Exception as e:
		print('[build.sh] FAILED to import faiss:')
		traceback.print_exc()
		import subprocess
		print('\n[build.sh] pip show faiss-cpu:')
		subprocess.run([sys.executable,'-m','pip','show','faiss-cpu'])
		print('\n[build.sh] pip freeze:')
		subprocess.run([sys.executable,'-m','pip','freeze'])
		sys.exit(2)
PY

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








