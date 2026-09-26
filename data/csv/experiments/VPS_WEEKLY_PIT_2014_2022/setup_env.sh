#!/usr/bin/env bash
# Creates a Python env IDENTICAL to the Mac that produced the reference results.
# HistGradientBoosting output can shift across sklearn/numpy versions -- pin them.
# Uses micromamba (single static binary, no root, works on any Linux x86_64).
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v micromamba >/dev/null 2>&1 && [ ! -x ./bin/micromamba ]; then
  echo "installing micromamba locally (./bin) ..."
  case "$(uname -m)" in
    aarch64|arm64) PLAT=linux-aarch64 ;;   # AWS Graviton etc.
    *)             PLAT=linux-64 ;;
  esac
  echo "platform: $PLAT"
  curl -Ls "https://micro.mamba.pm/api/micromamba/$PLAT/latest" | tar -xvj bin/micromamba
fi
MM=$(command -v micromamba || echo ./bin/micromamba)
export MAMBA_ROOT_PREFIX="$PWD/.mamba"
$MM create -y -n pit -c conda-forge \
  python=3.8.18 pandas=2.0.3 numpy=1.24.4 scikit-learn=1.3.2 pyarrow=17.0.0
echo
echo "env ready. versions:"
$MM run -n pit python -c "import sys,pandas,numpy,sklearn,pyarrow;print(sys.version.split()[0],'pandas',pandas.__version__,'numpy',numpy.__version__,'sklearn',sklearn.__version__,'pyarrow',pyarrow.__version__)"
echo "expected: 3.8.18 pandas 2.0.3 numpy 1.24.4 sklearn 1.3.2 pyarrow 17.0.0"
