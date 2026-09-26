#!/usr/bin/env bash
# run python inside the pinned env
cd "$(dirname "$0")"
MM=$(command -v micromamba || echo ./bin/micromamba)
export MAMBA_ROOT_PREFIX="$PWD/.mamba"
exec $MM run -n pit python "$@"
