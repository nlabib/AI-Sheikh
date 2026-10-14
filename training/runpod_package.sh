#!/usr/bin/env bash
# On the pod, after runpod_train.sh: merge the adapter into the base model, make the chat template non-thinking
# by default, and convert to an 8-bit GGUF (Q8_0) for LM Studio. Merging an 8B model needs ~16 GB of RAM and
# the base model, which the pod already has, so it is cheaper to do here than at home.
#
#   setsid bash runpod_package.sh ai-sheikh-8b-v2 > package.log 2>&1 < /dev/null &
#   then fetch /workspace/<name>-Q8_0.gguf (and check its sha256)
set -e
NAME=${1:?output folder name under outputs/}
cd /workspace/ai-sheikh
export HF_HOME=/workspace/hf-cache PYTHONUNBUFFERED=1
pip install --break-system-packages -q sentencepiece protobuf gguf
[ -d /workspace/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
python -m training.merge_lora --adapter "outputs/$NAME/adapter" --out "outputs/$NAME-merged"
python -m training.no_think_template "outputs/$NAME-merged"
python /workspace/llama.cpp/convert_hf_to_gguf.py "outputs/$NAME-merged" --outfile "/workspace/$NAME-Q8_0.gguf" --outtype q8_0
sha256sum "/workspace/$NAME-Q8_0.gguf"
ls -la "/workspace/$NAME-Q8_0.gguf"
echo PACKAGE_DONE
