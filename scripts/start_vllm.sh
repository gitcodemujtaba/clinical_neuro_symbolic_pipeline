#!/usr/bin/env bash
# scripts/start_vllm.sh -- start both Stage 3 ensemble members.
#
# WHY THIS FILE EXISTS. The launch flags below are not preferences; each one is
# a constraint discovered by hitting the failure it prevents. Leaving them in
# shell history means rediscovering them, so they are recorded here.
#
#   --quantization awq   Both models must be resident on one Tesla T4 (15.36GB).
#                        At fp16 they are ~14.5GB and ~16GB respectively, so
#                        neither pair nor even the larger one alone would fit.
#                        AWQ's GEMM kernels need compute capability >= 7.5; the
#                        T4 is exactly 7.5, so this is the last card that works.
#
#   --dtype half         The T4 cannot do bfloat16 (vLLM's
#                        check_if_supports_dtype requires >= 8.0), and fp8 does
#                        not exist on Turing. fp16 is the only option.
#
#   --max-model-len 4096 KV cache is what is left after weights. 8192 is the
#                        architectural ceiling (OpenBioLLM fine-tunes Llama-3,
#                        not 3.1) but does not fit alongside two sets of
#                        weights. See llm_client.CONTEXT_WINDOW_TOKENS.
#
#   --enforce-eager      Skips CUDA graph capture and torch.compile. Saves
#                        startup time and VRAM, costs throughput that a
#                        few-hundred-entities-per-note workload does not need.
#
#   utilisation 0.40 / 0.55   DISJOINT shares of the card, summing below 1.0.
#                        The fraction is of TOTAL card memory, and vLLM refuses
#                        to start unless that much is currently FREE:
#
#                            ValueError: Free memory on device cuda:0
#                            (8.58/14.56 GiB) on startup is less than desired
#                            GPU memory utilization (0.96, 13.98 GiB)
#
#                        So each server's fraction must fit in what its
#                        predecessors left behind. Measured on this card:
#                        BioMistral at 0.40 occupies 5.98 GiB and leaves 8.58
#                        GiB free, i.e. a ceiling of 0.58 for the second server.
#                        0.55 takes ~8.0GB of which ~5.7GB is weights, leaving
#                        ~2.3GB of KV cache -- about 4 concurrent 4096-token
#                        sequences, comfortably more than Stage 3's serial
#                        one-record-at-a-time workload needs.
#
#   sequential startup   The failure that actually cost a debugging session.
#                        Started 10 seconds apart, the second server begins
#                        allocating while the first is still profiling; the
#                        first then computes its KV budget against memory that
#                        is disappearing underneath it and dies with
#                        "No available memory for the cache blocks", after
#                        which the second succeeds against a freed card. The
#                        result is a half-up ensemble that looks like a model
#                        problem. wait_ready() polls /v1/models rather than
#                        sleeping a fixed interval, because load time varies
#                        with page cache and any fixed sleep is a race.
#
# NOT MEDGEMMA. The proposal named MedGemma 4B. It is Gemma-3-based and cannot
# run on a T4 under any dtype -- fp16 is refused by vLLM for numerical
# instability, bf16 by compute capability, fp32 by VRAM. BioMistral replaces it
# and preserves what mattered: a base-model family distinct from OpenBioLLM's
# Llama-3, so the two votes fail independently. See src/llm_client.py.
#
# Usage:  bash scripts/start_vllm.sh [stop]

set -u

VENV="${VLLM_VENV:-/home/ec2-user/vllm-env}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BIOMISTRAL_MODEL="${BIOMISTRAL_MODEL:-BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM}"
OPENBIOLLM_MODEL="${OPENBIOLLM_MODEL:-bartowski/OpenBioLLM-Llama3-8B-AWQ}"
LOG_DIR="${LOG_DIR:-/tmp}"

if [ "${1:-start}" = "stop" ]; then
    pkill -f "vllm.entrypoints.openai.api_server" && echo "stopped" || echo "nothing running"
    exit 0
fi

# The vLLM venv is a SEPARATE interpreter from the pipeline's. vLLM requires
# Python >= 3.10 (it uses PEP 604 unions at import time); the pipeline is
# pinned to 3.9 because scispaCy 0.5.3 holds spacy < 3.8. The two never share
# a process -- they communicate over HTTP.
if [ ! -x "$VENV/bin/python" ]; then
    echo "No vLLM venv at $VENV. Create it with a 3.10+ interpreter:"
    echo "  python3.11 -m venv $VENV && $VENV/bin/pip install vllm"
    exit 1
fi

is_serving () {
    curl -s -f "http://localhost:$1/v1/models" > /dev/null 2>&1
}

# Polls until the port answers /v1/models, the process is gone, or we time out.
# Returns non-zero on failure so the caller can stop rather than start a second
# server against a card in an unknown state.
wait_ready () {
    local port="$1" log="$2" pid="$3" waited=0 limit="${VLLM_START_TIMEOUT:-600}"
    while [ "$waited" -lt "$limit" ]; do
        if is_serving "$port"; then
            echo "  ready on :$port after ${waited}s"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  FAILED on :$port -- process exited. Root cause:"
            grep "EngineCore" "$log" | grep -E "Error|error:" | tail -3 | sed 's/^/    /'
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "  TIMEOUT on :$port after ${limit}s; see $log"
    return 1
}

start_one () {
    local model="$1" port="$2" util="$3" log="$4" template="${5:-}"
    if is_serving "$port"; then
        echo "  port $port already serving; skipping"
        return 0
    fi
    local extra=()
    if [ -n "$template" ]; then
        [ -f "$template" ] || { echo "  missing chat template: $template"; return 1; }
        extra+=(--chat-template "$template")
        echo "  using chat template $template"
    fi
    echo "  starting $model on :$port (util $util) -> $log"
    nohup "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "$model" --port "$port" \
        --quantization awq --dtype half \
        --max-model-len 4096 --gpu-memory-utilization "$util" \
        --enforce-eager "${extra[@]}" > "$log" 2>&1 &
    wait_ready "$port" "$log" "$!"
}

echo "Starting Stage 3 ensemble (sequentially -- see header)..."
start_one "$BIOMISTRAL_MODEL" 8000 "${BIOMISTRAL_UTIL:-0.40}" "$LOG_DIR/biomistral.log" || {
    echo "BioMistral failed to start; not starting OpenBioLLM. A half-up"
    echo "ensemble is worse than none: Stage 3 would route every record to"
    echo "HITL on LLMUnavailable and the run would look like a model problem."
    exit 1
}
# OpenBioLLM needs an explicit chat template. The bartowski AWQ repack ships no
# `chat_template` in tokenizer_config.json, and transformers >= 4.44 refuses to
# fall back to a default one:
#
#   As of transformers v4.44, default chat template is no longer allowed, so
#   you must provide a chat template if the tokenizer does not define one.
#
# The template is vendored in config/ rather than pulled from the unquantised
# upstream repo so the exact prompt format applied to clinical text is versioned
# with the code and can be shown in the write-up. It is the stock Llama-3
# Instruct format, which is what OpenBioLLM was fine-tuned with -- getting this
# wrong would not error, it would just degrade the model silently.
start_one "$OPENBIOLLM_MODEL" 8001 "${OPENBIOLLM_UTIL:-0.55}" \
    "$LOG_DIR/openbiollm.log" "$PROJECT_DIR/config/llama3_chat_template.jinja" || exit 1

echo
echo "Both members serving."
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>/dev/null
echo "Logs: $LOG_DIR/biomistral.log $LOG_DIR/openbiollm.log"
