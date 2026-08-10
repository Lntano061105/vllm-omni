#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"

curl --fail --silent --show-error "http://${HOST}:${PORT}/health"

curl --fail --silent --show-error "http://${HOST}:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍你自己\"}],\"modalities\":[\"text\"],\"max_tokens\":128}"

curl --fail --silent --show-error "http://${HOST}:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"先打个招呼，再用一句话介绍 vLLM。\"}],\"modalities\":[\"text\",\"audio\"],\"chat_template_kwargs\":{\"enable_thinking\":false,\"use_tts_template\":true}}"
