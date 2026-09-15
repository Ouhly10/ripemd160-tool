#!/bin/bash
# loop.sh — يشغّل ripemd160_tool في حلقة لا نهائية، مع self-test في بداية كل دورة.
set -u
TARGET="${1:-f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8}"
BATCH="${2:-16777216}"

while true; do
    /app/ripemd160_tool "$TARGET" "$BATCH" 1000
    echo "[loop.sh] دورة انتهت — إعادة تشغيل خلال 5 ثوانٍ..."
    sleep 5
done
