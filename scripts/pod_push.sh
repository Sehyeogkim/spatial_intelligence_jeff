#!/usr/bin/env bash
# Upload a local file to the RunPod pod through the PTY-only ssh proxy.
# The PTY line discipline caps input lines at 4095 bytes, so the payload is
# streamed as 76-column base64 and decoded on the pod.
#
#   bash scripts/pod_push.sh local/file.py /isaac-sim/world2work/scripts/file.py
set -euo pipefail
POD="${POD:-rmw4ikd9aespj0-644114fe@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/id_ed25519_runpod}"
LOCAL="$1"; REMOTE="$2"
{
  printf 'stty -echo 2>/dev/null; mkdir -p "%s"; base64 -d > "%s" <<"__B64_EOF__"\n' "$(dirname "$REMOTE")" "$REMOTE"
  base64 -b 76 -i "$LOCAL"
  printf '\n__B64_EOF__\necho PUSH_DONE $(stat -c %%s "%s") "%s"; exit\n' "$REMOTE" "$REMOTE"
} | ssh -tt -i "$KEY" -o ConnectTimeout=20 -o LogLevel=ERROR "$POD" 2>&1 | tr -d '\r' | grep -E "^PUSH_DONE" || { echo "push failed: $LOCAL" >&2; exit 1; }
echo "local bytes: $(stat -f %z "$LOCAL")  $LOCAL"
