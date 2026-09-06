#!/usr/bin/env bash
# Render the matched held-out pair (trained vs untrained) inside the Marble café
# on the RunPod Isaac Sim pod, then pull both MP4s back to assets/.
#
# Usage:  bash scripts/pod_render_holdout.sh
# Needs:  SSH access to the pod (the same key Codex uses). The proxy does not
#         support scp, so files are streamed through ssh.
set -euo pipefail

POD="${POD:-rmw4ikd9aespj0-644114fe@ssh.runpod.io}"
REMOTE_ROOT="${REMOTE_ROOT:-/isaac-sim/world2work}"
PY="${PY:-/isaac-sim/python.sh}"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

push() {  # push <local> <remote>
  ssh "$POD" "mkdir -p \"$(dirname "$2")\" && cat > \"$2\"" < "$1"
  echo "pushed $1 -> $2"
}
pull() {  # pull <remote> <local>
  mkdir -p "$(dirname "$2")"
  ssh "$POD" "cat \"$1\"" > "$2"
  echo "pulled $1 -> $2 ($(du -h "$2" | cut -f1))"
}

# 1. sync the pieces that changed locally
push "$LOCAL_ROOT/scripts/isaac_nav_episode.py"                          "$REMOTE_ROOT/scripts/isaac_nav_episode.py"
push "$LOCAL_ROOT/config/corgi_cafe_task.json"                            "$REMOTE_ROOT/config/corgi_cafe_task.json"
push "$LOCAL_ROOT/artifacts/corgi-cafe/nav/route_holdout_trained.json"    "$REMOTE_ROOT/artifacts/corgi-cafe/nav/route_holdout_trained.json"
push "$LOCAL_ROOT/artifacts/corgi-cafe/nav/route_holdout_untrained.json"  "$REMOTE_ROOT/artifacts/corgi-cafe/nav/route_holdout_untrained.json"
push "$LOCAL_ROOT/artifacts/corgi-cafe/nav/grid_meta.json"                "$REMOTE_ROOT/artifacts/corgi-cafe/nav/grid_meta.json"

# 2. render both episodes with the same camera and start pose
run_episode() {  # run_episode <label> <extra args...>
  local label="$1"; shift
  ssh "$POD" "cd $REMOTE_ROOT && $PY scripts/isaac_nav_episode.py \
      --route artifacts/corgi-cafe/nav/route_holdout_${label}.json \
      --output output/isaac/holdout_${label}.jsonl \
      --video-output output/isaac/holdout_${label}.mp4 \
      --disable-marble-visual $* 2>&1 | grep -E 'VIDEO_OK|EPISODE|outcome|Error|Traceback|IMPORT' || true"
}
run_episode trained
run_episode untrained --timeout-s 40

# 3. bring the results home
pull "$REMOTE_ROOT/output/isaac/holdout_trained.mp4"     "$LOCAL_ROOT/assets/isaac_holdout_trained.mp4"
pull "$REMOTE_ROOT/output/isaac/holdout_untrained.mp4"   "$LOCAL_ROOT/assets/isaac_holdout_untrained.mp4"
pull "$REMOTE_ROOT/output/isaac/holdout_trained.jsonl"   "$LOCAL_ROOT/output/isaac/holdout_trained.jsonl"
pull "$REMOTE_ROOT/output/isaac/holdout_untrained.jsonl" "$LOCAL_ROOT/output/isaac/holdout_untrained.jsonl"
echo "DONE — check assets/isaac_holdout_trained.mp4 first frame: robot wheels should sit on the café floor"
