#!/usr/bin/env bash
# Run a shell command on the RunPod Isaac Sim pod through the ssh.runpod.io proxy.
# The proxy insists on a PTY and rejects `ssh host cmd`, so commands are piped
# through stdin of an interactive shell and the terminal noise is stripped.
#
#   bash scripts/pod_ssh.sh 'nvidia-smi -L'
#   bash scripts/pod_ssh.sh 'cd /isaac-sim/world2work && ls output/isaac'
set -uo pipefail
POD="${POD:-rmw4ikd9aespj0-644114fe@ssh.runpod.io}"
KEY="${KEY:-$HOME/.ssh/id_ed25519_runpod}"
MARK="__POD_CMD_${RANDOM}__"
printf '%s\necho %s_START; %s; echo %s_END; exit\n' "stty -echo 2>/dev/null" "$MARK" "$1" "$MARK" \
  | ssh -tt -i "$KEY" -o ConnectTimeout=20 -o LogLevel=ERROR "$POD" 2>&1 \
  | tr -d '\r' \
  | awk -v s="${MARK}_START" -v e="${MARK}_END" 'index($0,s)&&!index($0,"echo "){f=1;next} index($0,e)&&!index($0,"echo "){f=0} f'
