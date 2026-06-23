#!/bin/bash
# Finds the SSH agent socket that has a step-ca certificate loaded.
# Usage: export SSH_AUTH_SOCK=$(bash find_ssh_agent.sh)
find /Users/lc/.ssh/agent -name "*.agent*" | while read sock; do
  if SSH_AUTH_SOCK="$sock" ssh-add -l 2>/dev/null | grep -q "CERT"; then
    echo "$sock"
    break
  fi
done
