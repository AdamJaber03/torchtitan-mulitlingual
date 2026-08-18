#!/bin/bash
# Finds the SSH agent socket that has a step-ca certificate loaded.
# Usage: export SSH_AUTH_SOCK=$(bash find_ssh_agent.sh)
# Each socket is probed under a 5s alarm: a stale/wedged agent socket makes
# `ssh-add -l` hang indefinitely, which would otherwise block the whole search
# (and any self-heal bsub/bkill that resolves the socket this way). perl's
# alarm() timer survives exec, so SIGALRM terminates a hung ssh-add after 5s.
find /Users/lc/.ssh/agent -name "*.agent*" | while read sock; do
  if SSH_AUTH_SOCK="$sock" perl -e 'alarm 5; exec @ARGV or exit 1' ssh-add -l 2>/dev/null | grep -q "CERT"; then
    echo "$sock"
    break
  fi
done
