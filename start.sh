#!/usr/bin/env bash
set -e

# Generate SSH host keys on first run.
ssh-keygen -A

# Start SSH daemon.
/usr/sbin/sshd

# Start FastAPI with reload enabled.
exec uvicorn app.main:app --host 0.0.0.0 --port 7900 --reload --reload-dir /app
