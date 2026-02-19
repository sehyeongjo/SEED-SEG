#!/usr/bin/env bash
set -e

# SSH host keys (처음 실행 시 생성)
ssh-keygen -A

# SSH daemon 시작
/usr/sbin/sshd

# FastAPI (reload 필수)
exec uvicorn app.main:app --host 0.0.0.0 --port 7900 --reload --reload-dir /app