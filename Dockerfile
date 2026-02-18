FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# SSH + 기본 유틸
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    ca-certificates \
    bash \
    vim \
    git \
 && rm -rf /var/lib/apt/lists/*

# SSH 설정
RUN mkdir -p /var/run/sshd

# 🔥 root 비밀번호 설정
RUN echo "root:root" | chpasswd

# 🔥 root 로그인 허용
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's@session\s\+required\s\+pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# App
COPY app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# data mount 지점
RUN mkdir -p /data

EXPOSE 7007 22

CMD ["/app/start.sh"]
