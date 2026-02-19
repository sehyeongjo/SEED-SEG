FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    ca-certificates \
    bash \
    vim \
    git \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

# Set root password.
RUN echo "root:root" | chpasswd

RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's@session\s\+required\s\+pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Data mount point.
RUN mkdir -p /data

EXPOSE 7900 22

CMD ["/app/start.sh"]
