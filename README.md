
Connect SSH

1. After connecting, run the following command in that folder:
git clone https://github.com/sehyeongjo/SEED-SEG.git


2. Copy the directory using:
cp -r SEED-SEG/app/ /app



Dockerfile
```
# Set user / pw
# Default is root/root (sudo permission)
RUN echo "root:root" | chpasswd
```

Docker Image Build
```
docker build -t seg-seed .
```

Docker Run
```
port 7007 : for web app
port 7022 : for ssh

docker run -d --name seg-seed2 \
  -p 7007:7007 \
  -p 7022:22 \
  --mount type=bind,source="{SEED CT DATA ROOT PATH}",target=/data \
  seg-seed
```