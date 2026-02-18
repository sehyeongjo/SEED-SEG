




```





port 7007 : for web app
port 7022 : for ssh

docker run -d --name seg-seed \
  -p 7007:7007 \
  -p 7022:22 \
  --mount type=bind,source="{SEED CT DATA ROOT PATH}",target=/data \
  seg-seed
```