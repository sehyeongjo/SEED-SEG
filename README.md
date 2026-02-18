
Connect SSH

1. After connecting, run the following command in that folder:
```
git clone https://github.com/sehyeongjo/SEED-SEG.git
```

2. Copy the directory using:
```
cp -r SEED-SEG/app/ /app
```


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



Data(SEED CT) File Tree Structure
```
.
└── Data(Root)/
    ├── tray/
    │   ├── 1
    │   ├── 2
    │   └── ...
    ├── trayseg_output/
    │   └── TBD
    └── mask_data/
        ├── 1/
        │   ├── mask/
        │   │   ├── merged_011040_output
        │   │   ├── merged_011041_output
        │   │   └── ...
        │   └── original/
        │       ├── merged_011040_output (Same mask)
        │       ├── merged_011041_output
        │       └── ...
        ├── 2
        └── ...
```