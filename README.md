
1. Clone this repository:
```
git clone https://github.com/sehyeongjo/SEED-SEG.git
cd SEED-SEG
```

2. Docker Image Build:
```
docker build -t seg-seed .
```

If a credential error occurs regarding python:3.11-slim, run the following command:
```
docker pull python:3.11-slim
```


3. Docker Container Run:
```
docker run -d --name seg-seed2 \
  --gpus all \
  -p 7900:7900 \
  -p 7922:22 \
  --mount type=bind,source="{SEED CT DATA ROOT PATH}",target=/data \
  seg-seed
```

4. Run the Web Application in a browser
```
http://localhost:7900
```



Info.

Dockerfile
```
# Set user / pw
# Default is root/root (sudo permission)
RUN echo "root:root" | chpasswd
```

Docker Run
```
port 7900 : for web app
port 7922 : for ssh
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
    │   ├── 1/
    │   │   ├── merged_002534/
    │   │   │   ├── cells/
    │   │   │   │   ├── A_1.png
    │   │   │   │   ├── A_2.png
    │   │   │   │   └── ...
    │   │   │   └── debug
    │   │   ├── merged_002535
    │   │   └── ...
    │   ├── 2
    │   └── ...
    └── mask_data/
        ├── 1/
        │   ├── mask/
        │   │   ├── merged_011040_output
        │   │   ├── merged_011041_output
        │   │   └── ...
        │   └── original/
        │       ├── merged_011040_output (Same mask's )
        │       ├── merged_011041_output
        │       └── ...
        ├── 2
        └── ...
```