<div align="center">
  <img width="181" height="721" alt="Image" src="https://github.com/user-attachments/assets/1306eb02-7c90-4c57-88be-86f4ace323e9" />
</div>

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

4-1. For SSH
```
ssh -P 7922 root@localhost / pw: root
```


4-2. For using command line (RECOMMEND)
```
IMPORTANT!!
1.
# Edit this list directly.
TRAYS=(1 10 20 30) or TRAYS=({1..30})

2. permission
chmod +x build_template.sh
chmod +x box_segmentation.sh


# Build template
./build_template.sh

# Run box segmentation
./box_segmentation.sh
```

5. SAM
```
cd sam
python sam.py --tray_num {TRAY_NUM:int}
```

6. U-Net
```
cd unet

# Train
python train.py

# Inference
python infer.py --tray_num {TRAY_NUM:int} --checkpoint_name {CHECKPOINT.ckpt}
```




Info.

Dockerfile
```
# Set root password
# Default: root/root (with sudo permission)
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
    │   ├── 1/
    │   │   ├── merged_010922.tif
    │   │   ├── merged_010923.tif
    │   │   └── ...
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
