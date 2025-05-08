  git config --global user.email "mertas2002@gmail.com"
  git config --global user.name "MERT T."



sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev


apt install openssl \
    build-essential \
    curl \
    gcc \
    libbz2-dev \
    libev-dev \
    libffi-dev \
    libgdbm-dev \
    liblzma-dev \
    libncurses-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    make \
    tk-dev \
    wget \
    git \
    zlib1g-dev
	
pip install ultralytics --break-system-packages
pip install "git+https://github.com/Tencent/ncnn.git"  --break-system-packages

sudo apt install -y python3-picamera2
sudo apt install -y python3-picamera2 --no-install-recommends

pip install picamera2

https://docs.ultralytics.com/ru/guides/raspberry-pi/#how-can-i-set-up-a-raspberry-pi-camera-module-to-work-with-ultralytics-yolo11

rpicam-vid -n -t 0 --inline --listen -o tcp://127.0.0.1:8888
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("tcp://127.0.0.1:8888")


sudo chown root:root /etc/systemd/system/screen_run.service
sudo chmod 777 /etc/systemd/system/screen_run.service
sudo chown root:root /home/mert/source/FollowingRobotCar/run.sh

sudo systemctl daemon-reload
sudo systemctl enable screen_run
sudo systemctl status screen_run
sudo systemctl start screen_run
sudo systemctl stop screen_run
sudo systemctl restart screen_run

which screen