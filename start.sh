sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=1
poetry run python mybroadcast/main.py $@
