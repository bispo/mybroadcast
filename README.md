# MyBroadcast

## Objective
> Create a virtual webcam with effects:
> - ChromaKey
> - Change background by video

## Install

> this project was development in python 3.9

### Ubuntu

> modules:
> - **v4l2loopback-dkms:** This module allows you to create “virtual video devices”
> - **protobuf:** is an open source library developed by Google that allows to serialize or deserialize structured data


```bash
# install module of system
sudo apt update
sudo apt -y install v4l2loopback-dkms protobuf-compiler

# dependencies of project:
conda create -n mybroadcast python=3.9.12 poetry
poetry install

# run the mybroadcast:
./start.sh
# or
./start.sh <video_path>
```


## Development Environment

```bash
conda create -n mybroadcast python=3.9.12 poetry
poetry install
pre-commit install
```
