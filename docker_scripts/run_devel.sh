#!/bin/bash

./stop.sh

docker run -it -d --privileged --net=host \
--name cloth-recon \
--env="QT_X11_NO_MITSHM=1" \
--env="DISPLAY" \
-v $PWD/../workspace:/app/workspace \
cloth-recon-dev:latest
