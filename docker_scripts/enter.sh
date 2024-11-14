xhost +local:docker  # Allow Docker to use the display
docker exec -it cloth-recon /bin/bash
xhost -local:docker
