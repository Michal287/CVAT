# Ubuntu 20.04

# Clone cvat
~ git clone https://github.com/opencv/cvat
~ cd cvat

# Download nuctl
~ wget https://github.com/nuclio/nuclio/releases/download/1.8.14/nuctl-1.8.14-linux-amd64

# Give permissions (check version in documentation!)
~ sudo chmod +x nuctl-1.8.14-linux-amd64
~ sudo ln -sf $(pwd)/nuctl-1.8.14-linux-amd64 /usr/local/bin/nuctl

# Build docker
~ docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d --build

# Check is nuctl is ready
~ nuctl get functions

# Create user
~ docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

# Model installation (First time use detectron2 in documentation)
~ ./serverless/deploy_cpu.sh ./serverless/pytorch/facebookresearch/detectron2/mask-rcnn/