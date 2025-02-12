#!/usr/bin/env bash
set -euo pipefail

# Change to the script's directory.
cd "$(dirname "$0")"

# Prompt for container build type.
echo "Select container build type:"
echo "1. CPU only"
echo "2. GPU (CUDA)"
echo "3. GPU (ROCm)"
read -p "Enter your choice (1, 2, or 3): " buildType

case "$buildType" in
  "1")
    prebuiltImage="dclipca/spongequant-cpu:latest"
    localTag="dclipca/spongequant-cpu:latest"
    containerName="spongequant-cpu"
    dockerfileName="Dockerfile.cpu"
    ;;
  "2")
    prebuiltImage="dclipca/spongequant-gpu-cuda:latest"
    localTag="dclipca/spongequant-gpu-cuda:latest"
    containerName="spongequant-gpu-cuda"
    dockerfileName="Dockerfile.gpu-cuda"
    ;;
  "3")
    prebuiltImage="dclipca/spongequant-gpu-rocm:latest"
    localTag="dclipca/spongequant-gpu-rocm:latest"
    containerName="spongequant-gpu-rocm"
    dockerfileName="Dockerfile.gpu-rocm"
    ;;
  *)
    echo "[ERROR] Invalid choice. Please run the script again."
    exit 1
    ;;
esac

# Set GPU option only for CUDA.
if [ "$buildType" == "2" ]; then
    gpuOption="--gpus=all"
else
    gpuOption=""
fi

# Display the menu for pulling or building the image.
echo ""
echo "Select an option:"
echo "1. Download prebuilt Docker image from Docker Hub"
echo "2. Build the Docker image locally"
read -p "Enter your choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    imageName="$prebuiltImage"
    echo "[INFO] Pulling image $imageName from Docker Hub..."
    sudo docker pull "$imageName"
elif [ "$choice" == "2" ]; then
    echo "[INFO] Building the Docker image locally using $dockerfileName..."
    sudo docker build -f "$dockerfileName" -t "$localTag" .
    imageName="$localTag"
else
    echo "[ERROR] Invalid choice. Please run the script again and enter either 1 or 2."
    exit 1
fi

# Functions to check for an existing container.
get_existing_container() {
    sudo docker ps -a --filter "name=$1" --format "{{.Names}}"
}

get_running_container() {
    sudo docker ps --filter "name=$1" --format "{{.Names}}"
}

existingContainer=$(get_existing_container "$containerName")

if [ -n "$existingContainer" ]; then
    # Container exists; check if it is running.
    runningContainer=$(get_running_container "$containerName")
    if [ -z "$runningContainer" ]; then
        echo "[INFO] Starting existing container $containerName..."
        sudo docker start "$containerName" >/dev/null
    fi
    echo "[INFO] Attaching to container $containerName..."
    sudo docker exec -it "$containerName" bash
else
    echo "[INFO] Running a new container named $containerName from image $imageName..."
    if [ -n "$gpuOption" ]; then
        sudo docker run $gpuOption -it --rm -p 7860:7860 --name "$containerName" \
            -v "$(pwd)/app/gguf:/app/gguf" \
            -v "$(pwd)/models:/app/models" \
            -v "$(pwd)/quantized_models:/app/quantized_models" \
            "$imageName"
    else
        sudo docker run -it --rm -p 7860:7860 --name "$containerName" \
            -v "$(pwd)/app/gguf:/app/gguf" \
            -v "$(pwd)/models:/app/models" \
            -v "$(pwd)/quantized_models:/app/quantized_models" \
            "$imageName"
    fi
fi

# Pause before exit (optional)
read -p "Press enter to exit..."