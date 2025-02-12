# Display container build type menu
Write-Host "Select container build type:"
Write-Host "1. CPU only"
Write-Host "2. GPU (CUDA)"
Write-Host "3. GPU (ROCm)"
$buildType = Read-Host "Enter your choice (1, 2, or 3)"

switch ($buildType) {
    "1" {
        $prebuiltImage = "dclipca/spongequant-cpu:latest"
        $localTag      = "dclipca/spongequant-cpu:latest"
        $containerName = "spongequant-cpu"
        $dockerfileName = "Dockerfile.cpu"
    }
    "2" {
        $prebuiltImage = "dclipca/spongequant-gpu-cuda:latest"
        $localTag      = "dclipca/spongequant-gpu-cuda:latest"
        $containerName = "spongequant-gpu-cuda"
        $dockerfileName = "Dockerfile.gpu-cuda"
    }
    "3" {
        $prebuiltImage = "dclipca/spongequant-gpu-rocm:latest"
        $localTag      = "dclipca/spongequant-gpu-rocm:latest"
        $containerName = "spongequant-gpu-rocm"
        $dockerfileName = "Dockerfile.gpu-rocm"
    }
    default {
        Write-Host "[ERROR] Invalid choice. Please run the script again."
        exit
    }
}

# Set GPU option only for CUDA mode.
if ($buildType -eq "2") {
    $gpuOption = "--gpus=all"
}
else {
    $gpuOption = ""
}

# Ask user whether to pull a prebuilt image or build locally.
Write-Host "`nSelect an option:"
Write-Host "1. Download prebuilt Docker image from Docker Hub"
Write-Host "2. Build the Docker image locally"
$choice = Read-Host "Enter your choice (1 or 2)"

if ($choice -eq "1") {
    $imageName = $prebuiltImage
    Write-Host "[INFO] Pulling image $imageName from Docker Hub..."
    docker pull $imageName
}
elseif ($choice -eq "2") {
    Write-Host "[INFO] Building the Docker image locally using $dockerfileName..."
    docker build -f $dockerfileName -t $localTag .
    $imageName = $localTag
}
else {
    Write-Host "[ERROR] Invalid choice. Please run the script again and enter either 1 or 2."
    Pause
    exit
}

# Check if a container with the fixed name already exists.
function Get-ExistingContainer($name) {
    docker ps -a --filter "name=$name" --format "{{.Names}}"
}

# Function to check if a container is running.
function Get-RunningContainer($name) {
    docker ps --filter "name=$name" --format "{{.Names}}"
}

$existingContainer = Get-ExistingContainer $containerName

if ($existingContainer) {
    # Container exists; check if it is running.
    $runningContainer = Get-RunningContainer $containerName
    if (-not $runningContainer) {
        Write-Host "[INFO] Starting existing container $containerName..."
        docker start $containerName | Out-Null
    }
    Write-Host "[INFO] Attaching to container $containerName..."
    docker exec -it $containerName bash
}
else {
    Write-Host "[INFO] Running a new container named $containerName from image $imageName..."
    if ($gpuOption -ne "") {
        docker run $gpuOption -it --rm -p 7860:7860 --name $containerName `
            -v "$PWD\app\gguf:/app/gguf" `
            -v "$PWD\models:/app/models" `
            -v "$PWD\quantized_models:/app/quantized_models" `
            $imageName
    }
    else {
        docker run -it --rm -p 7860:7860 --name $containerName `
            -v "$PWD\app\gguf:/app/gguf" `
            -v "$PWD\models:/app/models" `
            -v "$PWD\quantized_models:/app/quantized_models" `
            $imageName
    }
}

Pause