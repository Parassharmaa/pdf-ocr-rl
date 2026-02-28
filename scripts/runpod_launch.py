"""Launch a RunPod GPU pod for GRPO training.

Usage:
    python scripts/runpod_launch.py --action create
    python scripts/runpod_launch.py --action status
    python scripts/runpod_launch.py --action terminate --pod-id POD_ID
"""

import argparse
import json
import os
import sys
import time

import requests

RUNPOD_API = "https://api.runpod.io/graphql"


def get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("RUNPOD_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        break
    if not key:
        print("Error: RUNPOD_API_KEY not found in environment or .env file")
        sys.exit(1)
    return key


def graphql_request(query: str, variables: dict | None = None) -> dict:
    api_key = get_api_key()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    resp = requests.post(RUNPOD_API, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    if "errors" in result:
        print(f"GraphQL errors: {json.dumps(result['errors'], indent=2)}")
        sys.exit(1)

    return result.get("data", {})


def get_gpu_types():
    """List available GPU types and their pricing."""
    query = """
    query {
        gpuTypes {
            id
            displayName
            memoryInGb
            communityPrice
            securePrice
        }
    }
    """
    data = graphql_request(query)
    gpus = data.get("gpuTypes", [])

    # Filter to relevant GPUs
    relevant = [g for g in gpus if g.get("communityPrice") and g["memoryInGb"] >= 16]
    relevant.sort(key=lambda g: g["communityPrice"])

    print("\nAvailable GPUs (sorted by price):")
    print(f"{'GPU':<30} {'VRAM':>6} {'Community $/hr':>15} {'Secure $/hr':>12}")
    print("-" * 65)
    for g in relevant[:15]:
        secure = g.get("securePrice", "N/A")
        if secure:
            secure = f"${secure:.2f}"
        print(f"{g['displayName']:<30} {g['memoryInGb']:>4}GB  ${g['communityPrice']:>12.2f}   {secure:>10}")

    return relevant


def create_pod(gpu_type_id: str = "NVIDIA GeForce RTX 3090", volume_size: int = 30):
    """Create a RunPod pod for training."""
    # Training setup script
    setup_script = """#!/bin/bash
set -e

echo "=== Setting up PDF-OCR-RL training environment ==="

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install system deps for PDF rendering
apt-get update && apt-get install -y --no-install-recommends \\
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \\
    libffi-dev libcairo2 fonts-noto-cjk fonts-noto \\
    poppler-utils git && \\
    rm -rf /var/lib/apt/lists/*

# Clone and setup project
cd /workspace
if [ ! -d "pdf-ocr-rl" ]; then
    echo "Project not found. Please upload or clone it."
fi

echo "=== Environment ready ==="
echo "To start training:"
echo "  cd /workspace/pdf-ocr-rl"
echo "  uv sync --extra train --extra data"
echo "  uv run python scripts/create_dataset.py"
echo "  uv run python scripts/train_grpo.py --config configs/grpo_train.yaml"
"""

    query = """
    mutation createPod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
            imageName
            machineId
            machine {
                gpuDisplayName
            }
        }
    }
    """

    variables = {
        "input": {
            "name": "pdf-ocr-rl-training",
            "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "gpuTypeId": gpu_type_id,
            "gpuCount": 1,
            "volumeInGb": volume_size,
            "containerDiskInGb": 20,
            "minVcpuCount": 4,
            "minMemoryInGb": 16,
            "dockerArgs": "",
            "env": [],
            "templateId": None,
            "startJupyter": True,
            "startSsh": True,
            "cloudType": "COMMUNITY",
        }
    }

    print(f"\nCreating pod with {gpu_type_id}...")
    print(f"Volume: {volume_size}GB")
    data = graphql_request(query, variables)
    pod = data.get("podFindAndDeployOnDemand", {})

    if pod:
        print(f"\nPod created successfully!")
        print(f"  ID: {pod['id']}")
        print(f"  Name: {pod['name']}")
        print(f"  Status: {pod['desiredStatus']}")
        print(f"  Image: {pod['imageName']}")
        print(f"\nWait for pod to be RUNNING, then SSH in and set up the project.")
        return pod
    else:
        print("Failed to create pod")
        return None


def get_pods():
    """List all active pods."""
    query = """
    query {
        myself {
            pods {
                id
                name
                desiredStatus
                runtime {
                    uptimeInSeconds
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
                machine {
                    gpuDisplayName
                }
                costPerHr
            }
            spending {
                totalSpend
            }
        }
    }
    """
    data = graphql_request(query)
    myself = data.get("myself", {})
    pods = myself.get("pods", [])
    spending = myself.get("spending", {})

    print(f"\nTotal spending: ${spending.get('totalSpend', 0):.2f}")
    print(f"\nActive pods ({len(pods)}):")

    for pod in pods:
        runtime = pod.get("runtime", {})
        uptime_hrs = runtime.get("uptimeInSeconds", 0) / 3600 if runtime else 0
        cost_hr = pod.get("costPerHr", 0)
        gpu_name = pod.get("machine", {}).get("gpuDisplayName", "Unknown")

        print(f"\n  Pod: {pod['name']} ({pod['id']})")
        print(f"    GPU: {gpu_name}")
        print(f"    Status: {pod['desiredStatus']}")
        print(f"    Uptime: {uptime_hrs:.1f} hours")
        print(f"    Cost: ${cost_hr:.2f}/hr (est. total: ${uptime_hrs * cost_hr:.2f})")

        if runtime and runtime.get("ports"):
            for port in runtime["ports"]:
                if port.get("isIpPublic"):
                    print(f"    Port {port['privatePort']}: {port['ip']}:{port['publicPort']}")

    return pods


def terminate_pod(pod_id: str):
    """Terminate a pod."""
    query = """
    mutation terminatePod($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    variables = {"input": {"podId": pod_id}}

    print(f"\nTerminating pod {pod_id}...")
    graphql_request(query, variables)
    print("Pod terminated.")


def stop_pod(pod_id: str):
    """Stop a pod (keeps volume, stops billing for GPU)."""
    query = """
    mutation stopPod($input: PodStopInput!) {
        podStop(input: $input) {
            id
            desiredStatus
        }
    }
    """
    variables = {"input": {"podId": pod_id}}

    print(f"\nStopping pod {pod_id}...")
    data = graphql_request(query, variables)
    print(f"Pod stopped. Status: {data.get('podStop', {}).get('desiredStatus')}")


def main():
    parser = argparse.ArgumentParser(description="Manage RunPod GPU pods")
    parser.add_argument("--action", choices=["create", "status", "gpus", "stop", "terminate"], required=True)
    parser.add_argument("--pod-id", type=str, help="Pod ID for stop/terminate")
    parser.add_argument("--gpu", type=str, default="NVIDIA GeForce RTX 3090", help="GPU type")
    parser.add_argument("--volume", type=int, default=30, help="Volume size in GB")
    args = parser.parse_args()

    if args.action == "create":
        create_pod(args.gpu, args.volume)
    elif args.action == "status":
        get_pods()
    elif args.action == "gpus":
        get_gpu_types()
    elif args.action == "stop":
        if not args.pod_id:
            print("Error: --pod-id required for stop")
            sys.exit(1)
        stop_pod(args.pod_id)
    elif args.action == "terminate":
        if not args.pod_id:
            print("Error: --pod-id required for terminate")
            sys.exit(1)
        terminate_pod(args.pod_id)


if __name__ == "__main__":
    main()
