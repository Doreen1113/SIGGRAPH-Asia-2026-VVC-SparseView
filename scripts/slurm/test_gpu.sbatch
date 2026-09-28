#!/bin/bash
#SBATCH --job-name=vvc_gputest
#SBATCH --account=GOV114009
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/work/doreen071/vvc/jobs/gputest_%j.log
module load cuda/12.6
PY=/home/doreen071/.conda/envs/vvc312/bin/python
export TORCH_CUDA_ARCH_LIST=9.0
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY - <<'PY'
import torch, time
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16); torch.cuda.synchronize(); t=time.time()
for _ in range(20): y = x @ x
torch.cuda.synchronize(); print(f"fp16 matmul TFLOPS ~ {20*2*8192**3/(time.time()-t)/1e12:.0f}")
try:
    import gsplat; print("gsplat", gsplat.__version__)
except Exception as e: print("gsplat import:", e)
PY
echo GPUTEST_DONE
