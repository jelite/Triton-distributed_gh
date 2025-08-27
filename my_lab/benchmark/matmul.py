import argparse
import csv
import os

import torch
import torch.nn as nn
import time
from layer import LinearTorchB16, LinearTritonBF16

def warmup(x: torch.Tensor, layer: torch.nn.Module) -> None:
    for _ in range(10):
        y = layer(x)
    torch.cuda.synchronize()

def benchmark(x: torch.Tensor, layer: torch.nn.Module, iterations: int=100) -> tuple[torch.Tensor, float]:

    # ----- 벤치마크 -----      
    warmup(x, layer)

    time.start = time.time()
    for _ in range(iterations):
        y_dist = layer(x)  # [B, out_features] (gather_output=True)
    torch.cuda.synchronize()
    time.end = time.time()
    
    return y_dist, (time.end - time.start)/iterations

    
def main() -> None:
    torch.manual_seed(1234)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--in_features', type=int, default=4096, help='Input feature dimension')
    parser.add_argument('--out_features', type=int, default=32768, help='Output feature dimension')
    parser.add_argument('--model_name', type=str, default="", help='Model name')
    args = parser.parse_args()
    
    B = args.batch
    in_features = args.in_features
    out_features = args.out_features
    model_name = args.model_name
    
    x = torch.randn(B, in_features, device="cuda", dtype=torch.bfloat16)

    weight = nn.Parameter(torch.empty(in_features, out_features, device="cuda", dtype=torch.bfloat16))
    bias   = nn.Parameter(torch.empty(out_features, device="cuda", dtype=torch.bfloat16))
    nn.init.kaiming_uniform_(weight, a=5**0.5)
    bound = (1.0 / in_features) ** 0.5
    nn.init.uniform_(bias, -bound, bound)
        
        
    # Torch baseline
    layer_torch = LinearTorchB16(weight, bias).cuda()
    y_torch, t_torch = benchmark(x, layer_torch)

    # Triton kernel
    layer_triton = LinearTritonBF16(weight, bias).cuda()
    y_triton, t_triton = benchmark(x, layer_triton)

    # 결과 비교 (float으로 변환해서 오차 확인)
    ok = torch.allclose(y_torch, y_triton, rtol=1e-2, atol=1e-2)
    
    csv_path = "benchmark_results.csv"
    file_exists = os.path.exists(csv_path)

    # 모델 이름이 이미 존재하는지 확인
    model_exists = False
    if file_exists:
        with open(csv_path, mode='r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == model_name and row[1] == str(B):
                    model_exists = True
                    break

    # 존재하지 않으면 append
    if not model_exists:
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # 새 파일일 경우 헤더 추가
            if not file_exists:
                writer.writerow(["model_name", "in_features", "out_features", "batch", "t_torch", "t_triton", "ok"])
            writer.writerow([model_name,in_features, out_features,B,f"{t_torch:.4f}", f"{t_triton:.4f}", ok])

    

if __name__ == "__main__":
    main()
