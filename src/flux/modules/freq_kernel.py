import triton
import triton.language as tl
import torch
import math
@triton.jit
def freq_kernel(
    output_ptr,  # Pointer to output tensor
    half,        # Half of the dimension
    max_period,  # Maximum period
    num_elements,# Number of elements to process
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process
):
    # Get program ID
    pid = tl.program_id(axis=0)
    # Calculate start index for this program
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create mask for valid indices
    mask = offsets < num_elements

    # Calculate indices for valid elements
    idx = tl.where(mask, offsets, 0)
    
    # Main computation
    log_max_period = tl.log(max_period)
    x = -log_max_period * (idx.to(tl.float32) / half)
    result = tl.exp(x)
    
    # Write results back to memory
    output = tl.where(mask, result, 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def compute_frequencies_triton(half: int, max_period: float = 10000.0, device='cuda'):
    """
    Compute frequencies using Triton kernel
    """
    # Allocate output tensor
    output = torch.empty(half, device=device, dtype=torch.float32)
    
    # Calculate grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(half, BLOCK_SIZE),)
    
    # Launch kernel
    freq_kernel[grid](
        output,
        half,
        max_period,
        half,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Example usage and benchmark
def benchmark_freq_calculation(half=128, max_period=10000.0, num_runs=1000):
    # Original PyTorch implementation
    def torch_freq():
        return torch.exp(
            -math.log(max_period) * 
            torch.arange(start=0, end=half, dtype=torch.float32, device='cuda') / 
            half
        )
    
    # Triton implementation
    def triton_freq():
        return compute_frequencies_triton(half, max_period)
    
    # Warmup
    torch_result = torch_freq()
    triton_result = triton_freq()
    
    # Verify results match
    assert torch.allclose(torch_result, triton_result, rtol=1e-3, atol=1e-3)
    
    # Benchmark
    import time
    
    # Time PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        torch_freq()
    torch.cuda.synchronize()
    torch_time = (time.perf_counter() - start) / num_runs
    
    # Time Triton
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        triton_freq()
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_runs
    
    print(f"PyTorch time: {torch_time*1000:.3f}ms")
    print(f"Triton time:  {triton_time*1000:.3f}ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")

# Run benchmark
if __name__ == "__main__":
    benchmark_freq_calculation()