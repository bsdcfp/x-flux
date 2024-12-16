import torch
import os
from datetime import datetime

def memory_profiler(original_filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Start recording memory history
            print("Starting memory recording...")
            torch.cuda.memory._record_memory_history(max_entries=10_000_000)

            result = func(*args, **kwargs)
            try:
                gpu_id = os.getenv('CUDA_VISIBLE_DEVICES', '0')

                # Split filename and extension
                filename, file_extension = os.path.splitext(original_filename)

                # Insert GPU ID into the filename
                new_filename = f"{filename}_gpu{gpu_id}{file_extension}"

                # Dump memory snapshot
                torch.cuda.memory._dump_snapshot(new_filename)

                # Save memory summary to a file
                new_summary_filename = f"{filename}_mem_summary_gpu{gpu_id}.txt"
                with open(new_summary_filename, 'w') as summary_file:
                    summary = torch.cuda.memory_summary()
                    summary_file.write(summary)
                print(f"CUDA Memory Summary saved to {new_summary_filename}")
            except Exception as e:
                print(f"Failed to capture memory snapshot: {e}")

            # Stop recording memory history
            print("Stopping memory recording...")
            torch.cuda.memory._record_memory_history(enabled=False)
            return result
        return wrapper

    return decorator


def trace_handler(prof, trace_file):
    """
    
    """
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    # 导出chrome trace文件
    prof.export_chrome_trace(trace_file)
 
def profile(enabled=True, 
                  trace_file="./trace_prof_infer_xxx.json", 
                  log_dir="./log",
                  proc_type="infer",
                  is_use_tensorboard=False):
    """
    Decorator for profiling a function using torch.profiler.
 
    Args:
        enabled (bool or str): Whether to enable profiling. Can be a boolean or a string
                               that evaluates to a boolean ('true', '1', 'yes').
        trace_file (str): Path to the file where the profiling trace will be saved.
        proc_type (str): Process type, can be "infer" or "train".
        use_tensorboard (bool or str): Whether to use tensorboard for visualization. Can be a boolean or a string
                                        that evaluates to a boolean ('true', '1', 'yes').
 
    Note:
        The `enabled` argument can be used to conditionally enable profiling. If it is a string, it will be evaluated as a boolean.
 
    Returns:
        function: Decorated function.
    """
 
    def decorator(func):
        def wrapper(*args, **kwargs):
            enable_profiling = enabled
            if isinstance(enable_profiling, str):
                # Convert string representation of boolean to actual boolean
                enable_profiling = enable_profiling.lower() in ("true", "1", "yes")
            use_tensorboard = is_use_tensorboard
            if isinstance(use_tensorboard, str):
                # Convert string representation of boolean to actual boolean
                use_tensorboard = use_tensorboard.lower() in ("true", "1", "yes")

            t_warmup_latency = 0
            t_avg_latency = 0
            start_index = trace_file.find("trace_prof_") + len("trace_prof_")
            end_index = trace_file.rfind(".json")

            case_name = trace_file[start_index:end_index]
            # profile
            if enable_profiling:
                # warmup
                start_time = datetime.now()
                func(*args, **kwargs)
                end_time = datetime.now()
                t_warmup_latency = (end_time - start_time).total_seconds()
                print(
                    f"<case {case_name} warmup time cost: {t_warmup_latency * 1000:5.3f} ms>"
                )
                if proc_type == "infer":
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir) if use_tensorboard else trace_handler
                    ) as prof:
                        start_time = datetime.now()
                        result = func(*args, **kwargs)
                        end_time = datetime.now()
                        t_avg_latency = (end_time - start_time).total_seconds()
                    print(
                        f"<case {case_name} profile time cost: {t_avg_latency * 1000:5.3f} ms>"
                    )
                elif proc_type == "train":
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(
                            wait=1,
                            warmup=1,
                            active=1,
                            repeat=1),
                        record_shapes=True,
                        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir) if use_tensorboard else trace_handler
                    ) as prof:
                        start_time = datetime.now()
                        result = func(*args, **kwargs)
                        end_time = datetime.now()
                        t_avg_latency = (end_time - start_time).total_seconds()
                        prof.step()
            else:
                start_time = datetime.now()
                result = func(*args, **kwargs)
                end_time = datetime.now()

                t_avg_latency = (end_time - start_time).total_seconds()

                print(
                    f"<customer bugs case {case_name} avg time cost: {t_avg_latency * 1000:5.3f} ms>"
                )
            return result, t_avg_latency
 
        return wrapper
 
    return decorator
