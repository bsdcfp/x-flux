import json
import argparse
from op_utils import *
 
ENABLE_DEBUG=False
 
def find_end(s:str, sub_s:str, start_pos=0):
    return s.index(sub_s, start_pos) + len(sub_s)
 
def remove_address(line, item):
 
    start_pos = find_end(line, item)
    end_pos = find_end(line, ',', start_pos)
 
    return line[:start_pos - len(item)] + line[end_pos+1:]
 
def strip_conv_log(line):
    line = remove_address(line, 'handle: ')
    line = remove_address(line, 'x: ')
    line = remove_address(line, 'w: ')
    line = remove_address(line, 'workspace: ')
    line = line[0:find_end(line, 'y: ')-3]
    return line
 
def strip_sdpa_log(line):
    line = remove_address(line, 'handle: ')
    line = remove_address(line, 'q: ')
    line = remove_address(line, 'k: ')
    line = remove_address(line, 'v: ')
    line = remove_address(line, 'workspace: ')
    line = line[0:find_end(line, 'out: ')-5]
    return line
 
def init_call_id(l, call_id):
    assert l[call_id] is None
    l[call_id] = dict()
    l[call_id]['host'] = None
    l[call_id]['device'] = list()
 
def print_once(message):
    global printed
    if not printed:
        print(message)
        printed = True
 
# 初始化标志
printed = False
 
def parse_prof_op(trace_name):
    ignored_names = ['ProfilerStep#6', 'PyTorch Profiler (0)']
    cpu_ops = list()
 
    with open(trace_name) as f:
        trace = json.load(f)['traceEvents']
 
    for entry in trace:
        if 'cat' not in entry:
            continue
        if entry['cat'] == 'cpu_op':
            if entry['name'] in ignored_names or 'ProfilerStep' in entry['name'] or 'autograd::engine::evaluate_function:' in entry['name'] or 'Backward0' in entry['name']:
                continue
 
            new_op = CpuOp(name=entry['name'], start=entry['ts'], dur=entry['dur'])
 
            if entry['name'] == 'LaunchGraphExecMultiInstance':
                new_op.use_graph = True
            if 'External id' in entry['args']:
                new_op.add_call_id(entry['args']['External id'])
            if 'Concrete Inputs' in entry['args']:
                new_op.add_args(entry['args']['Concrete Inputs'])
            if 'Input Dims' in entry['args']:
                new_op.add_shape(entry['args']['Input Dims'])
            if 'Input type' in entry['args']:
                new_op.add_dtype(entry['args']['Input type'])
 
            for idx in range(len(cpu_ops)):
                op = cpu_ops[idx]
                if op.contains(new_op):
                    op.merge_call_id(new_op)
                    new_op = None
                    break
                elif new_op.contains(op):
                    new_op.merge_call_id(op)
                    cpu_ops.pop(idx)
                    cpu_ops.insert(idx, new_op)
                    new_op = None
                    break
                elif op.start > new_op.start:
                    cpu_ops.insert(idx, new_op)
                    new_op = None
                    break
 
            if new_op is not None:
                # assert idx == len(cpu_ops - 1) or len(cpu_ops) == 0
                cpu_ops.append(new_op)
                 
    idx = 0
    while idx < len(cpu_ops):
        if len(cpu_ops[idx].call_id) == 0:
            cpu_ops.pop(idx)
        else:
            idx += 1
             
    for entry in trace:
        if 'cat' not in entry:
            continue
        if entry['cat'] == 'kernel':
            if entry['name'] in ignored_names:
                continue
 
            if 'External id' not in entry['args']:
                print_once('\nWARNING: DeviceOp no callid, checkout .json correct or not!!\n')
                continue
            # repr = None
            # if '__repr__' in entry['args']:
            #     repr = entry['args']['__repr__']
            
            new_op = DeviceOp(name=entry['name'], 
                              start=entry['ts'], 
                              dur=entry['dur'], 
                              call_id=entry['args']['External id'], 
                              entry_args=entry['args'])
 
            for cpu_op in cpu_ops:
                if new_op.call_id in cpu_op.call_id:
                    cpu_op.add_device_time(new_op)
                    cpu_op.add_device_op(new_op)
                    new_op = None
                    break
 
            # print('new_op', entry['name'], entry['args']['call_id'])
            if new_op is not None:
                print_once('\nWARNING: new_op no callid, checkout .json correct or not!!\n')
                save_incorrect_strings(entry)
 
    idx = len(cpu_ops) - 1
    while idx >= 0:
        if cpu_ops[idx].device_start == None or cpu_ops[idx].device_end == None:
            #print('cpu_ops', cpu_ops[idx].name)
            unfind_device_id = "devcie unfind: " + cpu_ops[idx].name + " " + str(cpu_ops[idx].call_id)
            save_incorrect_strings(unfind_device_id)
            del cpu_ops[idx]
        idx -= 1
 
    return cpu_ops
def save_incorrect_strings(strings, filename='./trace_prof_dump.txt'):
    """将不正确的字符串保存到文件中"""
    with open(filename, 'a', encoding='utf-8') as file:
        json_data = json.dumps(strings)
        file.write(json_data)
 
def get_profiler_calculate_time(cpu_ops):
        op_time_dict = { }
        op_time_dict_times = { }
        cpu_ops = sorted(cpu_ops, key=lambda cuOp: cuOp.device_start, reverse=False)
        for idx in range(len(cpu_ops)):
            op = cpu_ops[idx]
            if idx != 0:
                bubble = cpu_ops[idx].device_start - cpu_ops[idx - 1].device_end
                if bubble < 0:
                    prev_node = cpu_ops[idx - 1].name + " " + str(cpu_ops[idx - 1].device_end) + " " + str(cpu_ops[idx-1].call_id)
                    cur_node = cpu_ops[idx].name + " " + str(cpu_ops[idx].device_end) + " " + str(cpu_ops[idx].call_id)
                    save_incorrect_strings(prev_node + '\n' + cur_node)
                else:
                    op_time_dict['bubble'] += bubble
 
            if op.name not in op_time_dict:
                op_time_dict[op.name] = op.get_device_time() - op.get_bubble_time()
                op_time_dict_times[op.name] = 1
            else:
                op_time_dict[op.name] += op.get_device_time() - op.get_bubble_time()
                op_time_dict_times[op.name] += 1
 
            if 'bubble' not in op_time_dict:
              op_time_dict['bubble'] = op.get_bubble_time()
              op_time_dict_times['bubble'] = 1
            else:
              op_time_dict['bubble'] += op.get_bubble_time()
              op_time_dict_times['bubble'] += 1
 
        total_time = 0
        bubble_time = 0
        op_time_dict = sorted(op_time_dict.items(), key=lambda x: x[1], reverse=True)
        print("{:<{width}} {:<{width}} {:<{width}} {:<{width}}".format("OpName", "DeviceActiveTime", "Calls", "Percentage\n", width=50))
        for name in op_time_dict:
            if name[0] == 'bubble':
                bubble_time = name[1] / 1000
 
            total_time += name[1] / 1000
 
        for name in op_time_dict:
            formatted_percentage = format(((name[1] / 1000) / total_time) * 100, '.2f')
            formatted_output = "{:<{width}} {:<{width}} {:<{width}} {:<{width}}".format(name[0], str(name[1] / 1000), str(op_time_dict_times[name[0]]), str(formatted_percentage) + "%", width=50)
            print(formatted_output)
 
        print('total_device_time: ', total_time, 'total_device_activate_time: ', total_time - bubble_time, 'total_bubble_time: ', bubble_time, '\n')
 
        return op_time_dict
def analyze_op_performance(cpu_ops, total_op, all_dlnn_dlblas_log, top_num = 3):
    #1. show top num ops
    analyze_top_num=top_num #show top 3 Slow op
 
    need_analyze_op_array = list()
    for op in total_op:
        if analyze_top_num != 0:
            if op[0] == "bubble":
                continue
            need_analyze_op_array.append(op[0])
            analyze_top_num -= 1
 
    print('Top '+str(top_num)+' ops', need_analyze_op_array)
 
    #2. calculate top ops all shape info
    prof_cu_list = list()
    prof_conv2d_list = list()
    prof_attn_list = list()
    prof_linear_list = list()
    prof_bmm_list = list()
    prof_matmul_list = list()
 
    for op in cpu_ops:
      if op.name in need_analyze_op_array:
        if op.name == "aten::conv2d":
            prof_conv2d_list = convert_CpuOp_to_Conv2dOp(op, prof_conv2d_list)
        elif op.name == "aten::scaled_dot_product_attention":
            prof_attn_list = convert_CpuOp_to_AttnOp(op, prof_attn_list)
        elif op.name == "aten::linear":
            prof_linear_list = convert_CpuOp_to_LinearOp(op, prof_linear_list)
        elif op.name == "aten::bmm":
            prof_bmm_list = convert_CpuOp_to_BmmOp(op, prof_bmm_list)
        elif op.name == "aten::matmul":
            prof_matmul_list = convert_CpuOp_to_MatmulOp(op, prof_matmul_list)
        else:
            prof_cu_list = convert_CpuOp_to_CuOp(op, prof_cu_list)
 
    if ENABLE_DEBUG:
        print('debug pytorch log:')
        debug_op_list(prof_cu_list)
        debug_op_list(prof_conv2d_list)
        debug_op_list(prof_attn_list)
        debug_op_list(prof_linear_list)
        debug_op_list(prof_bmm_list)
        debug_op_list(prof_matmul_list)
 
    new_prof_cu_list = sorted(prof_cu_list, key=lambda cuOp: cuOp.get_total_active_time(), reverse=True)
    new_prof_conv2d_list = sorted(prof_conv2d_list, key=lambda convOp: convOp.get_total_active_time(), reverse=True)
    new_prof_attn_list = sorted(prof_attn_list, key=lambda attnOp: attnOp.get_total_active_time(), reverse=True)
    new_prof_linear_list = sorted(prof_linear_list, key=lambda linearOp: linearOp.get_total_active_time(), reverse=True)
    new_prof_bmm_list = sorted(prof_bmm_list, key=lambda bmmOp: bmmOp.get_total_active_time(), reverse=True)
    new_prof_matmul_list = sorted(prof_matmul_list, key=lambda matmulOp: matmulOp.get_total_active_time(), reverse=True)
 
    for opname in need_analyze_op_array:
        print('opName:', opname)
        if opname == "aten::conv2d":
            debug_op_list(new_prof_conv2d_list)
        elif opname == "aten::scaled_dot_product_attention":
            debug_op_list(new_prof_attn_list)
        elif opname == "aten::linear":
            debug_op_list(new_prof_linear_list)
        elif opname == "aten::bmm":
            debug_op_list(new_prof_bmm_list)
        elif opname == "aten::matmul":
            debug_op_list(new_prof_matmul_list)
        else:
            debug_op_list(new_prof_cu_list, opname)
 
import threading
import queue
 
# 创建一个队列实例
prof_queue = queue.Queue()
log_queue = queue.Queue()
 
def worker_prof(prof):
    # 假设这是线程执行的计算
    all_prof_ops = parse_prof_op(prof)
    # 将结果放入队列
    prof_queue.put(all_prof_ops)
 
def main(args):
    # TODO need check file
    print('args', args)
 
    #解析profiler file 数据, cpu_op, and device_op
    if args.thread:
        thread_prof = threading.Thread(target=worker_prof, args=(args.prof,))
        thread_prof.start()
    else:
        all_prof_ops = parse_prof_op(args.prof)
     
    if args.thread:
        thread_prof.join()
        all_prof_ops = prof_queue.get()
 
    #统计各个op时间
    result = get_profiler_calculate_time(all_prof_ops)
 
    #获取详细前n个算子执行时间
    analyze_op_performance(all_prof_ops, result, None, top_num=args.top)  
 
if __name__ == "__main__":
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Pytorch Perfermance Tools!')
     
    # 添加参数
    parser.add_argument('--prof', type=str, default="./trace_unet_8.json", required=True, help='pytorch prof example: --prof ./trace_unet_8.json')
    parser.add_argument('--output', type=str, default="./pytorch_perf.log", help='output perf tools example: ./pytorch_perf.log')
    parser.add_argument('--top', type=int, default=3, help='need opt top N ops')
    parser.add_argument('--debug', type=bool, default=False, help='enable debug log')
    parser.add_argument('--thread', type=bool, default=True, help='enable muti threads')
 
    # 解析命令行参数
    args = parser.parse_args()
     
    # 调用main函数并传递解析后的参数
    main(args)