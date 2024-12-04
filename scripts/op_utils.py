class DeviceOp:
    def __init__(self, name, start, dur, call_id, entry_args) -> None:
        self.name = name
        self.start = start
        self.dur = dur
        self.end = start+dur
        self.call_id = call_id
        self.entry_args = entry_args
 
class CpuOp:
    def __init__(self, name, start, dur) -> None:
        self.name = name
        self.start = start
        self.dur = dur
        self.end = start+dur
        self.call_id = list()
        self.device_start = None
        self.device_end = None
        self.device_ops = list()
        self.shapes = list()
        self.dtypes = list()
        self.args = list()
        self.use_graph = False
        self.graph = GraphOp()
        self.last_name = ""
        self.last_shapes = list()
        self.last_dtypes = list()
        self.last_args = list()
        self.call_id = list()
 
    def contains(self, other):
        return self.start < other.start and self.end > other.end
 
    def add_shape(self, shape):
        self.shapes=shape
 
    def add_dtype(self, dtype):
        self.dtypes=dtype
 
    def add_args(self, arg):
        self.args=arg
 
    def add_call_id(self, call_id):
        self.call_id.append(call_id)
 
    def merge_call_id(self, other):
        if other.use_graph == True:
            for id in other.call_id:
                self.graph.graph_call_id_deivces_op[id] = []
 
        self.call_id += other.call_id
 
        if other.name == "aten::mm" or other.name == "aten::bmm" or other.name == "aten::addmm":
            self.last_name = other.name
            self.last_shapes = other.shapes
            self.last_dtypes = other.dtypes
            self.last_args = other.args
            self.call_id = other.call_id
 
    def add_device_time(self, device_op):
        if self.device_start is None:
            self.device_start = device_op.start
            self.device_end = device_op.end
        else:
            self.device_start = min(device_op.start, self.device_start)
            self.device_end = max(device_op.end, self.device_end)
 
    def is_same_shape_ops(self, cpuop):
        if self.shapes == cpuop.shapes and self.dtypes == cpuop.dtypes and self.args == cpuop.args:
            return true
 
        return false
 
    def get_shape_info(self):
        return ' Dims: ' + str(self.shapes) + ' Dtypes: '+ str(self.dtypes) + ' Args: ' + str(self.args)
     
    def get_last_shape_info(self):
        return 'LastName: ' + self.last_name +  ' Dims: ' + str(self.last_shapes) + ' Dtypes: '+ str(self.last_dtypes) + ' Args: ' + str(self.last_args)
     
    def get_device_time(self):
        return self.device_end - self.device_start
     
    def add_device_op(self, device_op: DeviceOp):
        found = False
        for idx in range(len(self.device_ops)):
            if self.device_ops[idx].start > device_op.start:
                self.device_ops.insert(idx, device_op)
                found = True
                break
 
        if not found:
            self.device_ops.append(device_op)
 
    def get_bubble_time(self):
        cur_end = 0
        bubble = 0
        for idx in range(len(self.device_ops)):
            if idx == 0:
                cur_end = self.device_ops[idx].end
            if self.device_ops[idx].start > cur_end:
                bubble += self.device_ops[idx].start - cur_end
                cur_end = self.device_ops[idx].end
            elif self.device_ops[idx].end > cur_end:
                cur_end = self.device_ops[idx].end
        return bubble
 
    def get_device_reduce_bubble_time(self):
        return self.get_device_time() - self.get_bubble_time()
 
NUM_TU = 4
GLOBAL_DISPATCH = False
 
class GraphOp:
    def __init__(self) -> None:
        self.graph_call_id_deivces_op = {}
        self.graph_time = {}
        self.graph_tu_time = {}
        self.graph_cu_time = {}
        self.graph_dma_time = {}
        self.graph_patch_buffer_time = {}
 
    def add_graph_op(self, op):
        if op.name == "graph":
            if op.call_id not in self.graph_time:
                self.graph_time[op.call_id] = op.dur
            else:
                self.graph_time[op.call_id] += op.dur
        else:
            self.graph_call_id_deivces_op[op.call_id].append(op)
    def get_all_graph_time(self):
        for key in self.graph_call_id_deivces_op:
            self.graph_tu_time[key] = 0
            self.graph_cu_time[key] = 0
            self.graph_dma_time[key] = 0
 
        for key in self.graph_call_id_deivces_op:
            self.graph_tu_time[key] = self.get_tu_time(self.graph_call_id_deivces_op[key])
            self.graph_cu_time[key] =self.get_cu_time(self.graph_call_id_deivces_op[key])
            self.graph_dma_time[key] = self.get_dma_time(self.graph_call_id_deivces_op[key])
            self.graph_patch_buffer_time[key] = self.get_patch_buffer_time(self.graph_call_id_deivces_op[key])
 
        return self.graph_tu_time, self.graph_cu_time, self.graph_dma_time, self.graph_patch_buffer_time
     
    def get_patch_buffer_time(self, device_ops):
        patch_buffer_time = 0
        for device_op in device_ops:
            assert isinstance(device_op, DeviceOp)
 
            if '__repr__' not in device_op.entry_args:
                continue
             
            elems = device_op.entry_args['__repr__'].split(':')
            if 'patch_buffer' in elems[0]:
                patch_buffer_time += device_op.dur
        patch_buffer_time
        return patch_buffer_time
 
    def get_tu_time(self, device_ops):
        tu_time = 0
        for device_op in device_ops:
            assert isinstance(device_op, DeviceOp)
            if '__repr__' not in device_op.entry_args:
                continue
             
            elems = device_op.entry_args['__repr__'].split(':')
            if 'tu' in elems[0] and 'tu' in elems[2]:
                tu_time += device_op.dur
        tu_time /= NUM_TU
        return tu_time
     
    def get_cu_time(self, device_ops):
        cu_time = 0
        # transpose_time = 0
        # set_zero_time = 0
        # conv_weight_time = 0
        for device_op in device_ops:
            assert isinstance(device_op, DeviceOp)
            if '__repr__' not in device_op.entry_args:
                continue
             
            elems = device_op.entry_args['__repr__'].split(':')
            if 'cu' in elems[0] and 'cu' in elems[2]:
                # if 'WriteOp' in device_op.name:
                #     transpose_time += device_op.dur
                # elif 'setzero' in device_op.name:
                #     set_zero_time += device_op.dur
                # else:
                #     assert 'conv_weight_conversion' in device_op.name
                #     conv_weight_time += device_op.dur
                cu_time += device_op.dur
        if not GLOBAL_DISPATCH:
            cu_time /= NUM_TU
            # transpose_time /= NUM_TU
            # set_zero_time /= NUM_TU
            # conv_weight_time /= NUM_TU
        return cu_time  # , transpose_time, set_zero_time, conv_weight_time
     
    def get_dma_time(self, device_ops):
        dma_time = 0
        for device_op in device_ops:
            assert isinstance(device_op, DeviceOp)
            if '__repr__' not in device_op.entry_args:
                continue
             
            elems = device_op.entry_args['__repr__'].split(':')
            if 'dma' in elems[0] and 'dma' in elems[2]:
                dma_time += device_op.dur
        dma_time /= NUM_TU
        return dma_time
     
    def debug_graph_info(self):
        self.get_all_graph_time()
        for call_id in self.graph_time:
            print(f'    Graph call_id: {call_id} graph time: {self.graph_time[call_id]} us tu_time: {self.graph_tu_time[call_id]} us cu_time: {self.graph_cu_time[call_id]} us dma_time: {self.graph_dma_time[call_id]} us patch_buffer: {self.graph_patch_buffer_time[call_id]} us ')
        print('\n')
 
class BaseOp:
    def __init__(self, name="", time = -1, bubble =-1, num_calls = 0, log = "") -> None:
        self.name = name
        self.num_calls = num_calls
        self.time = ()
        self.bubble = ()
        self.log_str = list()
 
        self.time += (time,)
        self.bubble += (bubble,)
        self.log_str.append(log)
        self.call_id = list()
        self.graph = None
        self.dtype = None
        self.last_layout_info = None
 
    def get_active_time(self):
        if len(self.time) == 0:
            return [-1]
 
        if len(self.time) == 1 and self.time[0] == -1:
           return [-1]
 
        return [self.time[idx] - self.bubble[idx] for idx in range(self.num_calls)]
     
    def get_op_time(self):
        return self.time
 
    def get_bubble_time(self):
        return self.bubble
     
    def get_total_time(self):
        return sum(self.time)
     
    def get_total_active_time(self):
        return sum(self.get_active_time())
     
    def merge(self, other):
        self.num_calls += other.num_calls
        self.time +=other.time
        self.bubble += other.bubble
        self.call_id += other.call_id
 
    def add_log(self, log):
        self.log_str.append(log)
 
    def set_call_id(self, call_id, graph, last_layout_info = None):
        self.call_id = call_id
        self.graph = graph
        self.last_layout_info = last_layout_info
 
class CuOp(BaseOp):
    def __init__(self, name, time, bubble, num_calls, log) -> None:
        super(CuOp, self).__init__(name, time, bubble, num_calls, log)
  
    def add_extension_log(self, log):
        self.log_str.append(log)
 
    def is_same(self, other):
        if not isinstance(other, CuOp):
            return False
        return self.log_str == other.log_str and \
               self.name == other.name
 
    def debug_op_info(self):
        print(f'    name     : {self.name}\n'
              f'    calls    : {self.num_calls} \n'
              f'    op time  (us)      {self.get_op_time()}\n'
              f'    bubble time (us)   {self.get_bubble_time()}\n'
              f'    active time (us)  {self.get_active_time()}\n'
              f'    total a time (us) {self.get_total_active_time()}\n'
              f'    call_id : {self.call_id}\n'
              f'    log      : {self.log_str}\n')
 
class ConvOp(BaseOp):
    def __init__(self, name, x_shape, w_shape, dtype, conv_padding, conv_strides, conv_dilations, time, bubble, num_calls, log) -> None:
        super().__init__(name, time, bubble, num_calls, log)
        self.x_shape = x_shape
        self.w_shape = w_shape
        self.conv_padding = conv_padding
        self.conv_strides = conv_strides
        self.conv_dilations = conv_dilations
        self.dtype = dtype
 
    def is_same(self, other):
        if not isinstance(other, ConvOp):
            return False
 
        return self.x_shape == other.x_shape and \
               self.w_shape == other.w_shape and \
               self.conv_padding == other.conv_padding and \
               self.conv_strides == other.conv_strides and \
               self.conv_dilations == other.conv_dilations and \
               self.name == other.name and \
               self.dtype == other.dtype
     
    def is_graph(self, str):
        if 'cudnn' not in str:
            return False
         
        start_pos = find_end(str, 'algo: ')
        end_pos = find_end(str, ',', start_pos)
        algo = str[start_pos: end_pos - 1]
        #print('algoXXXXXXX',algo, type(algo), len(algo))
 
        if 'cudnnConvolutionForward' in str and algo == "1" or \
        'cudnnConvolutionBackwardData' in str and algo == "2" or \
        'cudnnConvolutionBackwardFilter' in str and algo == "3" :
            return True
         
        else:
            return False
         
    def debug_op_info(self):
        print(f'    name  : {self.name}\n'
              f'    x     : {self.x_shape}\n'
              f'    w     : {self.w_shape}\n'
              f'    dtype     : {self.dtype}\n'
              f'    padding : {self.conv_padding}\n'
              f'    stride : {self.conv_strides}\n'
              f'    dilations : {self.conv_dilations}\n'
              f'    calls  : {self.num_calls} \n'
              f'    op time  (us)      {self.get_op_time()}\n'
              f'    bubble time (us)   {self.get_bubble_time()}\n'
              f'    active time (us)  {self.get_active_time()}\n'
              f'    total a time (us) {self.get_total_active_time()}\n'
              f'    call ids : {self.call_id}\n')
         
        if self.graph:
            self.graph.debug_graph_info()
 
        for index in range(len(self.log_str)):
            if self.is_graph(self.log_str[index]):
                print(f'    log{index}:  (Graph)', self.log_str[index])
            else:
                print(f'    log{index}:  ', self.log_str[index])
        print('\n')
 
class LinearOp(BaseOp):
    def __init__(self, name, m, n, k, beta, time, bubble, num_calls, mm_log) -> None:
        super().__init__(name, time, bubble, num_calls, mm_log)
        self.m = m
        self.n = n
        self.k = k
        self.beta=beta
 
    def add_extension_log(self, log):
        self.log_str.append(log)
 
    def is_same(self, other):
        if not isinstance(other, LinearOp):
            return False
 
        return self.m == other.m and \
               self.n == other.n and \
               self.k == other.k and \
               self.beta == other.beta 
     
    def debug_op_info(self):
        print(f'    name  : {self.name}\n'
              f'    m     : {self.m}\n'
              f'    n     : {self.n}\n'
              f'    k     : {self.k}\n'
              f'    beta     : {self.beta}\n'
              f'    calls  : {self.num_calls} \n'
              f'    device op time (us) {self.get_op_time()}\n'
              f'    bubble time (us)  {self.get_bubble_time()}\n'
              f'    active time (us)  {self.get_active_time()}\n'
              f'    total a time (us)  {self.get_total_active_time()}\n'
              f'    call ids : {self.call_id}\n'
              f'    last_log: {self.last_layout_info}\n')
         
        if self.graph:
            self.graph.debug_graph_info()
 
        for index in range(len(self.log_str)):
            print(f'    log{index}:  ', self.log_str[index])
 
        print('\n')
 
class BMMOp(LinearOp):
    def __init__(self, name, batchcount, m, n, k, transpose_m, transpose_n, transpose_k, beta, time, bubble, num_calls, mm_log) -> None:
        super().__init__(name, m, n, k, beta, time, bubble, num_calls, mm_log)
        self.batchcount = batchcount
        self.transpose_m = transpose_m
        self.transpose_n = transpose_n
        self.transpose_k = transpose_k
        self.transpose = False
     
    def is_same(self, other):
        if not isinstance(other, BMMOp):
            return False
         
        if self.batchcount == other.batchcount and \
               self.m == other.m and \
               self.n == other.n and \
               self.k == other.k and \
               self.beta == other.beta :
 
            return True
        else:
            #print('self.batchcount', self.batchcount, self.transpose_m, self.transpose_n, self.transpose_k, self.beta)
            #print('other.batchcount', other.batchcount, other.transpose_m, other.transpose_n, other.transpose_k, other.beta)
            if self.batchcount == other.batchcount and \
               self.transpose_m == other.transpose_m and \
               self.transpose_n == other.transpose_n and \
               self.transpose_k == other.transpose_k and \
               self.beta == other.beta:
                self.transpose = True
                return True
            else:
                return False
               
    def debug_op_info(self):
        print(f'    name  : {self.name}\n'
              f'    batchcount {self.batchcount}\n'
              f'    m     : {self.m}\n'
              f'    n     : {self.n}\n'
              f'    k     : {self.k}\n'
              f'    t_m     : {self.transpose_m}\n'
              f'    t_n     : {self.transpose_n}\n'
              f'    t_k     : {self.transpose_k}\n'
              f'    beta     : {self.beta}\n'
              f'    transpose : {self.transpose}\n'
              f'    calls  : {self.num_calls} \n'
              f'    device op time (us) {self.get_op_time()}\n'
              f'    bubble time (us)  {self.get_bubble_time()}\n'
              f'    active time (us)  {self.get_active_time()}\n'
              f'    total a time (us)  {self.get_total_active_time()}\n'
              f'    call ids : {self.call_id}\n'
              f'    last_log: {self.last_layout_info}\n')
         
        if self.graph:
            self.graph.debug_graph_info()
 
        for index in range(len(self.log_str)):
            print(f'    log{index}:  ', self.log_str[index])
 
        print('\n')
 
class MatmulOp():
    def __init__(self, op) -> None:
        self.op = op
        self.name = 'aten::matmul'
 
    def is_same(self, other):
        if not isinstance(other, MatmulOp) and not isinstance(other, BMMOp) and not isinstance(other, LinearOp):
            #print('other is not MatmulOp or BMMOp')
            return False
         
        return self.op.is_same(other)
    def get_active_time(self):
       return self.op.get_active_time()
     
    def get_op_time(self):
        return self.op.get_op_time()
 
    def get_bubble_time(self):
        return self.op.get_bubble_time()
     
    def get_total_time(self):
        return self.op.get_total_time()
     
    def get_total_active_time(self):
        return self.op.get_total_active_time()
    def debug_op_info(self):
       #print(f'    First name  : "aten::matmul')
        self.op.debug_op_info()
    def add_log(self, log):
        self.op.add_log(log)
 
class AttnOp(BaseOp):
    def __init__(self, name, q_shape, k_shape, v_shape, time, bubble, num_calls, log) -> None:
        super().__init__(name, time, bubble, num_calls, log)
        self.q_shape = q_shape
        self.k_shape = k_shape
        self.v_shape = v_shape
  
    def is_same(self, other):
        if not isinstance(other, AttnOp):
            return False
             
        return self.q_shape == other.q_shape and \
               self.k_shape == other.k_shape and \
               self.v_shape == other.v_shape
 
    def debug_op_info(self):
        print(f'    name  : {self.name}\n'
              f'    q     : {self.q_shape}\n'
              f'    k     : {self.k_shape}\n'
              f'    v     : {self.v_shape}\n'
              f'    calls  : {self.num_calls} \n'
              f'    device op time (us) {self.get_op_time()}\n'
              f'    bubble time (us)   {self.get_bubble_time()}\n'
              f'    active time (us)  {self.get_active_time()}\n'
              f'    total a time (us) {self.get_total_active_time()}\n'
              f'    call ids : {self.call_id}\n')
        for index in range(len(self.log_str)):
                print(f'    log{index}:  ', self.log_str[index])
 
        print('\n')
 
def debug_op_list(op_list, name=""):
    for op in op_list:
        if name == "" or name == op.name:
            op.debug_op_info()
 
def array_to_str(array):
    return '[' + ','.join(str(i) for i in array) + ']'
 
 
dtype_to_datatype = {
    'c10::Half':'2',
    'float':'0',
    'c10::BFloat16':'9',
}
 
def find_end(s:str, sub_s:str, start_pos=0):
    return s.index(sub_s, start_pos) + len(sub_s)
 
def convert_CpuOp_to_Conv2dOp(cpuOp, conv_list):
    #print('cpuOp.name', cpuOp.name, cpuOp.shapes)
    assert len(cpuOp.shapes) > 0, f'Conv2d cant find shape info, check profiler args Record_shape==True?'
    assert cpuOp.dtypes[0].replace(" ", "") in dtype_to_datatype, f'{cpuOp.dtypes[0].replace(" ", "")} not in dtype_to_datatype'
 
    x_shape = array_to_str(cpuOp.shapes[0])
    w_shape = array_to_str(cpuOp.shapes[1])
    x_dtype = dtype_to_datatype[cpuOp.dtypes[0].replace(" ", "")]
    #print('cpuOp.args', cpuOp.args)
    stride = cpuOp.args[3].replace(" ", "")
    padding = cpuOp.args[4].replace(" ", "")
    dilations = cpuOp.args[5].replace(" ", "")
    convOp = ConvOp(cpuOp.name, x_shape, w_shape, x_dtype, padding, stride, dilations, cpuOp.get_device_time(), cpuOp.get_bubble_time(), 1, cpuOp.get_shape_info())
    convOp.set_call_id(cpuOp.call_id, cpuOp.graph)
    for key in conv_list:
        if key.is_same(convOp):
            key.merge(convOp)
            return conv_list
 
    conv_list.append(convOp)
    return conv_list
 
def convert_CpuOp_to_Conv2dBackwardOp(cpuOp, conv_list):
    #print('cpuOp.name', cpuOp.name, cpuOp.shapes)
    assert len(cpuOp.shapes) > 0, f'Conv2d Back cant find shape info, check profiler args Record_shape==True?'
    assert cpuOp.dtypes[0].replace(" ", "") in dtype_to_datatype, f'{cpuOp.dtypes[0].replace(" ", "")} not in dtype_to_datatype'
    x_shape = array_to_str(cpuOp.shapes[1])
    w_shape = array_to_str(cpuOp.shapes[2])
    x_dtype = dtype_to_datatype[cpuOp.dtypes[0].replace(" ", "")]
 
    #print('cpuOp.args', cpuOp.args)
    stride = cpuOp.args[4].replace(" ", "")
    padding = cpuOp.args[5].replace(" ", "")
    dilations = cpuOp.args[6].replace(" ", "")
 
    #print('cpuOp.name', cpuOp.name, cpuOp.shapes)
    convOp = ConvOp(cpuOp.name, x_shape, w_shape, x_dtype, padding, stride, dilations, cpuOp.get_device_time(), cpuOp.get_bubble_time(), 1, cpuOp.get_shape_info())
    convOp.set_call_id(cpuOp.call_id, cpuOp.graph)
    for key in conv_list:
        if key.is_same(convOp):
            key.merge(convOp)
            return conv_list
 
    conv_list.append(convOp)
    return conv_list
 
def convert_CpuOp_to_AttnOp(cpuOp, sdp_list):
    assert len(cpuOp.shapes) > 0, f'SDP cant find shape info, check profiler args Record_shape==True?'
    q_shape = array_to_str(cpuOp.shapes[0])
    k_shape = array_to_str(cpuOp.shapes[1])
    v_shape = array_to_str(cpuOp.shapes[2])
 
    attnOp = AttnOp(cpuOp.name, q_shape, k_shape, v_shape, cpuOp.get_device_time(), cpuOp.get_bubble_time(), 1, cpuOp.get_shape_info())
    attnOp.set_call_id(cpuOp.call_id, cpuOp.graph)
    for key in sdp_list:
        if key.is_same(attnOp):
            key.merge(attnOp)
            return sdp_list
 
    sdp_list.append(attnOp)
 
    return sdp_list
 
def multiply_except_last(t):
    if len(t) > 2:
        # 使用reduce函数和operator.mul从t元组中除最后一个元素外的所有元素计算乘积
        from functools import reduce
        from operator import mul
        product = reduce(mul, t[:-1])
        return product
    elif len(t) == 2 or len(t) == 1:
        return t[0]
    else:
        return t
 
def convert_CpuOp_to_LinearOp(cpuOp, mm_list, enable_last = False):
    if not enable_last :
        n = str(multiply_except_last(cpuOp.shapes[0]))
        m = str(multiply_except_last(cpuOp.shapes[1]))
        k = str(cpuOp.shapes[0][len(cpuOp.shapes[0]) - 1])
        beta = str( 0 if len(cpuOp.shapes) < 3 or len(cpuOp.shapes[2]) <= 0 else 1)
    else:
        n = str(multiply_except_last(cpuOp.last_shapes[0]))
        m = str(multiply_except_last(cpuOp.last_shapes[1]))
        k = str(cpuOp.shapes[0][len(cpuOp.last_shapes[0]) - 1])
        beta = str( 0 if len(cpuOp.last_shapes) < 3 or len(cpuOp.last_shapes[2]) <= 0 else 1)
         
    linearOp = LinearOp(cpuOp.name, m, n, k, beta, cpuOp.get_device_time(), cpuOp.get_bubble_time(), 1, cpuOp.get_shape_info())
    linearOp.set_call_id(cpuOp.call_id, cpuOp.graph, cpuOp.get_last_shape_info())
 
    for key in mm_list:
        if key.is_same(linearOp):
            key.merge(linearOp)
            return mm_list
     
    mm_list.append(linearOp)
    return mm_list
 
def convert_Shape_to_BMNK(input, weight, transpose_result):
    batchcount = input[0]
     
    #print('cpuOp.shapes[0][0]', cpuOp.shapes[0][0], type(cpuOp.shapes[0][0]))
    result_sizes = (input[0], input[1],weight[2])
    leading_dim = 1 if transpose_result else 2
    m = result_sizes[2 if transpose_result else 1]
    n = result_sizes[leading_dim]
    batch = weight if transpose_result else input
    k = batch[leading_dim]
 
    return batchcount, m, n, k
 
def convert_CpuOp_to_BmmOp(cpuOp, bmm_list, enable_last = False):
    if not enable_last :
        b, m, n, k = convert_Shape_to_BMNK(cpuOp.shapes[0], cpuOp.shapes[1], False)
        b, transpose_m, transpose_n, transpose_k = convert_Shape_to_BMNK(cpuOp.shapes[0], cpuOp.shapes[1], True)
 
        beta = str( 0 if len(cpuOp.shapes) < 3 or len(cpuOp.shapes[2]) <= 0 else 1)
    else:
        b, m, n, k = convert_Shape_to_BMNK(cpuOp.last_shapes[0], cpuOp.last_shapes[1], False)
        b, transpose_m, transpose_n, transpose_k = convert_Shape_to_BMNK(cpuOp.last_shapes[0], cpuOp.last_shapes[1], True)
 
        beta = str( 0 if len(cpuOp.last_shapes) < 3 or len(cpuOp.last_shapes[2]) <= 0 else 1)
     
    bmmOp = BMMOp(cpuOp.name, str(b), str(m), str(n), str(k), str(transpose_m), str(transpose_n), str(transpose_k), beta, cpuOp.get_device_time(), cpuOp.get_bubble_time(), 1, cpuOp.get_shape_info())
    bmmOp.set_call_id(cpuOp.call_id, cpuOp.graph, cpuOp.get_last_shape_info())
    for key in bmm_list:
        if key.is_same(bmmOp):
            key.merge(bmmOp)
            return bmm_list
     
    bmm_list.append(bmmOp)
    return bmm_list
 
def convert_CpuOp_to_MatmulOp(cpuOp, matmul_list):
    if len(cpuOp.shapes[0]) < 3 and len(cpuOp.shapes[1]) < 3:
        print('MatmulOp cpuOp.shapes', len(cpuOp.shapes[0]), len(cpuOp.shapes[1]))
        assert False, "error message"
 
    if cpuOp.last_name == 'aten::mm':
        mmlist = list()
        mmlist= convert_CpuOp_to_LinearOp(cpuOp, mmlist, True)
        matmulOp = MatmulOp(mmlist[0])
    else:
        bmmlist = list()
        bmmlist= convert_CpuOp_to_BmmOp(cpuOp, bmmlist, True)
        matmulOp = MatmulOp(bmmlist[0])
 
    for key in matmul_list:
        if key.is_same(matmulOp):
            key.merge(matmulOp)
            return matmul_list
     
    matmul_list.append(matmulOp)
 
    return matmul_list
 
def convert_CpuOp_to_CuOp(cpuOp, cu_list):
    cuOp = CuOp(cpuOp.name, cpuOp.get_device_time(), cpuOp.get_bubble_time(), 1, cpuOp.get_shape_info())
    cuOp.set_call_id(cpuOp.call_id, cpuOp.graph)
    for key in cu_list:
        if key.is_same(cuOp):
            key.merge(cuOp)
            return cu_list
 
    cu_list.append(cuOp)
 
    return cu_list
 
def merge_shape_log(profiler_ops, dldnn_dlblas_log_ops):
    for prof_op in profiler_ops:
        for dl_op in dldnn_dlblas_log_ops:
            #print('dl_op,', prof_op.name, dl_op.name, dl_op.log_str)
            if prof_op.is_same(dl_op):
                for log in dl_op.log_str:
                    prof_op.add_log(log)