import onnx
import onnx.numpy_helper
import os
import string
import numpy as np
import math

def onnx2txt(input_path, output_path, fp16=True):
    QUANTIZE_UINT8 = False
    INFER_SHAPES = True
    DBG_PRINT_OUT = False

    if INFER_SHAPES:
        onnx.shape_inference.infer_shapes_path(input_path)

    model = onnx.load(input_path)
    output_dir = os.path.join(os.getcwd(), output_path)

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        for filename in os.listdir(output_dir):
             os.remove(os.path.join(output_dir, filename))
                
    def quantize(a: np.ndarray, t: str, op_type: str, in_out_index: int, from_left: float = 0.001, from_right: float = 0.001):
        
        q = True
        
        if (op_type == "Conv" and in_out_index == 2) or \
            (op_type == "InstanceNormalization" and in_out_index != 0) or \
            (op_type == "Resize" and in_out_index == 2):
            q = False
        
        if q:
            flat = a.flatten().tolist()
            s = [f for f in flat if math.isfinite(f)]
            s.sort()
            if len(s) == 1 and len(flat) == 1:
                scale = abs(flat[0])
                zero = 0 if flat[0] >= 0 else 2
                a = np.array([1], dtype="ubyte")
                t = "uint8[" + str(scale) + "," + str(zero) + "]"
            elif len(s) >= 2:
                left = s[int(len(s) * from_left)]
                right = s[int(len(s) * from_right * -1 - 1)]
                if left > 0 and right > 0:
                    left = 0
                elif left < 0 and right < 0:
                    right = 0
                if right > left:
                    scale = (right - left) / 255.0
                    zero = int(abs(left) / scale)
                    if zero > 255: zero = 255
                    a = (a / scale) + zero
                    a = np.clip(a, 0, 255).astype("ubyte")
                    t = "uint8[" + str(scale) + "," + str(zero) + "]"

        return a, t
                
    def add_line_to_model(line: str):
        
        with open(os.path.join(output_dir, "model.txt"), "a") as f:
            f.write(line + "\n")

    def get_final_name(name: str):
        
        final_name = ""
        for c in name:
            if c in string.ascii_letters + string.digits:
                final_name += c
            else:
                final_name += "_" + format(ord(c), 'X') + "_"
        
        return final_name

    op_constants = {}

    def search_name(name: str, node: onnx.NodeProto, in_out_index: int):

        weights = [t for t in model.graph.initializer if t.name == name]
        input_idxs = [i for i, n in enumerate(model.graph.node) for x in n.input if x == name]
        output_idxs = [i for i, n in enumerate(model.graph.node) for o in n.output if o == name]
        graph_inputs = [i for i in model.graph.input if i.name == name]
        graph_outputs = [o for o in model.graph.output if o.name == name]

        shapes = [i for i in model.graph.value_info if i.name == name]
        shape = ""
        
        if name in op_constants and len(weights) == 0:
            weights = [op_constants[name]]
        
        name = get_final_name(name)
        
        if len(shapes) == 1 and len(weights) == 1:
            shapes = []

        if len(shapes) + len(graph_inputs) + len(graph_outputs) + len(weights) != 1:
            raise ValueError("Error: " + name)
        elif len(shapes) == 1:
            shape = ",".join(str(d.dim_value) for d in shapes[0].type.tensor_type.shape.dim)
        elif len(graph_inputs) == 1:
            shape = ",".join(str(d.dim_value) for d in graph_inputs[0].type.tensor_type.shape.dim)
        elif len(graph_outputs) == 1:
            shape = ",".join(str(d.dim_value) for d in graph_outputs[0].type.tensor_type.shape.dim)
        elif len(weights) == 1:
            
            a = onnx.numpy_helper.to_array(weights[0])
            
            if node.op_type == "Mul" and in_out_index == 1 and str(a.dtype) == "int64":
                a = a.astype("float32")
            
            if fp16 == True and str(a.dtype) == "float32":
                a = a.astype("float16")
            
            t = str(a.dtype)
            if t != "float32" and t != "int64" and t != "float16":
                raise ValueError("Error")
            
            if QUANTIZE_UINT8 == True and str(a.dtype) == "float32":
                a, t = quantize(a, t, node.op_type, in_out_index)
                
            def save_to_disk(n, arr):
                nonlocal shape
                shape = t + ":" + ",".join(str(d) for d in arr.shape)
                n = n + ".bin"
                arr.tofile(os.path.join(output_dir, n))
                return n
                
            if node.op_type == "Gemm":
                transA = next(iter(a for a in node.attribute if a.name == "transA" and a.i != 0 and in_out_index == 0), None)
                transB = next(iter(a for a in node.attribute if a.name == "transB" and a.i != 0 and in_out_index == 1), None)
                trans = False
                if transA is not None:
                    node.attribute.remove(transA)
                    trans = True
                if transB is not None:
                    node.attribute.remove(transB)
                    trans = True
                if trans:
                    a = np.transpose(a)
                    name = name + "_transposed"
            elif node.op_type == "Conv":
                if in_out_index == 0 or in_out_index == 1:
                    if len(a.shape) != 4:
                        raise ValueError('Error')
                    save_to_disk(name + "_nhwc", np.transpose(a, (0, 2, 3, 1)))
                    name = name + "_nchw"
                    
            name = save_to_disk(name, a)

        else:
            raise ValueError("Error")

        return name, weights, input_idxs, output_idxs, graph_inputs, graph_outputs, shape

    op_stats = {}

    for idx, node in enumerate(model.graph.node):
        
        if len(node.input) == 0 or len(node.output) == 0:
            if node.op_type == "Constant" and len(node.output) == 1:
                values = [a for a in node.attribute if a.name == "value"]
                if len(values) != 1:
                    raise ValueError("Error")
                value = values[0]
                if value.type != onnx.AttributeProto.TENSOR:
                    raise ValueError("Error")
                op_constants[node.output[0]] = value.t
                continue
            raise ValueError("Error")
        
        if node.op_type in op_stats:
            op_stats[node.op_type] += 1;
        else:
            op_stats[node.op_type] = 1;
        
        line = []
        
        line.append(node.name + ":" + node.op_type)

        inputs = []
        for input_index, input_name in enumerate(node.input):
            
            if len(input_name) == 0:
                inputs.append("")
                continue
            
            input_name, weights, input_idxs, output_idxs, graph_inputs, graph_outputs, shape = search_name(input_name, node, input_index)
            
            if len(output_idxs) >= 2:
                raise ValueError("Error")
            elif len(output_idxs) == 1 and output_idxs[0] >= idx:
                raise ValueError("Error")
            elif len(weights) == 0 and len(output_idxs) == 0 and len(graph_inputs) == 0:
                raise ValueError("Error")
                
            inputs.append(input_name + "(" + shape + ")")
            
        if len(inputs) == 0:
            raise ValueError("Error")
        else:
            line.append("input:" + ";".join(inputs))

        outputs = []
        for output_index, output_name in enumerate(node.output):
            
            if len(output_name) == 0: raise ValueError("Error")
                
            output_name, weights, input_idxs, output_idxs, graph_inputs, graph_outputs, shape = search_name(output_name, node, -output_index-1)
            
            if any(i <= idx for i in input_idxs):
                raise ValueError("Error")
            elif len(input_idxs) == 0 and len(graph_outputs) == 0:
                raise ValueError("Error")
            elif len(weights) != 0:
                raise ValueError("Error")
                
            outputs.append(output_name + "(" + shape + ")")
            
        if len(outputs) == 0:
            raise ValueError("Error")
        else:
            line.append("output:" + ";".join(outputs))
                    
        attrs = []
        for a in node.attribute:
            
            attr = ""
            if a.type == onnx.AttributeProto.INT:
                attr = str(a.i)
            elif a.type == onnx.AttributeProto.FLOAT:
                attr = str(a.f)
            elif a.type == onnx.AttributeProto.STRING:
                if isinstance(a.s, str):
                    attr = a.s
                elif isinstance(a.s, bytes):
                    attr = a.s.decode("utf-8", errors="ignore")
                else:
                    attr = str(a.s)
            elif a.type == onnx.AttributeProto.INTS:
                attr = ",".join(str(x) for x in a.ints)
            elif a.type == onnx.AttributeProto.TENSOR:
                v = onnx.numpy_helper.to_array(a.t).flatten().tolist()
                if len(v) != 1:
                    raise ValueError("Error")
                attr = str(v[0])
            else:
                raise ValueError("Error")
                
            attrs.append(a.name + ":" + attr)
            
        if len(attrs) != 0:
            line.append(";".join(attrs))
            
        if any("*" in t for t in line):
            raise ValueError("Error")
        else:
            line_str = "*".join(line)
            add_line_to_model(line_str)
            if DBG_PRINT_OUT == True: print(line_str)

    total = 0
    for name, count in op_stats.items():
        total += count
        print(name, "->", count)
    print("TOTAL", "->", total)
