import onnx
from onnx import shape_inference

model = './vit.onnx'

onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model)), model)