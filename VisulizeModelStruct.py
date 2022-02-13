import timm
import torch
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
model = timm.create_model( "vit_base_patch16_224", pretrained = True)
dummy_input1  = torch.randn(1, 3, 224, 224)
input_names = [ "actual_input_1"]
output_names = [ "output1" ]
torch.onnx.export(model, dummy_input1, "./vit.onnx", verbose=True, input_names=input_names, output_names=output_names)

#def PredictImage(modelName):
#    print("modelName:", modelName)
#    model = timm.create_model(modelName, pretrained = True)
#    
#
#PredictImage( "resnet18")