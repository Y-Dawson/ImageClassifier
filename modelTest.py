import timm
import torch
m = timm.create_model('mobilenetv3_large_100', pretrained=True)
print(m.eval())