import timm
import torch
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
def PrintParas(model):
    params = list(model.parameters())
    num_params = 0
    for param in params:
        curr_num_params = 1
        for size_count in param.size():
            curr_num_params *= size_count
        num_params += curr_num_params
    print("total number of parameters: " + str(num_params))


def PredictImage(filePath, modelName):
    #print("modelName:", modelName)
    
    model = timm.create_model(modelName, pretrained = True)
    print(model)
    PrintParas(model)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    img = Image.open(filePath).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
    print(out.shape)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    with open("./PredictCategory.txt", 'r') as f:
        categories = [s.strip() for s in f.readlines()]
    #print(probabilities[:10])
    #torch.save(model.state_dict(), 'C:\\Users\\Dawson\\Desktop\\毕设\\ImageClassifier\\ConstructModel\\ResMlp.pth')
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    return

#PredictImage("C:\\Users\\Dawson\\Desktop\\1.jpg", "resnet18")

#PredictImage("C:\\Users\\Dawson\\Desktop\\1.jpg", "vit_base_patch16_224")

PredictImage("C:\\Users\\Dawson\\Desktop\\1.jpg", "resmlp_12_224")