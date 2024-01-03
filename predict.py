import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse 
import os

from PIL import Image
import numpy as np

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image_path)
    # reize
    pil_image=pil_image.resize((256,256))
    
    # centre crop
    width, height = pil_image.size   # Get dimensions
    new_width, new_height = 224, 224
    
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    xcord_right = round(width - new_width) - left
    xcord_bottom = round(height - new_height) - top
    right = width - xcord_right
    bottom = height - xcord_bottom

    pil_image = pil_image.crop((left, top, right, bottom))
    
    np_image = np.array(pil_image)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean)/std
    
    # tranpose color channge to 1st dim
    np_image = np_image.transpose((2 , 0, 1))
    
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)
   
    return tensor


def build_classifier(arch, hidden_units):
    if(arch=="vgg19"):
        input_features=25088
    else: input_features=9216
    classifier= nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    return classifier


def load_model(path):
    chk_point = torch.load(path, map_location=torch.device('cpu'))
    arch=chk_point['architecture']
    if(arch=="vgg19"):
        model=models.vgg19(pretrained=True)
    else:
        model=models.alexnet(pretrained=True)   
    hidden_units=chk_point['hidden_units']
    model.classifier=build_classifier(arch, hidden_units)
    model.load_state_dict(chk_point['model_state_dict'])
    model.class_to_idx = chk_point['class_to_idx'] 
    return model


def predict(image_path, model, topk, cat_to_json, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img=process_image(image_path)
    #add batch dimensions
    img = img.unsqueeze(0) 
    img=img.to(device)
    with torch.no_grad():
        logps=model.forward(img)
        ps=torch.exp(logps)
        top_p, top_class = ps.topk(topk,dim=1)
        top_cls_indices = top_cls[0].tolist()
        top_cls_names = [cat_to_name[str(index)] for index in top_cls_indices]
    return top_p, top_cls_names




def get_input_args():
    parser=argparse.ArgumentParser()

    parser.add_argument('image_path', help="This is a image file that you want to classify")
    parser.add_argument('model_checkpoint', help="This is file path of a checkpoint file")

    parser.add_argument('--category_names', help="path to a json file that maps categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', help="most top_k category", default=5, type=int)
    parser.add_argument('--gpu', help="Use gpu", action='store_true')
    return parser.parse_args()

def main():
    args=get_input_args()
    print(args)
    image_path=args.image_path
    checkpoint_path=args.model_checkpoint
    category_name=args.category_names
    top_k=args.top_k
    gpu=args.gpu
    if(gpu):
        device="cuda" if torch.cuda.is_available() else "cpu"
        if(device=="cpu"):
            print("cuda is not available selecting cpu")
        else: 
            device="cpu"
    model=load_model(checkpoint_path)
    model=model.to(device)
    top_ps, top_cls=predict(image_path, model, top_k, category_name, device)
    for i in range(len(top_cls)):
        print(f"Category: {top_cls[i]}, Probability: {top_ps[0][i].item():.4f}")
        
if __name__=="__main__":
    main()
        
        
        
        

       
        
        
        

    