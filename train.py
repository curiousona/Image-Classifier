import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse 
import os

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--arch", type=str, default="vgg19", help="model for training")
    parser.add_argument("--learning_rate", help="learning rate for the training process", default=0.001, type=float)
    parser.add_argument("--epochs", help="Number of training iterations", type=int, default=1)
    parser.add_argument("--save_dir", help="Directory for saving the model", type=str, default="")
    parser.add_argument("--hidden_units", help="Number of hidden units", type=int, default=512)
    parser.add_argument("--gpu", help="using gpu for processing", action="store_true")
    return parser.parse_args()



# defining dataloaders
def load_data(data_dir, batch_size=16):
    test_dir=data_dir+'/train'
    train_dir=data_dir+"/test"
    valid_dir=data_dir+"/valid"
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    # Load the datasets with ImageFolder
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    val_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])

    image_datasets = [train_data, test_data, val_data]

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    dataloaders = [trainloader, testloader, valloader]

    return image_datasets, dataloaders




# classifier builder for the models
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

def validate(model,valloader,criterion, device):
    validation_loss = 0
    accuracy = 0
    model.eval()
                
    with torch.no_grad():
        for images, labels in valloader:  # Using the validation loader here
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
            validation_loss += batch_loss.item()
                    
            # Getting the accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        validation_loss/=len(valloader)
        validation_accuracy=accuracy/len(valloader)
        return validation_loss, validation_accuracy


def train_model(model, epochs, trainloader, valloader,criterion, optimizer, device):
    model.to(device)
    running_loss=0
    steps=0
    print_every=5
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps+=1
            optimizer.zero_grad()
            images, labels=images.to(device), labels.to(device)
            logps=model.forward(images)
            loss=criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            
            if(steps%print_every==0):
                validation_loss, validation_accuracy=validate(model,valloader, criterion, device)
                print(f"Epoch {epoch + 1}/{epochs}.."
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {validation_loss:.3f}.. "
                      f"Validation accuracy: {validation_accuracy:.3f} ")
                running_loss = 0
                model.train()
    return model

def test(model, criterion, testloader, device):
    test_loss = 0
    accuracy = 0
    model.eval()
    model.to(device)

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            logps = model.forward(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            # Getting the accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    test_accuracy=accuracy * 100 / len(testloader)
    test_loss=test_loss / len(testloader)
    return test_loss, test_accuracy


# method for saving the model

def save_model(model , arch, optimizer, hidden_units, image_datasets, epochs, save_dir):
    chk_point = {
        'architecture':arch,
        'hidden_units':hidden_units,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'class_to_idx': image_datasets[0].class_to_idx
    }
    os.makedirs(save_dir, exist_ok=True)
    path=os.path.join(save_dir,'checkpoint.pth')
    torch.save(chk_point, path)


def main():
    args = arg_parse()
    data_dir=args.data_dir
    arch=args.arch
    learning_rate=args.learning_rate
    epochs=args.epochs
    hidden_units=args.hidden_units
    gpu=args.gpu
    save_dir=args.save_dir
    available_models=['vgg19', 'alexnet']
    if(arch in available_models):
        if(arch=='vgg19'):
            model=models.vgg19(pretrained=True)
        else:
            model=models.alexnet(pretrained=True)
        for params in model.parameters():
            params.require_grads=False
        
        if(gpu):
            device="cuda" if torch.cuda.is_available() else "cpu"
            
            if(device=="cpu"):
               print("cuda is not available selecting cpu")
        else: 
            device="cpu"
        print("\n\n\nTraining details")
        print("Data Directory:",data_dir)
        print("Architecture:", arch)
        print("Learning Rate:", learning_rate)
        print("Epochs:", epochs)
        print("Save Directory:", save_dir)
        print("Hidden Units:", hidden_units)
        print("device:", device)
        print()
        
#       load the image datasets
        image_datasets, dataloaders=load_data(data_dir)
        trainloader, testloader,valloader=dataloaders
        print("Successfully loaded the data")  
        
#       load the classifier
        model.classifier=build_classifier(arch, hidden_units)
        print("Model built now ready for the training--------")
        optimizer=optim.Adam(model.classifier.parameters(),lr=learning_rate)
        criterion=nn.NLLLoss()
        
        print("---------- Training the model ---------")
        model=train_model(model, epochs, trainloader, valloader, criterion, optimizer, device)
        print("---------- Model trained successfully -----------")
        print("---------- Testing the model -------")
        test_loss, test_accuracy=test(model, criterion, testloader, device)    
        print("----------- Testing done -----")
        print(f"Test accuracy: {test_accuracy:.3f}%")
        print(f"Test loss: {test_loss}")
        
        if(save_model !=""):
            print(f"Saving the model at {save_dir}")
            save_model(model , arch, optimizer, hidden_units, image_datasets, epochs, save_dir)
            
    else:
        print(f"Please choose from the available models {available_models}")


if __name__ =="__main__":
    main()