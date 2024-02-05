import numpy as np # linear algebra
import torch   
import numpy as np
from torchvision import transforms
import torch.utils.data
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from modelfinal import CNN
device = torch.device('cpu')
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    download=True
)




loaders = {
   
    
    'test'  : torch.utils.data.DataLoader(test_data, 
                                          batch_size=1, 
                                          shuffle=False, 
                                          ),
}

model = CNN()
model.load_state_dict(torch.load('weights.pth',map_location=torch.device('cpu')))

all_labels = []
all_preds = []


model.eval()
model.to(device)
    
correct = 0
total = 0
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in loaders['test']:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)

        _, predicted = torch.max(preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    test_accuracy = correct / total
    print(f'Test Accuracy: {test_accuracy:.3f}')
    
def testing(model,images):
    model.eval()
    with torch.no_grad():
        images=images.float()
        output=model(images)
        _, predicted = torch.max(output, 1)
        print(predicted)
        
    
