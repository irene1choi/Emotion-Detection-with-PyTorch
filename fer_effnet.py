import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import csv

# Used https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html as a resource

# import pretrained effnet model and replace final layer
def create_model(device):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    # freeze all layer except the last one
    for param in model.parameters():
        param.requires_grad = False
    # reinitialize the last predictor layer for finetuning
    num_ftrs = model._fc.in_features
    # 6 classes for 6 emotions
    model._fc = torch.nn.Linear(num_ftrs, 6)  
    # move the model to the gpu
    model.to(device)
    return model

# Training Function
def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=25):
    # store the accuracies to compare parameter results
    accuracies = []
    # training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        # get the epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}")
        # evaluate model after each epoch and record accuracy
        epoch_accuracy = evaluate_model(model, test_loader, device)
        accuracies.append(epoch_accuracy)
    return accuracies

# Evaluation function
def evaluate_model(model, dataloader, device):
    # set the model to eval mode
    model.eval()
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            # move inptus and labels to gpu
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # get the index of the predicted label
            _, predicted = torch.max(outputs.data, 1)
            # add batch size to the total number of instances
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predicted) * 100
    print(f"Accuracy on the test set: {accuracy:.2f}%")
    print("=" * 20)
    # calculate f1 and recall scores for each class
    f1_scores = f1_score(all_labels, all_predicted, average=None)
    precision_scores = precision_score(all_labels, all_predicted, average=None)
    recall_scores = recall_score(all_labels, all_predicted, average=None)
    # print out recall and precision scores for each class
    for i, class_name in enumerate(dataloader.dataset.classes):
        print(f"Class: {class_name}")
        print(f"Precision: {precision_scores[i]:.4f}")
        print(f"Recall: {recall_scores[i]:.4f}")
        print("=" * 20)
    return accuracy

def main():
    # check for GPU and set device to either cuda or cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set torch seed for reproducability
    torch.manual_seed(42)
    # define the data transformations
    # ensure that all the images are the same size (48x48 pixels)
    # convert jpgs to tensors and normalize
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    }
    # create model
    model = create_model(device)
    # define loss function
    criterion = torch.nn.CrossEntropyLoss()
    # use ImageFolder to parse the training dataset
    train_dataset = ImageFolder(root='data/train', transform=data_transforms['train'])
    test_dataset = ImageFolder(root='data/test', transform=data_transforms['test'])
    # parameter grid for testing
    learning_rates = [0.0001]
    batch_sizes = [32, 64]
    optimizers = ['Adam', 'SGD']
    # epochs = [5, 10, 15, 20]
    # store results in an array
    results = []
    # loop over parameter grid
    for lr in learning_rates:
        for bs in batch_sizes:
            for opt_name in optimizers:
                # adjusting data loaders for different batch sizes
                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
                # adjust optimizer
                if opt_name == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                # print for sanity check
                print(f"Training for lr={lr}, bs={bs}, optimizer={opt_name}")
                # get the accuracies of each epoch
                epoch_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=25)
                # save results for the parameters used
                results.append([lr, bs, opt_name] + epoch_accuracies)         
    # save results to CSV
    with open('sensitivity_analysis.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['learning_rate', 'batch_size', 'optimizer'] + [f'epoch_{i+1}_accuracy' for i in range(25)]
        writer.writerow(headers)
        writer.writerows(results)

if __name__ == "__main__":
    main()
