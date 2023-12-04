import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import random

# Define the visualization function
def visualize_labeled_images(model, dataloader, class_names, num_samples=2):
    # set up the figure and axis for visualization
    fig, axes = plt.subplots(len(class_names), num_samples, figsize=(2*num_samples, 2*len(class_names)))
    # loop over each class
    for i, class_name in enumerate(class_names):
        # get indices of images and randomly sample images
        class_indices = [idx for idx, label in enumerate(dataloader.dataset.targets) if label == i]
        random_samples = random.sample(class_indices, num_samples)
        # loop over each random sample for visualization
        for j, idx in enumerate(random_samples):
            # get the image, true label and make prediction with model
            img, label = dataloader.dataset[idx]
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output.logits, 1)
            # display the image on the plot
            ax = axes[i, j]
            ax.imshow(np.transpose(img.squeeze().cpu().numpy(), (1, 2, 0)))
            ax.axis('off')
            # set the title of the plot to show true label and predicted label
            ax.set_title(f'True: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
    # plot and save visualizations
    plt.tight_layout()
    plt.savefig('visualizations.png')
    plt.show()

def main():
    # load the saved fine-tuned ViT model
    model_path = 'model_vit_0.0001_32_SGD.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    # define same data transformations
    data_transforms = {
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    }
    # only need test set for evaluation/metrics
    test_dataset = ImageFolder(root='data/test', transform=data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # get the confusion matrix and other metrics
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    confusion_mat = confusion_matrix(all_labels, all_predicted)
    accuracy = accuracy_score(all_labels, all_predicted)
    # calculate precision, recall, and f1-score for each class
    precision_scores = precision_score(all_labels, all_predicted, average=None)
    recall_scores = recall_score(all_labels, all_predicted, average=None)
    f1_scores = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores)
    # calculate global precision, recall, and F1-score
    overall_precision = precision_score(all_labels, all_predicted, average='macro')
    overall_recall = recall_score(all_labels, all_predicted, average='macro')
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    # print accuracy metrics
    print(f"Total Accuracy: {accuracy:.2f}")
    # get classes from test set and print metrics
    class_names = test_loader.dataset.classes
    for i, class_name in enumerate(class_names):
        print(f"Class: {class_name}")
        print(f"Precision: {precision_scores[i]:.2f}")
        print(f"Recall: {recall_scores[i]:.2f}")
        print(f"F1-Score: {f1_scores[i]:.2f}")
        print("=" * 20)
    print("Globally:")
    print(f"Overall Precision: {overall_precision:.2f}")
    print(f"Overall Recall: {overall_recall:.2f}")
    print(f"Overall F1-Score: {overall_f1:.2f}")
    print("=" * 20)
    # plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    # visualize random images from each class
    visualize_labeled_images(model, test_loader, test_dataset.classes, num_samples=2)

if __name__ == "__main__":
    main()
