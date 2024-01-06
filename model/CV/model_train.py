
from model.CV.Data_loader import *
# import torch
# print(torch.__version__)

# # 查看数据形状
# DATADIR = 'workspace/CV/train'
# train_loader = data_loader(DATADIR,batch_size=20, mode='train')
# data_reader = train_loader()
# data = next(data_reader) #返回迭代器的下一个项目给data
# # 输出表示： 图像数据（batchsize，通道数，224*224）标签（batchsize，标签维度）
# print("train mode's shape:")
# print("data[0].shape = %s, data[1].shape = %s" %(data[0].shape, data[1].shape))

# eval_loader = data_loader(DATADIR,batch_size=20, mode='eval')
# data_reader = eval_loader()
# data = next(data_reader)
# # 输出表示： 图像数据（batchsize，通道数，224*224）标签（batchsize，标签维度）
# print("eval mode's shape:")
# print("data[0].shape = %s, data[1].shape = %s" %(data[0].shape, data[1].shape))





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision import models

# Define the CNN model
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define a custom dataset
class EmotionDataset(Dataset):
    def __init__(self, datadir, transform=None):
        self.datadir = datadir
        self.transform = transform
        self.filenames = os.listdir(datadir)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = os.path.join(self.datadir, self.filenames[idx])
        img = cv2.imread(filepath)
        img = transform_img(img)

        if self.filenames[idx][:2] == 'SA':
            label = 0
        elif self.filenames[idx][:2] == 'DI':
            label = 1
        elif self.filenames[idx][:2] == 'HA':
            label = 2
        elif self.filenames[idx][:2] == 'FE':
            label = 3
        elif self.filenames[idx][:2] == 'SU':
            label = 4
        elif self.filenames[idx][:2] == 'NE':
            label = 5
        elif self.filenames[idx][:2] == 'AN':
            label = 6
        else:
            raise Exception('Unexpected file name')

        return img, label


if __name__=="__main__":
    # Create data loaders
    train_dataset = EmotionDataset(datadir='workspace/CV/train', transform=None)
    valid_dataset = EmotionDataset(datadir='workspace/CV/val', transform=None)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EmotionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize variables for tracking best model
    best_accuracy = 0.0
    best_model_state = None

    # Train the model
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()



            total_loss += loss.item()
            total_batches += 1
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            # Print training statistics every few batches
            if (batch_idx + 1) % 20 == 0:
                average_loss = total_loss / total_batches
                accuracy = total_correct / total_samples
                print \
                    (f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
                total_loss = 0.0
                total_batches = 0

        # 在每个 epoch 结束后打印整个 epoch 的平均损失
        average_epoch_loss = total_loss / total_batches
        epoch_accuracy = total_correct / total_samples
        print \
            (f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_epoch_loss:.4f}, Epoch Accuracy: {epoch_accuracy:.4f}')

        # Validate the model
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            accuracy = total_correct / total_samples
            print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')

            # Save the model if it has the best validation accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict()


    # Save the best model to a file
    torch.save(best_model_state, 'best_emotion_model.pth')
