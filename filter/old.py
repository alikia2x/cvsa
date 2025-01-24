import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, data_path, sentence_transformer):
        self.data = []
        self.sentence_transformer = sentence_transformer
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        title = item["title"]
        description = item["description"]
        tags = item["tags"]
        label = item["label"]

        # 获取每个特征的嵌入
        title_embedding = self.get_embedding(title)
        description_embedding = self.get_embedding(description)
        tags_embedding = self.get_embedding(" ".join(tags))

        # 将嵌入连接起来
        combined_embedding = torch.cat([title_embedding, description_embedding, tags_embedding], dim=0)

        return combined_embedding, label

    def get_embedding(self, text):
        # 使用SentenceTransformer生成嵌入
        embedding = self.sentence_transformer.encode(text)
        return torch.tensor(embedding)

class VideoClassifier(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=256, output_dim=3):
        super(VideoClassifier, self).__init__()
        # 每个特征的嵌入维度是embedding_dim，总共有3个特征
        total_embedding_dim = embedding_dim * 3

        # 全连接层
        self.fc1 = nn.Linear(total_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, embedding_features):
        # 全连接层
        x = torch.relu(self.fc1(embedding_features))
        output = self.fc2(x)
        output = self.log_softmax(output)
        return output

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for embedding_features, labels in dataloader:
        embedding_features = embedding_features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(embedding_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for embedding_features, labels in dataloader:
            embedding_features = embedding_features.to(device)
            labels = labels.to(device)
            outputs = model(embedding_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# 超参数
hidden_dim = 256
output_dim = 3
batch_size = 32
num_epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载数据集
tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3")
sentence_transformer = SentenceTransformer("Thaweewat/jina-embedding-v3-m2v-1024")
dataset = VideoDataset("labeled_data.jsonl", sentence_transformer=sentence_transformer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# 初始化模型
model = VideoClassifier(embedding_dim=768, hidden_dim=256, output_dim=3).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 5
# 训练和验证
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, dataloader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

# 保存模型
torch.save(model.state_dict(), "video_classifier.pth")
model.eval()  # 设置为评估模式

# 2. 定义推理函数
def predict(model, sentence_transformer, title, description, tags, device):
    # 将输入数据转换为嵌入
    title_embedding = torch.tensor(sentence_transformer.encode(title)).to(device)
    description_embedding = torch.tensor(sentence_transformer.encode(description)).to(device)
    tags_embedding = torch.tensor(sentence_transformer.encode(" ".join(tags))).to(device)

    # 将嵌入连接起来
    combined_embedding = torch.cat([title_embedding, description_embedding, tags_embedding], dim=0).unsqueeze(0)

    # 推理
    with torch.no_grad():
        output = model(combined_embedding)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()