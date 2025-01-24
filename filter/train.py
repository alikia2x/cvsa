import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import MultiChannelDataset
from modelV3_2 import VideoClassifierV3_2
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report
import os
import torch
from torch.utils.tensorboard import SummaryWriter  # 引入 TensorBoard
import time


# 动态生成子目录名称
run_name = f"run_{time.strftime('%Y%m%d_%H%M')}"
log_dir = os.path.join('./filter/runs', run_name)

# 初始化 SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

# 创建数据集
train_dataset = MultiChannelDataset('./data/filter/labeled_data.jsonl', mode='train')
eval_dataset = MultiChannelDataset('./data/filter/labeled_data.jsonl', mode='eval')

# 加载test数据集
test_file = './data/filter/test.jsonl'
if not os.path.exists(test_file):
    # 如果test文件不存在，先创建
    _ = MultiChannelDataset('./data/filter/labeled_data.jsonl', mode='train')
    
test_dataset = MultiChannelDataset(test_file, mode='test')

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=24, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

# 初始化模型和SentenceTransformer
sentence_transformer = SentenceTransformer("Thaweewat/jina-embedding-v3-m2v-1024")
model = VideoClassifierV3_2()
checkpoint_name = './filter/checkpoints/best_model_V3.2.pt'

# 模型保存路径
os.makedirs('./filter/checkpoints', exist_ok=True)

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=2e-3)
criterion = nn.CrossEntropyLoss()

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 传入文本字典和sentence_transformer
            logits = model(input_texts=batch['texts'], sentence_transformer=sentence_transformer)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
    
    # 计算每个类别的 F1、Recall、Precision 和 Accuracy
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 获取每个类别的详细指标
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    
    return f1, recall, precision, accuracy, class_report

print(f"Trainable parameters: {count_trainable_parameters(model)}")

# 训练循环
best_f1 = 0
total_step = 0
eval_interval = 50

for epoch in range(8):
    model.train()
    epoch_loss = 0
    
    # 训练阶段
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        # 传入文本字典和sentence_transformer
        logits = model(input_texts=batch['texts'], sentence_transformer=sentence_transformer)
        
        loss = criterion(logits, batch['label'])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # 记录训练损失
        writer.add_scalar('Train/Loss', loss.item(), total_step)
        total_step += 1
        
        # 每隔 eval_interval 步执行验证
        if total_step % eval_interval == 0:
            eval_f1, eval_recall, eval_precision, eval_accuracy, eval_class_report = evaluate(model, eval_loader)
            writer.add_scalar('Eval/F1', eval_f1, total_step)
            writer.add_scalar('Eval/Recall', eval_recall, total_step)
            writer.add_scalar('Eval/Precision', eval_precision, total_step)
            writer.add_scalar('Eval/Accuracy', eval_accuracy, total_step)
            
            print(f"Step {total_step}")
            print(f"  Eval F1: {eval_f1:.4f} | Eval Recall: {eval_recall:.4f} | Eval Precision: {eval_precision:.4f} | Eval Accuracy: {eval_accuracy:.4f}")
            print("  Eval Class Report:")
            for cls, metrics in eval_class_report.items():
                if cls.isdigit():  # 只打印类别的指标
                    print(f"    Class {cls}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}, Support: {metrics['support']}")
            
            # 保存最佳模型
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                torch.save(model.state_dict(), checkpoint_name)
                print("  Saved best model")
    
    # 记录每个 epoch 的平均训练损失
    avg_epoch_loss = epoch_loss / len(train_loader)
    writer.add_scalar('Train/Epoch_Loss', avg_epoch_loss, epoch)
    
    # 每个 epoch 结束后执行一次完整验证
    train_f1, train_recall, train_precision, train_accuracy, train_class_report = evaluate(model, train_loader)
    eval_f1, eval_recall, eval_precision, eval_accuracy, eval_class_report = evaluate(model, eval_loader)
    
    writer.add_scalar('Train/Epoch_F1', train_f1, epoch)
    writer.add_scalar('Train/Epoch_Recall', train_recall, epoch)
    writer.add_scalar('Train/Epoch_Precision', train_precision, epoch)
    writer.add_scalar('Train/Epoch_Accuracy', train_accuracy, epoch)
    writer.add_scalar('Eval/Epoch_F1', eval_f1, epoch)
    writer.add_scalar('Eval/Epoch_Recall', eval_recall, epoch)
    writer.add_scalar('Eval/Epoch_Precision', eval_precision, epoch)
    writer.add_scalar('Eval/Epoch_Accuracy', eval_accuracy, epoch)
    
    print(f"Epoch {epoch+1}")
    print(f"  Train Loss: {avg_epoch_loss:.4f}")
    print(f"  Train F1: {train_f1:.4f} | Train Recall: {train_recall:.4f} | Train Precision: {train_precision:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print("  Train Class Report:")
    for cls, metrics in train_class_report.items():
        if cls.isdigit():  # 只打印类别的指标
            print(f"    Class {cls}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}, Support: {metrics['support']}")
    
    print(f"  Eval F1: {eval_f1:.4f} | Eval Recall: {eval_recall:.4f} | Eval Precision: {eval_precision:.4f} | Eval Accuracy: {eval_accuracy:.4f}")
    print("  Eval Class Report:")
    for cls, metrics in eval_class_report.items():
        if cls.isdigit():  # 只打印类别的指标
            print(f"    Class {cls}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}, Support: {metrics['support']}")

# 测试阶段
print("\nTesting...")
model.load_state_dict(torch.load(checkpoint_name))
test_f1, test_recall, test_precision, test_accuracy, test_class_report = evaluate(model, test_loader)
writer.add_scalar('Test/F1', test_f1, total_step)
writer.add_scalar('Test/Recall', test_recall, total_step)
writer.add_scalar('Test/Precision', test_precision, total_step)
writer.add_scalar('Test/Accuracy', test_accuracy, total_step)
print(f"Test F1: {test_f1:.4f} | Test Recall: {test_recall:.4f} | Test Precision: {test_precision:.4f} | Test Accuracy: {test_accuracy:.4f}")
print("  Test Class Report:")
for cls, metrics in test_class_report.items():
    if cls.isdigit():  # 只打印类别的指标
        print(f"    Class {cls}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}, Support: {metrics['support']}")
        writer.add_scalar(f'Test/Class_{cls}_Precision', metrics['precision'], total_step)
        writer.add_scalar(f'Test/Class_{cls}_Recall', metrics['recall'], total_step)
        writer.add_scalar(f'Test/Class_{cls}_F1', metrics['f1-score'], total_step)
# 关闭 TensorBoard
writer.close()
