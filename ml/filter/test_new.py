import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MultiChannelDataset
from filter.modelV3_15 import VideoClassifierV3_15
from embedding import prepare_batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse


def load_model(checkpoint_path):
    """加载模型权重"""
    model = VideoClassifierV3_15()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    return model


def load_test_data(test_file):
    """加载测试数据"""
    test_dataset = MultiChannelDataset(test_file, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
    return test_loader


def convert_to_binary_labels(labels):
    """将三分类标签转换为二分类：类别1和2合并为类别1"""
    binary_labels = np.where(labels >= 1, 1, 0)
    return binary_labels


def run_inference(model, test_loader):
    """运行推理"""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # 准备文本数据
            batch_tensor = prepare_batch(batch['texts'])

            # 前向传播
            logits = model(batch_tensor)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def calculate_metrics(y_true, y_pred):
    """计算二分类指标"""
    # 转换为二分类
    y_true_binary = convert_to_binary_labels(y_true)
    y_pred_binary = convert_to_binary_labels(y_pred)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true_binary, y_pred_binary)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # 如果只有一类的情况
        tn = fp = fn = tp = 0
        if y_true_binary.sum() == 0:
            tn = (y_true_binary == y_pred_binary).sum()
        else:
            tp = ((y_true_binary == 1) & (y_pred_binary == 1)).sum()
            fp = ((y_true_binary == 0) & (y_pred_binary == 1)).sum()
            fn = ((y_true_binary == 1) & (y_pred_binary == 0)).sum()

    # 计算指标
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    return {
        'Acc': accuracy,
        'Prec': precision,
        'Recall': recall,
        'F1': f1,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn
    }


def main():
    parser = argparse.ArgumentParser(description='Test model on JSONL data')
    parser.add_argument('--model_path', type=str, default='./filter/checkpoints/best_model_V3.17.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--test_file', type=str, default='./data/filter/test1.jsonl',
                        help='Path to test JSONL file')
    args = parser.parse_args()

    # 加载模型
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path)

    # 加载测试数据
    print(f"Loading test data from {args.test_file}")
    test_loader = load_test_data(args.test_file)

    # 运行推理
    print("Running inference...")
    y_pred, y_true = run_inference(model, test_loader)

    # 计算指标
    print("\nCalculating metrics...")
    metrics = calculate_metrics(y_true, y_pred)

    # 打印结果
    print("\n=== Test Results (Binary Classification) ===")
    print(f"Accuracy (Acc):     {metrics['Acc']:.4f}")
    print(f"Precision (Prec):   {metrics['Prec']:.4f}")
    print(f"Recall:             {metrics['Recall']:.4f}")
    print(f"F1 Score:           {metrics['F1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP (True Positive):  {metrics['TP']}")
    print(f"  TN (True Negative):  {metrics['TN']}")
    print(f"  FP (False Positive): {metrics['FP']}")
    print(f"  FN (False Negative): {metrics['FN']}")

    # 显示原始三分类分布
    print(f"\n=== Original Label Distribution ===")
    unique, counts = np.unique(y_true, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Class {label}: {count} samples")


if __name__ == '__main__':
    main()
