import torch
from modelV3_12 import VideoClassifierV3_12


def export_onnx(model_path="./filter/checkpoints/best_model_V3.13.pt", 
               onnx_path="./model/video_classifier_v3_13.onnx"):
    # 初始化模型
    model = VideoClassifierV3_12()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 创建符合输入规范的虚拟输入
    dummy_input = torch.randn(1, 4, 1024)  # [batch=1, channels=4, embedding_dim=1024]
    
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["channel_features"],
        output_names=["logits"],
        dynamic_axes={
            "channel_features": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=13,
        do_constant_folding=True
    )
    print(f"模型已成功导出到 {onnx_path}")

# 执行导出
export_onnx()