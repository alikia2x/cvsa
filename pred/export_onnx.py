import torch
import torch.onnx
from model import CompactPredictor

def export_model(input_size, checkpoint_path, onnx_path):
    model = CompactPredictor(input_size)
    model.load_state_dict(torch.load(checkpoint_path))

    dummy_input = torch.randn(1, input_size)

    model.eval()

    torch.onnx.export(model,  # Model to be exported
                    dummy_input,  # Model input
                    onnx_path,  # Save path
                    export_params=True,  # Whether to export model parameters
                    opset_version=11,  # ONNX opset version
                    do_constant_folding=True,  # Whether to perform constant folding optimization
                    input_names=['input'],  # Input node name
                    output_names=['output'],  # Output node name
                    dynamic_axes={'input': {0: 'batch_size'},  # Dynamic batch size
                                    'output': {0: 'batch_size'}})

    print(f"ONNX model has been exported to: {onnx_path}")

if __name__ == '__main__':
    export_model(10, './pred/checkpoints/long_term.pt', 'long_term.onnx')
    export_model(12, './pred/checkpoints/short_term.pt', 'short_term.onnx')
