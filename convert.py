
import torch
import torch.nn as nn
from pathlib import Path

from train import TTTNet 

MODEL_PATH = Path("trailing_ttt_nn.pt")
OUT_ONNX   = Path("trailing_ttt_nn.onnx")
INPUT_SHAPE = (1, 36)
OPSET = 17


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Cannot find model weights: {MODEL_PATH}")

    print(f"Loading trained CNN from: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location="cpu")

    model = TTTNet()
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    class PolicyOnlyWrapper(nn.Module):
        def __init__(self, base_model: nn.Module):
            super().__init__()
            self.base = base_model

        def forward(self, x):
            logits, _ = self.base(x)
            return logits

    wrapped = PolicyOnlyWrapper(model)

    dummy = torch.randn(*INPUT_SHAPE, dtype=torch.float32)
    print(f"Exporting to ONNX: {OUT_ONNX}  (input={INPUT_SHAPE}, opset={OPSET})")

    torch.onnx.export(
        wrapped,
        dummy,
        str(OUT_ONNX),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        # No dynamic_axes needed for our use case
    )

    print("✅ ONNX saved →", OUT_ONNX)


if __name__ == "__main__":
    main()
