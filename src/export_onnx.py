import torch
import torch.nn as nn
import onnx
from pathlib import Path
from torchvision.models import mobilenet_v3_small


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

CHECKPOINT_PATH = MODEL_DIR / "mobilenet_v3_small_best.pth"
ONNX_PATH = MODEL_DIR / "vehicle_classifier.onnx"
CLASSES_PATH = OUTPUT_DIR / "classes.txt"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_SIZE = 224
NUM_CLASSES = 12

device = torch.device("cpu")


# --------------------------------------------------
# Load class names
# --------------------------------------------------

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]

if len(classes) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, found {len(classes)}"
    )


# --------------------------------------------------
# Build MobileNetV3-Small
# --------------------------------------------------

model = mobilenet_v3_small(weights=None)

in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    in_features,
    NUM_CLASSES
)


# --------------------------------------------------
# Load trained checkpoint
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("MobileNetV3-Small checkpoint loaded.")
print("Classes:", classes)
print("Input size: 3 x 224 x 224")


# --------------------------------------------------
# Dummy input
# --------------------------------------------------

dummy_input = torch.randn(
    1,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
    device=device
)


# --------------------------------------------------
# Export ONNX
# --------------------------------------------------

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    dynamo=False
)

print("\nONNX model exported to:")
print(ONNX_PATH)

print(
    f"File size: "
    f"{ONNX_PATH.stat().st_size / (1024 ** 2):.2f} MB"
)


# --------------------------------------------------
# Validate ONNX model
# --------------------------------------------------

onnx_model = onnx.load(str(ONNX_PATH))

onnx.checker.check_model(onnx_model)

print("\nONNX model is valid.")
print("IR version:", onnx_model.ir_version)
print(
    "Opset version:",
    onnx_model.opset_import[0].version
)