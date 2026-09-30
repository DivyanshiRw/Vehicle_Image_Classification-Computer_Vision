import argparse
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as rt
import torch
from PIL import Image
from torchvision import transforms


class LetterboxTransform:
    """
    Resize an image while preserving aspect ratio,
    then pad it to a square image.
    """

    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = image.convert("RGB")

        scale = min(
            self.size / image.width,
            self.size / image.height
        )

        new_width = max(1, int(image.width * scale))
        new_height = max(1, int(image.height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            (0, 0, 0)
        )

        x = (self.size - new_width) // 2
        y = (self.size - new_height) // 2

        canvas.paste(image, (x, y))

        return canvas


def get_transform(image_size=224):
    return transforms.Compose([
        LetterboxTransform(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def main(model_path, classes_path, image_size):

    # --------------------------------------------------
    # Load ONNX model
    # --------------------------------------------------
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    sess = rt.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    # --------------------------------------------------
    # Model input/output information
    # --------------------------------------------------
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    output_name = sess.get_outputs()[0].name
    output_shape = sess.get_outputs()[0].shape

    print("ONNX model loaded successfully.")
    print("Input name:", input_name)
    print("Input shape:", input_shape)
    print("Output name:", output_name)
    print("Output shape:", output_shape)

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------
    transform = get_transform(image_size)

    # --------------------------------------------------
    # Load class names
    # --------------------------------------------------
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    print("Number of classes:", len(classes))
    print("Classes:", classes)

    # --------------------------------------------------
    # Verification images
    # --------------------------------------------------
    project_dir = Path(__file__).resolve().parent.parent

    verification_dir = project_dir / "verification_images"

    image_names = [
        "autorickshaw.png",
        "car.png",
        "van.png"
    ]

    # --------------------------------------------------
    # Output directories
    # --------------------------------------------------
    runs_dir = project_dir / "runs"
    pred_dir = runs_dir / "pred"

    pred_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------
    for image_name in image_names:

        image_path = verification_dir / image_name

        if not image_path.exists():
            print(f"\nImage not found: {image_path}")
            continue

        # Read image using PIL
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        image_tensor = transform(image)

        # Add batch dimension
        input_array = image_tensor.unsqueeze(0).numpy()

        # ONNX inference
        outputs = sess.run(
            [output_name],
            {input_name: input_array}
        )[0]

        # Get prediction
        prediction = outputs[0]

        pred_index = int(np.argmax(prediction))
        pred_class = classes[pred_index]

        # Convert logits to probabilities
        probabilities = torch.softmax(
            torch.tensor(prediction),
            dim=0
        ).numpy()

        confidence = float(probabilities[pred_index])

        print("\n----------------------------------------")
        print("Image:", image_name)
        print("Predicted class:", pred_class)
        print(f"Confidence: {confidence:.4f}")
        print("----------------------------------------")

        # --------------------------------------------------
        # Save annotated image
        # --------------------------------------------------
        org_image = cv2.imread(str(image_path))

        text = f"{pred_class} ({confidence:.2f})"

        cv2.putText(
            org_image,
            text,
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        save_path = pred_dir / f"pred_{image_name}"

        cv2.imwrite(
            str(save_path),
            org_image
        )

        print("Saved prediction to:", save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Verify vehicle classification using ONNX Runtime"
    )

    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Path to ONNX model"
    )

    parser.add_argument(
        "-c",
        "--classes_path",
        required=True,
        help="Path to classes.txt"
    )

    parser.add_argument(
        "-s",
        "--image_size",
        type=int,
        default=224,
        help="Input image size"
    )

    args = parser.parse_args()

    main(
        args.model_path,
        args.classes_path,
        args.image_size
    )