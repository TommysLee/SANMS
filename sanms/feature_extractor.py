"""
Feature extraction for image retrieval.

Uses ResNet-50 (ImageNet-pretrained, optionally fine-tuned with lifted
structured loss) to produce 512-dimensional L2-normalized embeddings.
"""

import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FeatureExtractor:
    """ResNet-50 feature extractor for retrieval.

    Produces 512-dim L2-normalized embeddings from cropped image regions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        embedding_dim: int = 512,
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.model = None
        self.transform = None

        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. FeatureExtractor will use random embeddings.")
            return

        # Load ResNet-50 and replace final FC for 512-dim embedding
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove final classification layer
        modules = list(backbone.children())[:-1]  # keep up to avgpool
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, embedding_dim)

        # Load fine-tuned weights if provided
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=device)
            if "model_state" in checkpoint:
                self.backbone.load_state_dict(
                    {k.replace("backbone.", ""): v for k, v in checkpoint["model_state"].items()
                     if k.startswith("backbone.")}
                )
                self.fc.load_state_dict(
                    {k.replace("fc.", ""): v for k, v in checkpoint["model_state"].items()
                     if k.startswith("fc.")}
                )
            else:
                self.backbone.load_state_dict(checkpoint, strict=False)

        self.backbone = self.backbone.to(device).eval()
        self.fc = self.fc.to(device).eval()

        # Image preprocessing (standard for ResNet)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image: "Image.Image") -> np.ndarray:
        """Extract feature from a PIL Image.

        Args:
            image: PIL Image (already cropped to the detection box).

        Returns:
            (512,) L2-normalized embedding.
        """
        if not TORCH_AVAILABLE:
            return np.random.randn(self.embedding_dim).astype(np.float32)

        with torch.no_grad():
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            features = self.backbone(img_tensor)  # (1, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)  # (1, 2048)
            embedding = self.fc(features)  # (1, 512)

            # L2 normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            return embedding.cpu().numpy().squeeze()

    def extract_batch(self, images: list) -> np.ndarray:
        """Extract features for a batch of images.

        Args:
            images: list of PIL Images.

        Returns:
            (N, 512) L2-normalized embeddings.
        """
        if not TORCH_AVAILABLE or len(images) == 0:
            return np.random.randn(len(images), self.embedding_dim).astype(np.float32)

        with torch.no_grad():
            tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
            features = self.backbone(tensors)  # (N, 2048, 1, 1)
            features = features.squeeze(-1).squeeze(-1)
            embeddings = self.fc(features)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy()


def crop_image(image: "Image.Image", box: np.ndarray) -> "Image.Image":
    """Crop an image to the given bounding box.

    Args:
        image: PIL Image.
        box: (4,) [x1, y1, x2, y2] in pixel coordinates.

    Returns:
        Cropped PIL Image.
    """
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)
    return image.crop((x1, y1, x2, y2))


def extract_top_box_feature(
    image: "Image.Image",
    boxes: np.ndarray,
    scores: np.ndarray,
    extractor: FeatureExtractor,
) -> np.ndarray:
    """Extract feature from the top-scoring detection box.

    This is the standard retrieval pipeline: detect -> NMS -> SANMS ->
    select top box -> crop -> feature extraction.

    Args:
        image: PIL Image.
        boxes: (N, 4) detection boxes (already NMS+SANMS filtered).
        scores: (N,) confidence scores (sorted descending).
        extractor: FeatureExtractor instance.

    Returns:
        (D,) L2-normalized embedding.
    """
    if len(boxes) == 0:
        # Fallback: use full image
        return extractor.extract(image)

    # Top-scoring box
    top_box = boxes[0]  # already sorted by confidence
    crop = crop_image(image, top_box)
    return extractor.extract(crop)
