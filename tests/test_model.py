import torch

from src.model import build_model


def test_forward_shape():
    model = build_model(num_classes=2, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)


def test_num_classes():
    model = build_model(num_classes=3, pretrained=False)
    assert model.fc.out_features == 3
