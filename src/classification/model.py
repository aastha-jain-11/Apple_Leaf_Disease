import torch
import torch.nn as nn
import timm


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "convnextv2_base.fcmae_ft_in1k"

NUM_CLASSES = 4


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    num_classes=NUM_CLASSES,
    pretrained=True
):

    print("=" * 70)
    print("CREATING CONVNEXT V2 MODEL")
    print("=" * 70)

    print()
    print("Model:")
    print(MODEL_NAME)

    print()
    print("Pretrained:")
    print(pretrained)

    print()
    print("Number of classes:")
    print(num_classes)

    # --------------------------------------------------------
    # Load pretrained ConvNeXt V2
    # --------------------------------------------------------

    model = timm.create_model(
        MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model


# ============================================================
# TEST MODEL DIRECTLY
# ============================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = create_model()

    model = model.to(device)

    print()
    print("Model successfully created.")

    print()
    print("Device:")
    print(device)

    print()

    print("Testing forward pass...")

    dummy_input = torch.randn(
        2,
        3,
        224,
        224,
        device=device
    )

    with torch.no_grad():

        output = model(
            dummy_input
        )

    print()
    print("Input shape:")
    print(dummy_input.shape)

    print()
    print("Output shape:")
    print(output.shape)

    print()
    print("Output:")
    print(output)

    print()

    if output.shape == (2, NUM_CLASSES):

        print("=" * 70)
        print("CONVNEXT V2 MODEL TEST PASSED")
        print("=" * 70)

    else:

        raise RuntimeError(
            f"Unexpected output shape: "
            f"{output.shape}"
        )