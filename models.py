import segmentation_models_pytorch as smp


MODEL_ARCHITECTURES = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
}


def build_model(
    architecture,
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    **kwargs,
):
    if architecture not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. "
            f"Choose from: {list(MODEL_ARCHITECTURES)}"
        )

    return MODEL_ARCHITECTURES[architecture](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
