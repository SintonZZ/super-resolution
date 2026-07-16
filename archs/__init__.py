from .spanf import SPANF


def build_model(model_config):
    model_type = model_config.get("type", "spanf")
    if model_type != "spanf":
        raise ValueError(f"Unsupported model type: {model_type}")

    return SPANF(
        num_in_ch=int(model_config.get("in_channels", 3)),
        num_out_ch=int(model_config.get("out_channels", 3)),
        feature_channels=int(model_config.get("feature_channels", 32)),
        upscale=int(model_config.get("upscale", 2)),
        bias=bool(model_config.get("bias", True)),
        nearest_init=bool(model_config.get("nearest_init", True)),
    )


__all__ = ["SPANF", "build_model"]
