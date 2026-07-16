from .span import SPAN


def build_span(model_config):
    model_type = model_config.get("type", "span")
    if model_type != "span":
        raise ValueError(f"Unsupported model type: {model_type}")

    return SPAN(
        num_in_ch=int(model_config.get("in_channels", 3)),
        num_out_ch=int(model_config.get("out_channels", 3)),
        feature_channels=int(model_config.get("feature_channels", 48)),
        upscale=int(model_config.get("upscale", 2)),
        bias=bool(model_config.get("bias", True)),
        img_range=float(model_config.get("img_range", 255.0)),
        rgb_mean=tuple(model_config.get("rgb_mean", (0.4488, 0.4371, 0.4040))),
    )


__all__ = ["SPAN", "build_span"]
