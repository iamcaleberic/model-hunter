import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoModel, AutoVideoProcessor

processor = AutoVideoProcessor.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256"
)
model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    device_map="auto",
    attn_implementation="sdpa"
)

video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

vr = VideoDecoder(video_url)

# choosing some frames. here, you can define more complex sampling strategy
frame_idx = np.arange(0, 64)
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
video = processor(video, return_tensors="pt").to(model.device)
outputs = model(**video)

# V-JEPA 2 encoder outputs, same as calling `model.get_vision_features()`
encoder_outputs = outputs.last_hidden_state

# V-JEPA 2 predictor outputs
predictor_outputs = outputs.predictor_output.last_hidden_state
