from segmenter import SindhiTextSegmenter

# Process single image WITH automatic skew correction
segmenter = SindhiTextSegmenter(
    image_path="input_images/image1.png",
    output_dir="output_sindhi",
    line_padding=5,         # Extra space for diacritics above/below
    word_padding=5,         # Extra space for connected ligatures
    letter_padding=7,       # Extra space around dots
    output_height=64,       # Resize for ML (64, 128, or 256)
    denoise_strength=12,    # Adjust for image quality
    auto_deskew=True,       # 🆕 Enable automatic skew correction
    skew_threshold=0.5      # 🆕 Only correct if angle > 0.5°
)

# For word recognition (recommended)
segmenter.process(segment_level="words")

# For character recognition (separate call)
# segmenter.process(segment_level="ligatures")