from segmenter import SindhiTextSegmenter

segmenter = SindhiTextSegmenter(
    image_path="input_images/image1.png",
    output_dir="output_sindhi",
    line_padding=5,
    word_padding=5,
    letter_padding=7,
    output_height=64,
    denoise_strength=12,
    auto_deskew=True,
    skew_threshold=0.5,
    min_text_pixels=50,      # For lines and words
    text_threshold=0.05      # For lines and words
)

# Now this will generate ligatures!
segmenter.process(segment_level="ligatures")