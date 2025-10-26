from sindhi_ocr import SindhiOCR
import pytesseract

# Set Tesseract CMD path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize OCR
ocr = SindhiOCR(engine="tesseract")  # or "easyocr", "trocr", "ensemble"

# Process all word images
results = ocr.process_directory(
    input_dir="output_sindhi/words",
    output_file="output_sindhi/ocr_results.json"
)

# Reconstruct full text (line by line)
full_text = ocr.reconstruct_text(
    output_file="output_sindhi/recognized_text.txt"
)

# Create visualization with annotations
ocr.create_annotated_visualization(
    output_image="output_sindhi/ocr_visualization.png"
)