import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️  easyocr not installed. Install with: pip install easyocr")

    
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  pytesseract not installed. Install with: pip install pytesseract")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("⚠️  transformers not installed. Install with: pip install transformers pillow torch")




class SindhiOCR:
    """
    Sindhi OCR system with multiple engine support
    Converts segmented word images to Sindhi text
    """
    
    def __init__(self, engine="tesseract", model_path=None):
        """
        Initialize OCR engine
        
        Args:
            engine: "tesseract", "easyocr", "trocr", or "ensemble"
            model_path: Path to custom trained model (optional)
        """
        self.engine = engine
        self.model_path = model_path
        self.results = []
        
        # Initialize selected engine
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the selected OCR engine"""
        print(f"\n{'='*60}")
        print(f"Initializing Sindhi OCR Engine: {self.engine}")
        print(f"{'='*60}\n")
        
        if self.engine == "tesseract":
            if not TESSERACT_AVAILABLE:
                raise ImportError("pytesseract not installed")
            self._init_tesseract()
            
        elif self.engine == "easyocr":
            if not EASYOCR_AVAILABLE:
                raise ImportError("easyocr not installed")
            self._init_easyocr()
            
        elif self.engine == "trocr":
            if not TROCR_AVAILABLE:
                raise ImportError("transformers not installed")
            self._init_trocr()
            
        elif self.engine == "ensemble":
            print("Ensemble mode: Will use all available engines")
            
        else:
            raise ValueError(f"Unknown engine: {self.engine}")
    
    def _init_tesseract(self):
        """Initialize Tesseract OCR"""
        try:
            # Check if Tesseract is installed
            pytesseract.get_tesseract_version()
            print("✓ Tesseract OCR initialized")
            
            # Check for Sindhi language support
            langs = pytesseract.get_languages()
            if 'sd' in langs or 'sin' in langs or 'sindhi' in langs:
                print("✓ Sindhi language pack detected")
                self.tesseract_lang = 'sd'
            elif 'ara' in langs or 'Arabic' in langs:
                print("⚠️  Sindhi not found, using Arabic (similar script)")
                self.tesseract_lang = 'ara'
            else:
                print("⚠️  No suitable language pack found")
                print("   Install Sindhi: sudo apt-get install tesseract-ocr-sin")
                self.tesseract_lang = 'eng'
                
        except Exception as e:
            print(f"❌ Tesseract initialization failed: {e}")
            raise
    
    def _init_easyocr(self):
        """Initialize EasyOCR"""
        try:
            # EasyOCR supports Urdu which is very similar to Sindhi
            print("Loading EasyOCR (Urdu/Arabic script)...")
            self.easyocr_reader = easyocr.Reader(['ur', 'ar'], gpu=True)
            print("✓ EasyOCR initialized")
        except Exception as e:
            print(f"❌ EasyOCR initialization failed: {e}")
            raise
    
    def _init_trocr(self):
        """Initialize TrOCR transformer model"""
        try:
            print("Loading TrOCR model (multilingual)...")
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-printed"
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-printed"
            )
            print("✓ TrOCR initialized")
        except Exception as e:
            print(f"❌ TrOCR initialization failed: {e}")
            raise
    
    def preprocess_for_ocr(self, image):
        """
        Preprocess image for better OCR accuracy
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Binarization
        _, binary = cv2.threshold(enhanced, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def recognize_with_tesseract(self, image):
        """Use Tesseract OCR"""
        try:
            # Preprocess
            processed = self.preprocess_for_ocr(image)
            
            # Configure Tesseract for Sindhi/Arabic script
            custom_config = r'--oem 3 --psm 7'  # PSM 7: single text line
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed,
                lang=self.tesseract_lang,
                config=custom_config
            ).strip()
            
            # Get confidence
            data = pytesseract.image_to_data(
                processed,
                lang=self.tesseract_lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return text, avg_confidence
            
        except Exception as e:
            print(f"Tesseract error: {e}")
            return "", 0
    
    def recognize_with_easyocr(self, image):
        """Use EasyOCR"""
        try:
            # EasyOCR expects RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Perform OCR
            results = self.easyocr_reader.readtext(image_rgb, detail=1)
            
            if results:
                # Combine all detected text
                texts = [result[1] for result in results]
                confidences = [result[2] for result in results]
                
                text = " ".join(texts)
                avg_confidence = np.mean(confidences) * 100 if confidences else 0
                
                return text, avg_confidence
            else:
                return "", 0
                
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return "", 0
    
    def recognize_with_trocr(self, image):
        """Use TrOCR transformer model"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(image_rgb)
            
            # Process with TrOCR
            pixel_values = self.trocr_processor(
                pil_image, 
                return_tensors="pt"
            ).pixel_values
            
            generated_ids = self.trocr_model.generate(pixel_values)
            text = self.trocr_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # TrOCR doesn't provide confidence, use fixed value
            confidence = 75.0
            
            return text, confidence
            
        except Exception as e:
            print(f"TrOCR error: {e}")
            return "", 0
    
    def recognize_image(self, image_path):
        """
        Recognize text from a single image
        
        Args:
            image_path: Path to word image
            
        Returns:
            dict with text, confidence, and metadata
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        # Select recognition method
        if self.engine == "tesseract":
            text, confidence = self.recognize_with_tesseract(image)
            engine_used = "tesseract"
            
        elif self.engine == "easyocr":
            text, confidence = self.recognize_with_easyocr(image)
            engine_used = "easyocr"
            
        elif self.engine == "trocr":
            text, confidence = self.recognize_with_trocr(image)
            engine_used = "trocr"
            
        elif self.engine == "ensemble":
            # Use all available engines and combine
            results = []
            
            if TESSERACT_AVAILABLE:
                t_text, t_conf = self.recognize_with_tesseract(image)
                if t_text:
                    results.append(("tesseract", t_text, t_conf))
            
            if EASYOCR_AVAILABLE:
                e_text, e_conf = self.recognize_with_easyocr(image)
                if e_text:
                    results.append(("easyocr", e_text, e_conf))
            
            if TROCR_AVAILABLE:
                tr_text, tr_conf = self.recognize_with_trocr(image)
                if tr_text:
                    results.append(("trocr", tr_text, tr_conf))
            
            if results:
                # Use result with highest confidence
                engine_used, text, confidence = max(results, key=lambda x: x[2])
            else:
                text, confidence, engine_used = "", 0, "none"
        
        else:
            text, confidence, engine_used = "", 0, "unknown"
        
        result = {
            "image_path": str(image_path),
            "text": text,
            "confidence": float(confidence),
            "engine": engine_used,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def process_directory(self, input_dir, output_file=None, 
                         pattern="*.png", show_progress=True):
        """
        Process all word images in a directory
        
        Args:
            input_dir: Directory containing word images
            output_file: JSON file to save results (optional)
            pattern: File pattern to match
            show_progress: Print progress
            
        Returns:
            List of recognition results
        """
        input_path = Path(input_dir)
        image_files = sorted(input_path.glob(pattern))
        
        print(f"\n📁 Processing {len(image_files)} images from: {input_dir}\n")
        
        results = []
        
        for idx, img_file in enumerate(image_files, 1):
            if show_progress and idx % 10 == 0:
                print(f"   Processed {idx}/{len(image_files)} images...")
            
            result = self.recognize_image(img_file)
            if result:
                results.append(result)
        
        print(f"\n✓ Completed: {len(results)} images processed")
        
        # Calculate statistics
        successful = sum(1 for r in results if r['text'])
        avg_confidence = np.mean([r['confidence'] for r in results if r['text']])
        
        print(f"   Successful recognitions: {successful}/{len(results)}")
        print(f"   Average confidence: {avg_confidence:.2f}%")
        
        # Save results if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        self.results = results
        return results
    
    def save_results(self, results, output_file):
        """Save OCR results to JSON file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def reconstruct_text(self, results=None, output_file=None):
        """
        Reconstruct full text from word-level results
        Groups words by line number if available in filename
        
        Args:
            results: OCR results (uses self.results if None)
            output_file: Text file to save output (optional)
            
        Returns:
            Reconstructed text string
        """
        if results is None:
            results = self.results
        
        if not results:
            print("No results to reconstruct")
            return ""
        
        # Group by line if filenames contain line info
        lines = {}
        
        for result in results:
            text = result['text']
            if not text:
                continue
            
            # Extract line number from filename (e.g., word_0000_line0.png)
            filename = Path(result['image_path']).stem
            
            if 'line' in filename:
                try:
                    line_num = int(filename.split('line')[1])
                except:
                    line_num = 0
            else:
                line_num = 0
            
            if line_num not in lines:
                lines[line_num] = []
            
            lines[line_num].append(text)
        
        # Reconstruct text line by line
        reconstructed = []
        for line_num in sorted(lines.keys()):
            # Sindhi is RTL, but words are already in correct order
            line_text = " ".join(lines[line_num])
            reconstructed.append(line_text)
        
        full_text = "\n".join(reconstructed)
        
        print(f"\n📝 Reconstructed Text ({len(reconstructed)} lines):")
        print("="*60)
        print(full_text)
        print("="*60)
        
        # Save if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            print(f"\n💾 Text saved to: {output_file}")
        
        return full_text
    
    def create_annotated_visualization(self, results=None, output_image=None):
        """
        Create visualization with original images and recognized text
        
        Args:
            results: OCR results (uses self.results if None)
            output_image: Path to save visualization
        """
        if results is None:
            results = self.results
        
        if not results:
            print("No results to visualize")
            return
        
        # Create visualization
        max_width = 800
        row_height = 100
        padding = 10
        
        total_height = len(results) * (row_height + padding)
        canvas = np.ones((total_height, max_width, 3), dtype=np.uint8) * 255
        
        y_offset = padding
        
        for result in results:
            # Load image
            img = cv2.imread(result['image_path'])
            if img is None:
                continue
            
            # Resize to fit
            h, w = img.shape[:2]
            scale = min((row_height - 20) / h, 300 / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h))
            
            # Place image
            canvas[y_offset:y_offset+new_h, padding:padding+new_w] = img_resized
            
            # Add text
            text = result['text'] if result['text'] else "[empty]"
            confidence = result['confidence']
            label = f"{text} ({confidence:.1f}%)"
            
            cv2.putText(canvas, label, (padding + new_w + 20, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            y_offset += row_height + padding
        
        if output_image:
            cv2.imwrite(str(output_image), canvas)
            print(f"\n🖼️  Visualization saved to: {output_image}")
        
        return canvas


# Example usage and utilities
def main():
    """Example usage of Sindhi OCR"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           Sindhi OCR - Words to Text Converter            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Method 1: Using Tesseract (recommended if available)
    print("\n📌 Method 1: Tesseract OCR")
    print("-" * 60)
    
    try:
        ocr = SindhiOCR(engine="tesseract")
        
        # Process directory of word images
        results = ocr.process_directory(
            input_dir="output_sindhi/words",
            output_file="output_sindhi/ocr_results.json",
            show_progress=True
        )
        
        # Reconstruct full text
        full_text = ocr.reconstruct_text(
            output_file="output_sindhi/recognized_text.txt"
        )
        
        # Create visualization
        ocr.create_annotated_visualization(
            output_image="output_sindhi/ocr_visualization.png"
        )
        
    except Exception as e:
        print(f"Tesseract method failed: {e}")
    
    # Method 2: Using EasyOCR (more accurate for Urdu/Arabic script)
    print("\n📌 Method 2: EasyOCR")
    print("-" * 60)
    
    try:
        ocr = SindhiOCR(engine="easyocr")
        results = ocr.process_directory(
            input_dir="output_sindhi/words",
            output_file="output_sindhi/ocr_results_easyocr.json"
        )
        ocr.reconstruct_text(
            output_file="output_sindhi/recognized_text_easyocr.txt"
        )
    except Exception as e:
        print(f"EasyOCR method failed: {e}")
    
    # Method 3: Ensemble (uses all available engines)
    print("\n📌 Method 3: Ensemble (Best Results)")
    print("-" * 60)
    
    try:
        ocr = SindhiOCR(engine="ensemble")
        results = ocr.process_directory(
            input_dir="output_sindhi/words",
            output_file="output_sindhi/ocr_results_ensemble.json"
        )
        ocr.reconstruct_text(
            output_file="output_sindhi/recognized_text_ensemble.txt"
        )
    except Exception as e:
        print(f"Ensemble method failed: {e}")


if __name__ == "__main__":
    main()