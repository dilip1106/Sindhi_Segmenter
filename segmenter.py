import cv2
import numpy as np
import os
from pathlib import Path

class SindhiTextSegmenter:
    """
    Advanced Sindhi Text Segmenter with automatic skew correction
    Designed for Sindhi script characteristics:
    - 52 characters (largest extension of Arabic alphabet)
    - 24 differentiating characters with up to 4 dots
    - Cursive and context-sensitive
    - Right-to-left writing direction
    - Connected ligatures
    - Diacritical marks above and below
    """
    
    def __init__(self, image_path, output_dir="segmented_output",
                 line_padding=20, word_padding=10, letter_padding=7,
                 output_height=64, output_width=None,
                 denoise_strength=10, morph_kernel_size=2,
                 auto_deskew=True, skew_threshold=0.5,
                 min_text_pixels=50, text_threshold=0.05):
        """
        Initialize the segmenter optimized for Sindhi script
        
        Args:
            image_path: Path to the scanned Sindhi text image
            output_dir: Directory to save segmented images
            line_padding: Extra padding for lines (captures diacritics above/below)
            word_padding: Extra padding for words (handles connected ligatures)
            letter_padding: Extra padding for letters
            output_height: Height to resize images (64-128 recommended for ML)
            output_width: Width to resize (None = maintain aspect ratio)
            denoise_strength: Noise reduction strength (10-15 for scanned docs)
            morph_kernel_size: Morphological operation kernel size
            auto_deskew: Automatically detect and correct skew
            skew_threshold: Minimum skew angle to correct (in degrees)
            min_text_pixels: Minimum number of text pixels to consider valid
            text_threshold: Minimum ratio of text pixels to total pixels (0.01-0.1)
        """
        self.image_path = image_path
        self.output_dir = output_dir
        self.line_padding = line_padding
        self.word_padding = word_padding
        self.letter_padding = letter_padding
        self.output_height = output_height
        self.output_width = output_width
        self.denoise_strength = denoise_strength
        self.morph_kernel_size = morph_kernel_size
        self.auto_deskew = auto_deskew
        self.skew_threshold = skew_threshold
        self.min_text_pixels = min_text_pixels
        self.text_threshold = text_threshold
        
        self.image = None
        self.gray = None
        self.binary = None
        self.processed = None
        self.skew_angle = 0
        
        # Create output directories
        self._create_directories()
    
    def has_sufficient_text(self, image, binary_image=None):
        """
        Check if image has sufficient text content
        Returns True if image contains meaningful text, False if empty/background
        
        Args:
            image: Color or grayscale image
            binary_image: Pre-computed binary image (optional)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use provided binary or create one
        if binary_image is None:
            # Quick thresholding for checking
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            binary = binary_image
        
        # Count text pixels (white pixels in binary)
        text_pixels = np.sum(binary > 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        
        # Calculate text ratio
        text_ratio = text_pixels / total_pixels if total_pixels > 0 else 0
        
        # Check both absolute count and ratio
        has_text = (text_pixels >= self.min_text_pixels and 
                   text_ratio >= self.text_threshold)
        
        return has_text, text_pixels, text_ratio
        
    def _create_directories(self):
        """Create output directory structure"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/lines").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/words").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/ligatures").mkdir(exist_ok=True)
        Path(f"{self.output_dir}/debug").mkdir(exist_ok=True)
    
    def detect_skew_angle(self, image):
        """
        Detect skew angle using multiple methods for robustness
        Returns angle in degrees (positive = clockwise rotation needed)
        """
        # Method 1: Hough Line Transform (most reliable for text)
        angle1 = self._detect_skew_hough(image)
        
        # Method 2: Projection Profile (backup method)
        angle2 = self._detect_skew_projection(image)
        
        # Use Hough if confident, otherwise use projection
        if angle1 is not None and abs(angle1) < 45:
            final_angle = angle1
            method = "Hough Transform"
        elif angle2 is not None:
            final_angle = angle2
            method = "Projection Profile"
        else:
            final_angle = 0
            method = "No skew detected"
        
        print(f"   Skew detection method: {method}")
        print(f"   Detected skew angle: {final_angle:.2f}°")
        
        return final_angle
    
    def _detect_skew_hough(self, image):
        """Detect skew using Hough Line Transform"""
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Hough Line Transform
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=100,
                minLineLength=image.shape[1] // 4,  # At least 1/4 of image width
                maxLineGap=20
            )
            
            if lines is None or len(lines) < 5:
                return None
            
            # Calculate angles of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Normalize angle to [-45, 45] range
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                
                # Filter out near-vertical lines (not text lines)
                if abs(angle) < 45:
                    angles.append(angle)
            
            if not angles:
                return None
            
            # Use median angle (more robust than mean)
            median_angle = np.median(angles)
            
            return median_angle
            
        except Exception as e:
            print(f"   Hough method failed: {e}")
            return None
    
    def _detect_skew_projection(self, image):
        """Detect skew using projection profile method"""
        try:
            # Try angles from -45 to 45 degrees
            angles_to_test = np.arange(-45, 46, 0.5)
            scores = []
            
            h, w = image.shape
            center = (w // 2, h // 2)
            
            for angle in angles_to_test:
                # Rotate image
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
                
                # Calculate horizontal projection
                projection = np.sum(rotated, axis=1)
                
                # Score is variance of projection (peaks = good alignment)
                score = np.var(projection)
                scores.append(score)
            
            # Find angle with maximum variance
            best_idx = np.argmax(scores)
            best_angle = angles_to_test[best_idx]
            
            return best_angle
            
        except Exception as e:
            print(f"   Projection method failed: {e}")
            return None
    
    def deskew_image(self, image, angle):
        """
        Rotate image to correct skew
        Positive angle = clockwise rotation
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image size to avoid cropping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new size
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform rotation
        if len(image.shape) == 3:
            deskewed = cv2.warpAffine(
                image, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
        else:
            deskewed = cv2.warpAffine(
                image, M, (new_w, new_h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
        
        return deskewed
        
    def load_and_preprocess(self, save_debug=True):
        """
        Advanced preprocessing pipeline for Sindhi text
        Includes automatic skew detection and correction
        """
        # Read image
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        print(f"Image loaded: {self.image.shape}")
        
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Auto-deskew if enabled
        if self.auto_deskew:
            print("\n🔍 Detecting skew...")
            
            # Detect skew on grayscale image
            self.skew_angle = self.detect_skew_angle(self.gray)
            
            # Correct skew if above threshold
            if abs(self.skew_angle) > self.skew_threshold:
                print(f"✓ Correcting skew: {self.skew_angle:.2f}°")
                
                # Deskew both color and grayscale images
                self.image = self.deskew_image(self.image, self.skew_angle)
                self.gray = self.deskew_image(self.gray, self.skew_angle)
                
                if save_debug:
                    cv2.imwrite(f"{self.output_dir}/debug/00_deskewed_color.png", 
                               self.image)
                    cv2.imwrite(f"{self.output_dir}/debug/00_deskewed_gray.png", 
                               self.gray)
            else:
                print(f"✓ No significant skew detected ({self.skew_angle:.2f}°)")
        
        # Denoise - important for Sindhi dots (4 dots on some characters)
        denoised = cv2.fastNlMeansDenoising(
            self.gray, None, 
            self.denoise_strength, 7, 21
        )
        
        # Adaptive thresholding - better for varying lighting conditions
        self.binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            15, 4  # Adjusted for Sindhi script
        )
        
        # Morphological closing to connect broken characters
        # Important for cursive Sindhi script
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        self.processed = cv2.morphologyEx(self.binary, cv2.MORPH_CLOSE, kernel)
        
        # Save debug images
        if save_debug:
            cv2.imwrite(f"{self.output_dir}/debug/01_grayscale.png", self.gray)
            cv2.imwrite(f"{self.output_dir}/debug/02_denoised.png", denoised)
            cv2.imwrite(f"{self.output_dir}/debug/03_binary.png", self.binary)
            cv2.imwrite(f"{self.output_dir}/debug/04_processed.png", self.processed)
        
        return self.processed
    
    def resize_image(self, img):
        """Resize image while maintaining aspect ratio"""
        if self.output_height is None:
            return img
        
        h, w = img.shape[:2]
        
        if self.output_width is None:
            # Maintain aspect ratio
            aspect_ratio = w / h
            new_width = int(self.output_height * aspect_ratio)
            resized = cv2.resize(img, (new_width, self.output_height), 
                               interpolation=cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(img, (self.output_width, self.output_height), 
                               interpolation=cv2.INTER_CUBIC)
        
        return resized
    
    def segment_lines(self, min_line_height=10):
        """
        Segment into text lines
        Handles non-uniform line heights common in Sindhi
        """
        # Horizontal projection
        horizontal_projection = np.sum(self.processed, axis=1)
        
        # Dynamic threshold based on image statistics
        threshold = np.mean(horizontal_projection) * 0.3
        
        lines = []
        in_line = False
        start = 0
        
        for i, val in enumerate(horizontal_projection):
            if val > threshold and not in_line:
                start = i
                in_line = True
            elif val <= threshold and in_line:
                line_height = i - start
                if line_height > min_line_height:  # Filter noise
                    # Generous padding for diacritics
                    lines.append((
                        max(0, start - self.line_padding),
                        min(len(horizontal_projection), i + self.line_padding)
                    ))
                in_line = False
        
        # Handle last line
        if in_line and (len(horizontal_projection) - start) > min_line_height:
            lines.append((max(0, start - self.line_padding), 
                         len(horizontal_projection)))
        
        # Save line images
        line_images = []
        saved_lines = 0
        skipped_lines = 0
        
        for idx, (start, end) in enumerate(lines):
            line_img = self.image[start:end, :]
            line_binary = self.processed[start:end, :]
            
            # Check if line has sufficient text
            has_text, text_pixels, text_ratio = self.has_sufficient_text(
                line_img, line_binary
            )
            
            if not has_text:
                skipped_lines += 1
                continue
            
            # Resize if specified
            line_img_resized = self.resize_image(line_img)
            
            cv2.imwrite(f"{self.output_dir}/lines/line_{saved_lines:03d}.png", 
                       line_img_resized)
            line_images.append((line_binary, line_img, saved_lines))
            saved_lines += 1
        
        print(f"✓ Segmented {saved_lines} lines (skipped {skipped_lines} empty)")
        return line_images
    
    def segment_words_or_ligatures(self, line_images, min_width=5):
        """
        Segment lines into words/ligatures
        Sindhi uses connected ligatures - this handles them properly
        """
        all_words = []
        word_count = 0
        skipped_words = 0
        
        for line_binary, line_color, line_idx in line_images:
            # Vertical projection
            vertical_projection = np.sum(line_binary, axis=0)
            
            # Dynamic threshold
            threshold = np.mean(vertical_projection) * 0.2
            
            words = []
            in_word = False
            start = 0
            gap_counter = 0
            gap_threshold = 3  # Pixels of gap to separate words
            
            for i, val in enumerate(vertical_projection):
                if val > threshold:
                    if not in_word:
                        start = i
                        in_word = True
                    gap_counter = 0
                else:
                    if in_word:
                        gap_counter += 1
                        if gap_counter > gap_threshold:
                            # Found word boundary
                            word_width = i - gap_counter - start
                            if word_width > min_width:
                                words.append((
                                    max(0, start - self.word_padding),
                                    min(len(vertical_projection), 
                                        i - gap_counter + self.word_padding)
                                ))
                            in_word = False
                            gap_counter = 0
            
            # Handle last word
            if in_word and (len(vertical_projection) - start) > min_width:
                words.append((max(0, start - self.word_padding),
                            len(vertical_projection)))
            
            # Save word/ligature images
            for w_idx, (start, end) in enumerate(words):
                word_img = line_color[:, start:end]
                word_binary = line_binary[:, start:end]
                
                if word_img.shape[0] > 5 and word_img.shape[1] > 5:
                    # Check if word has sufficient text
                    has_text, text_pixels, text_ratio = self.has_sufficient_text(
                        word_img, word_binary
                    )
                    
                    if not has_text:
                        skipped_words += 1
                        continue
                    
                    word_img_resized = self.resize_image(word_img)
                    
                    filename = f"word_{word_count:04d}_line{line_idx}.png"
                    cv2.imwrite(f"{self.output_dir}/words/{filename}", 
                              word_img_resized)
                    
                    all_words.append((
                        word_binary, 
                        word_img, 
                        word_count, 
                        line_idx
                    ))
                    word_count += 1
        
        print(f"✓ Segmented {word_count} words/ligatures (skipped {skipped_words} empty)")
        return all_words
    
    def segment_ligatures_advanced(self, word_images, min_area=25):
        """
        Advanced ligature/character segmentation for Sindhi
        Uses connected components with proper handling of dots and diacritics
        """
        ligature_count = 0
        skipped_ligatures = 0
        
        for word_binary, word_color, word_idx, line_idx in word_images:
            h, w = word_binary.shape
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                word_binary, connectivity=8
            )
            
            # Collect valid components
            components = []
            for i in range(1, num_labels):  # Skip background (0)
                x, y, w_comp, h_comp, area = stats[i]
                
                # Filter noise but keep dots (important for Sindhi)
                if area > min_area or (area > 5 and h_comp < h * 0.3):
                    components.append((x, y, w_comp, h_comp, area, i))
            
            # Sort by x-coordinate (RTL: right to left for Sindhi)
            components.sort(key=lambda c: c[0], reverse=True)
            
            # Group nearby components (handle multi-part characters)
            grouped = self._group_components(components, word_binary.shape[1])
            
            # Save ligature images
            for group in grouped:
                # Get bounding box of group
                x_min = min(c[0] for c in group)
                y_min = min(c[1] for c in group)
                x_max = max(c[0] + c[2] for c in group)
                y_max = max(c[1] + c[3] for c in group)
                
                # Add padding
                x1 = max(0, x_min - self.letter_padding)
                y1 = max(0, y_min - self.letter_padding)
                x2 = min(word_color.shape[1], x_max + self.letter_padding)
                y2 = min(word_color.shape[0], y_max + self.letter_padding)
                
                ligature_img = word_color[y1:y2, x1:x2]
                ligature_binary = word_binary[y1:y2, x1:x2]
                
                if ligature_img.shape[0] > 5 and ligature_img.shape[1] > 5:
                    # More lenient checking for ligatures (they're smaller)
                    # Use much lower thresholds since ligatures can be small
                    text_pixels = np.sum(ligature_binary > 0)
                    total_pixels = ligature_binary.shape[0] * ligature_binary.shape[1]
                    text_ratio = text_pixels / total_pixels if total_pixels > 0 else 0
                    
                    # Very lenient thresholds for ligatures
                    has_text = (text_pixels >= 10 and text_ratio >= 0.01)
                    
                    if not has_text:
                        skipped_ligatures += 1
                        continue
                    
                    ligature_img_resized = self.resize_image(ligature_img)
                    
                    filename = f"lig_{ligature_count:05d}_w{word_idx}_l{line_idx}.png"
                    cv2.imwrite(f"{self.output_dir}/ligatures/{filename}", 
                              ligature_img_resized)
                    ligature_count += 1
        
        print(f"✓ Segmented {ligature_count} ligatures/characters (skipped {skipped_ligatures} empty)")
        return ligature_count
    
    def _group_components(self, components, width, max_gap=15):
        """Group nearby components (handles dots and multi-part characters)"""
        if not components:
            return []
        
        grouped = []
        current_group = [components[0]]
        
        for i in range(1, len(components)):
            prev_x = current_group[-1][0]
            curr_x = components[i][0]
            
            # Calculate gap (consider RTL direction)
            gap = abs(prev_x - curr_x)
            
            if gap < max_gap:
                current_group.append(components[i])
            else:
                grouped.append(current_group)
                current_group = [components[i]]
        
        grouped.append(current_group)
        return grouped
    
    def process(self, segment_level="words", save_debug=True):
        """
        Main processing pipeline
        
        Args:
            segment_level: "lines", "words", or "ligatures"
            save_debug: Save intermediate images for debugging
        """
        print(f"\n{'='*60}")
        print(f"Processing Sindhi Text: {self.image_path}")
        print(f"Segment Level: {segment_level}")
        print(f"Auto-deskew: {self.auto_deskew}")
        print(f"{'='*60}\n")
        
        # Preprocess (includes skew correction)
        self.load_and_preprocess(save_debug=save_debug)
        
        # Segment lines
        line_images = self.segment_lines()
        
        if segment_level == "lines":
            print(f"\n✅ Complete! Check '{self.output_dir}/lines' directory")
            return
        
        # Segment words/ligatures
        word_images = self.segment_words_or_ligatures(line_images)
        
        if segment_level == "words":
            print(f"\n✅ Complete! Check '{self.output_dir}/words' directory")
            return
        
        # Segment ligatures/characters
        self.segment_ligatures_advanced(word_images)
        
        print(f"\n✅ Complete! Check '{self.output_dir}/ligatures' directory")
        print(f"Debug images saved in '{self.output_dir}/debug'")


def batch_process(input_dir, output_base_dir, segment_level="words", **kwargs):
    """
    Process multiple images in batch
    
    Args:
        input_dir: Directory containing Sindhi text images
        output_base_dir: Base directory for outputs
        segment_level: Segmentation level
        **kwargs: Additional parameters for SindhiTextSegmenter
    """
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.jpg")) + \
                 list(input_path.glob("*.png")) + \
                 list(input_path.glob("*.jpeg"))
    
    print(f"\nFound {len(image_files)} images to process\n")
    
    for idx, img_file in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_file.name}")
        
        output_dir = f"{output_base_dir}/{img_file.stem}"
        
        segmenter = SindhiTextSegmenter(
            image_path=str(img_file),
            output_dir=output_dir,
            **kwargs
        )
        
        segmenter.process(segment_level=segment_level)


# Example usage
if __name__ == "__main__":
    # Single image processing with auto-deskew
    segmenter = SindhiTextSegmenter(
        image_path="input_images/sindhi_page.jpg",
        output_dir="output_sindhi",
        line_padding=20,       # Extra space for diacritics
        word_padding=10,       # Extra space for ligatures
        letter_padding=7,      # Extra space for dots
        output_height=64,      # Standard size for ML models
        output_width=None,     # Maintain aspect ratio
        denoise_strength=12,   # Noise reduction
        morph_kernel_size=2,   # Connect broken characters
        auto_deskew=True,      # Enable automatic skew correction
        skew_threshold=0.5,    # Minimum angle to correct (degrees)
        min_text_pixels=50,    # Minimum text pixels to keep segment
        text_threshold=0.05    # Minimum text ratio (5% of pixels)
    )
    
    # For word-level segmentation (recommended)
    segmenter.process(segment_level="words")
    
    # For ligature/character segmentation
    # segmenter.process(segment_level="ligatures")
    
    # Disable auto-deskew if needed
    # segmenter_no_deskew = SindhiTextSegmenter(
    #     image_path="input_images/sindhi_page.jpg",
    #     output_dir="output_sindhi_no_deskew",
    #     auto_deskew=False
    # )
    
    # Batch processing multiple images
    # batch_process(
    #     input_dir="input_images",
    #     output_base_dir="batch_output",
    #     segment_level="words",
    #     line_padding=20,
    #     word_padding=10,
    #     output_height=64,
    #     auto_deskew=True,
    #     skew_threshold=0.5
    # )