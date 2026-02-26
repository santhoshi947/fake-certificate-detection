import cv2
import numpy as np
import os
from pathlib import Path
import logging
from typing import Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CertificatePreprocessor:
    """
    A comprehensive image preprocessing pipeline for certificate detection.
    Handles both genuine and fake certificates with various quality issues.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            target_size: Desired output image size (width, height)
        """
        self.target_size = target_size
        self.processed_images = []
        self.labels = []
        self.image_paths = []
        
    def read_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Safely read an image with error handling.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image array or None if reading fails
        """
        try:
            # Read image using OpenCV
            img = cv2.imread(str(image_path))
            
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
                
            return img
            
        except Exception as e:
            logger.error(f"Error reading image {image_path}: {str(e)}")
            return None
    
    def is_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """
        Detect if an image is blurry using Laplacian variance.
        
        Args:
            image: Input image (grayscale)
            threshold: Variance threshold for blur detection
            
        Returns:
            True if image is blurry, False otherwise
        """
        # Compute Laplacian variance
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        
        # Log blurriness for debugging
        logger.debug(f"Laplacian variance: {laplacian_var:.2f}")
        
        return laplacian_var < threshold
    
    def enhance_text_clarity(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance text clarity using adaptive thresholding and morphological operations.
        
        Args:
            image: Grayscale image
            
        Returns:
            Enhanced image with clearer text
        """
        # Apply adaptive thresholding to handle varying lighting conditions
        # This is crucial for scanned/xeroxed certificates
        binary = cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11,  # Block size (must be odd)
            2    # Constant subtracted from mean
        )
        
        # Apply morphological operations to clean up the text
        # Remove small noise dots while preserving text structure
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def preprocess_single_image(self, image_path: str, label: int) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline for a single image.
        
        Args:
            image_path: Path to the image
            label: Class label (0 for fake, 1 for genuine)
            
        Returns:
            Preprocessed image array or None if processing fails
        """
        try:
            # Step 1: Read image
            img = self.read_image(image_path)
            if img is None:
                return None
            
            # Step 2: Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Step 3: Handle low-quality/blurry images
            if self.is_blurry(gray):
                logger.warning(f"Blurry image detected: {image_path}")
                # Apply sharpening for blurry images
                kernel_sharpen = np.array([[-1,-1,-1],
                                         [-1, 9,-1],
                                         [-1,-1,-1]])
                gray = cv2.filter2D(gray, -1, kernel_sharpen)
            
            # Step 4: Noise reduction
            # Use bilateral filter to preserve edges while reducing noise
            # This is better than Gaussian blur for document images
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Alternative: Use median blur for salt-and-pepper noise
            # denoised = cv2.medianBlur(gray, 3)
            
            # Step 5: Improve contrast for better text visibility
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            contrast_enhanced = clahe.apply(denoised)
            
            # Step 6: Enhance text clarity (handles xerox/scanned variations)
            text_enhanced = self.enhance_text_clarity(contrast_enhanced)
            
            # Step 7: Resize to fixed size
            resized = cv2.resize(text_enhanced, self.target_size, 
                                interpolation=cv2.INTER_CUBIC)
            
            # Step 8: Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Store metadata
            self.image_paths.append(image_path)
            
            logger.debug(f"Successfully processed: {image_path}")
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing {image_path}: {str(e)}")
            return None
    
    def load_and_preprocess_folder(self, folder_path: str, label: int) -> None:
        """
        Load and preprocess all images from a folder.
        
        Args:
            folder_path: Path to folder containing images
            label: Class label for images in this folder
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.error(f"Folder does not exist: {folder_path}")
            return
        
        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Get all image files
        image_files = [
            f for f in folder.iterdir() 
            if f.suffix.lower() in extensions and f.is_file()
        ]
        
        logger.info(f"Found {len(image_files)} images in {folder_path}")
        
        # Process images (optionally using parallel processing for speed)
        # For small datasets, sequential processing is fine
        for img_path in image_files:
            processed = self.preprocess_single_image(img_path, label)
            
            if processed is not None:
                self.processed_images.append(processed)
                self.labels.append(label)
        
        logger.info(f"Successfully processed {len(self.processed_images)} images from {folder_path}")
    
    def process_dataset(self, genuine_folder: str, fake_folder: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process the entire dataset.
        
        Args:
            genuine_folder: Path to folder with genuine certificates
            fake_folder: Path to folder with fake certificates
            
        Returns:
            Tuple of (X, y) where X is the image array and y is labels array
        """
        # Clear previous data
        self.processed_images = []
        self.labels = []
        self.image_paths = []
        
        # Process genuine certificates (label = 1)
        self.load_and_preprocess_folder(genuine_folder, label=1)
        
        # Process fake certificates (label = 0)
        self.load_and_preprocess_folder(fake_folder, label=0)
        
        # Convert to numpy arrays
        X = np.array(self.processed_images)
        y = np.array(self.labels)
        
        # Add channel dimension for CNN input
        X = X.reshape(X.shape[0], self.target_size[0], self.target_size[1], 1)
        
        logger.info(f"Dataset processed: {X.shape[0]} images, {X.shape[1]}x{X.shape[2]} pixels")
        logger.info(f"Class distribution - Genuine: {sum(y==1)}, Fake: {sum(y==0)}")
        
        return X, y
    
    def visualize_preprocessing(self, image_path: str, save_path: Optional[str] = None):
        """
        Visualize the preprocessing steps for debugging.
        
        Args:
            image_path: Path to test image
            save_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        
        # Read original image
        original = cv2.imread(image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing steps
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # Denoising
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        
        # Text enhancement
        text_enhanced = self.enhance_text_clarity(contrast)
        
        # Final resized
        final = cv2.resize(text_enhanced, self.target_size)
        
        # Plot results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(denoised, cmap='gray')
        axes[0, 2].set_title('Denoised')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(contrast, cmap='gray')
        axes[1, 0].set_title('Contrast Enhanced')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(text_enhanced, cmap='gray')
        axes[1, 1].set_title('Text Enhanced')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(final, cmap='gray')
        axes[1, 2].set_title(f'Final ({self.target_size[0]}x{self.target_size[1]})')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CertificatePreprocessor(target_size=(256, 256))
    
    # Process the dataset
    # Update these paths to your actual folder locations
    genuine_folder = "data/genuine"
    fake_folder = "data/fake"
    
    # Check if folders exist
    if not os.path.exists(genuine_folder) or not os.path.exists(fake_folder):
        logger.error("Please update the folder paths to your actual certificate locations")
        logger.info("Example structure:")
        logger.info("  genuine/")
        logger.info("    cert1.jpg")
        logger.info("    cert2.png")
        logger.info("  fake/")
        logger.info("    fake1.jpg")
        logger.info("    fake2.png")
    else:
        # Process the entire dataset
        X, y = preprocessor.process_dataset(genuine_folder, fake_folder)
        
        # Save processed data for ML training
        np.savez_compressed('certificate_dataset.npz', X=X, y=y, 
                           image_paths=preprocessor.image_paths)
        
        logger.info("Dataset saved to 'certificate_dataset.npz'")
        
        # Optional: Visualize preprocessing for a sample image
        if len(preprocessor.image_paths) > 0:
            sample_path = preprocessor.image_paths[0]
            preprocessor.visualize_preprocessing(sample_path, 'preprocessing_steps.png')