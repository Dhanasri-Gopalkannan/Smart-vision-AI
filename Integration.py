import os
import sys

# Suppress TensorFlow and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

# Suppress Python warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Suppress logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('h5py').setLevel(logging.ERROR)

# Suppress all output to stderr temporarily
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# ========== IMPORTS ==========
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import json
import argparse
from pathlib import Path

# Suppress matplotlib warnings
plt.rcParams.update({'figure.max_open_warning': 0})
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# PIL warnings
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# TensorFlow imports
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# YOLO imports
import ultralytics
ultralytics.utils.LOGGER.setLevel(logging.ERROR)
from ultralytics import YOLO

# Restore stderr
sys.stderr.close()
sys.stderr = original_stderr

print("="*70)
print("OBJECT DETECTION & CLASSIFICATION PIPELINE")
print("="*70)
print("✅ All imports successful")


# ========== CONFIGURATION ==========
class Config:
    """Configuration class for paths and parameters"""
    
    # ===== PATHS - UPDATE THESE =====
    
    # Model paths
    YOLO_MODEL_PATH = r"C:\Users\Satish Kumar\Desktop\pythoncode - guvi\Smartvision_AI\yolov8_complete_results\best_model.pt"
    CLASSIFICATION_MODEL_PATH = r"C:\Users\Satish Kumar\Desktop\pythoncode - guvi\Smartvision_AI\Transfer_learningmodel_results\VGG16_rebuilt.h5"
    
    # ===== IMAGE FOLDER PATH - UPDATE THIS =====
    # Point this to your folder containing test images
    IMAGE_FOLDER_PATH = r"C:\Users\Satish Kumar\Desktop\pythoncode - guvi\Smartvision_AI\dataset_processing_output\detection_processed\val\images"
    
    # Or specify a single image (comment out the above and uncomment below)
    # SINGLE_IMAGE_PATH = r"C:\Users\Satish Kumar\Desktop\pythoncode - guvi\Smartvision_AI\dataset_processing_output\detection_processed\val\images\image_000001.jpg"
    
    # ===== PROCESSING OPTIONS =====
    PROCESS_MODE = "folder"  # "single" or "folder" or "batch"
    BATCH_SIZE = 5  # Number of images to process if using folder
    
    # Class names (26 classes from your dataset)
    CLASS_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'traffic light', 'stop sign', 'bench', 'bird', 'cat', 'dog', 'horse', 'cow',
        'elephant', 'bottle', 'cup', 'bowl', 'pizza', 'cake', 'chair', 'couch',
        'potted plant', 'bed'
    ]
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.25
    NMS_IOU_THRESHOLD = 0.5
    FINAL_CONFIDENCE_THRESHOLD = 0.5
    USE_CLASSIFICATION = True  # Set to False to disable classification
    
    # Image parameters
    IMAGE_SIZE = (224, 224)  # For classification
    
    # Output directories
    OUTPUT_DIR = "output"
    SAVE_PLOTS = True


# ========== POST-PROCESSING CLASS ==========
class PostProcessor:
    """Post-processing utilities for object detection"""
    
    @staticmethod
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def apply_nms(detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return detections
            
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            current = detections.pop(0)
            keep.append(current)
            
            to_remove = []
            for i, det in enumerate(detections):
                if PostProcessor.calculate_iou(current['bbox'], det['bbox']) > iou_threshold:
                    to_remove.append(i)
            
            for i in reversed(to_remove):
                detections.pop(i)
        
        return keep
    
    @staticmethod
    def filter_by_confidence(detections, threshold=0.5):
        """Filter by confidence threshold"""
        return [d for d in detections if d['confidence'] >= threshold]
    
    @staticmethod
    def refine_bbox(image, bbox, padding=0.05):
        """Refine bounding box"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        pad_x = (x2 - x1) * padding
        pad_y = (y2 - y1) * padding
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        return [int(x1), int(y1), int(x2), int(y2)]
    
    @staticmethod
    def format_output(detections, class_names):
        """Format detections for JSON output"""
        formatted = []
        for det in detections:
            class_name = class_names[det['class_id']] if det['class_id'] < len(class_names) else f"class_{det['class_id']}"
            formatted.append({
                'class': class_name,
                'class_id': det['class_id'],
                'confidence': round(det['confidence'], 3),
                'bbox': [int(x) for x in det['bbox']]
            })
        return formatted


# ========== DETECTION PIPELINE CLASS ==========
class ObjectDetectionPipeline:
    """Complete end-to-end prediction pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.post_processor = PostProcessor()
        
        # Load YOLO model
        print(f"\n📥 Loading YOLO model...")
        if os.path.exists(config.YOLO_MODEL_PATH):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.detection_model = YOLO(config.YOLO_MODEL_PATH)
            print("✅ YOLO model loaded successfully")
        else:
            raise FileNotFoundError(f"YOLO model not found at {config.YOLO_MODEL_PATH}")
        
        # Load classification model
        self.classification_model = None
        if os.path.exists(config.CLASSIFICATION_MODEL_PATH):
            try:
                print(f"📥 Loading classification model...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Suppress TensorFlow loading messages
                    original_stdout = sys.stdout
                    sys.stdout = open(os.devnull, 'w')
                    
                    self.classification_model = load_model(
                        config.CLASSIFICATION_MODEL_PATH, 
                        compile=False
                    )
                    
                    sys.stdout.close()
                    sys.stdout = original_stdout
                    
                print("✅ Classification model loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load classification model: {e}")
                self.classification_model = None
        else:
            print("⚠️ Classification model not found. Using YOLO only.")
        
        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        print(f"\n📁 Output directory: {config.OUTPUT_DIR}")
    
    def preprocess_for_classification(self, image_array):
        """Preprocess image for classification model"""
        img = cv2.resize(image_array, self.config.IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def extract_object(self, image, bbox):
        """Extract object from image using bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        
        h, w = image.shape[:2]
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        if x2 > x1 and y2 > y1:
            return image[y1:y2, x1:x2]
        return None
    
    def classify_object(self, object_img):
        """Classify extracted object using CNN model"""
        if self.classification_model is None:
            return None, None
        
        try:
            # Preprocess
            img_array = self.preprocess_for_classification(object_img)
            
            # Predict (suppress all output)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                
                predictions = self.classification_model.predict(img_array, verbose=0)
                
                sys.stdout.close()
                sys.stdout = original_stdout
            
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            return class_idx, confidence
        except Exception as e:
            return None, None
    
    def detect(self, image_path, use_classification=True, save_result=True):
        """Main detection function"""
        print(f"\n🔍 Processing: {os.path.basename(image_path)}")
        start_time = time.time()
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO Detection (suppress warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self.detection_model(
                image_path, 
                conf=self.config.CONFIDENCE_THRESHOLD, 
                verbose=False
            )
        
        # Process detections
        raw_detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'confidence': confidence,
                        'classification_verified': False
                    }
                    
                    # Optional Classification Verification
                    if use_classification and self.classification_model:
                        object_img = self.extract_object(image, [x1, y1, x2, y2])
                        if object_img is not None:
                            cnn_class, cnn_conf = self.classify_object(object_img)
                            if cnn_class is not None and cnn_conf > 0.5:
                                detection['class_id'] = cnn_class
                                detection['confidence'] = (confidence + cnn_conf) / 2
                                detection['classification_verified'] = True
                    
                    raw_detections.append(detection)
        
        detection_time = time.time() - start_time
        print(f"   Detection time: {detection_time*1000:.2f} ms")
        print(f"   Raw detections: {len(raw_detections)}")
        
        # Post-processing
        processed_detections = self.post_processor.apply_nms(
            raw_detections, iou_threshold=self.config.NMS_IOU_THRESHOLD
        )
        print(f"   After NMS: {len(processed_detections)}")
        
        processed_detections = self.post_processor.filter_by_confidence(
            processed_detections, threshold=self.config.FINAL_CONFIDENCE_THRESHOLD
        )
        print(f"   After confidence filter: {len(processed_detections)}")
        
        # Refine bounding boxes
        for det in processed_detections:
            det['bbox'] = self.post_processor.refine_bbox(image, det['bbox'])
        
        # Visualize and save
        if save_result:
            self.visualize_results(
                image, 
                processed_detections, 
                save_path=os.path.join(self.config.OUTPUT_DIR, f"result_{os.path.basename(image_path)}")
            )
        
        # Format output
        formatted_output = self.post_processor.format_output(
            processed_detections, self.config.CLASS_NAMES
        )
        
        return {
            'image': os.path.basename(image_path),
            'detections': formatted_output,
            'count': len(formatted_output),
            'processing_time_ms': detection_time * 1000
        }
    
    def visualize_results(self, image, detections, save_path=None):
        """Visualize results with bounding boxes and labels"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(image)
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                
                class_name = self.config.CLASS_NAMES[det['class_id']] if det['class_id'] < len(self.config.CLASS_NAMES) else f"Class_{det['class_id']}"
                label = f"{class_name} ({det['confidence']:.2f})"
                
                ax.text(
                    x1, y1 - 5, label,
                    fontsize=10, color='white',
                    bbox=dict(facecolor='red', alpha=0.7, pad=2)
                )
            
            ax.set_title(f"Detection Results ({len(detections)} objects)")
            ax.axis('off')
            
            if save_path and self.config.SAVE_PLOTS:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   Results saved to: {save_path}")
            plt.show()
            plt.close(fig)
    
    def save_results_json(self, results, filename="detection_results.json"):
        """Save detection results to JSON file"""
        json_path = os.path.join(self.config.OUTPUT_DIR, filename)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to: {json_path}")
        return json_path
    
    def process_folder(self, folder_path, max_images=None):
        """Process all images in a folder"""
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            return []
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(image_extensions)]
        
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            print(f"❌ No images found in {folder_path}")
            return []
        
        print(f"\n📁 Found {len(image_files)} images in folder")
        
        results = []
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(folder_path, img_file)
            print(f"\n[{i+1}/{len(image_files)}] Processing...")
            
            result = self.detect(
                img_path, 
                use_classification=self.config.USE_CLASSIFICATION,
                save_result=True
            )
            results.append(result)
            
            print(f"   Found {result['count']} objects")
        
        return results


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING PIPELINE WITH CONFIGURED PATHS")
    print("="*70)
    
    # Initialize pipeline
    try:
        pipeline = ObjectDetectionPipeline(Config)
    except Exception as e:
        print(f"\n❌ Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Check which mode to run
    if hasattr(Config, 'SINGLE_IMAGE_PATH') and os.path.exists(Config.SINGLE_IMAGE_PATH):
        # Process single image
        print(f"\n📷 Processing single image: {Config.SINGLE_IMAGE_PATH}")
        result = pipeline.detect(
            Config.SINGLE_IMAGE_PATH, 
            use_classification=Config.USE_CLASSIFICATION
        )
        
        print("\n📊 DETECTION RESULTS:")
        print(json.dumps(result, indent=2))
        pipeline.save_results_json(result, "single_image_results.json")
        
    elif os.path.exists(Config.IMAGE_FOLDER_PATH):
        # Process folder
        print(f"\n📁 Processing folder: {Config.IMAGE_FOLDER_PATH}")
        
        if Config.PROCESS_MODE == "folder":
            # Process all images
            results = pipeline.process_folder(
                Config.IMAGE_FOLDER_PATH,
                max_images=None
            )
        elif Config.PROCESS_MODE == "batch":
            # Process batch of images
            results = pipeline.process_folder(
                Config.IMAGE_FOLDER_PATH,
                max_images=Config.BATCH_SIZE
            )
        else:
            print(f"❌ Invalid PROCESS_MODE: {Config.PROCESS_MODE}")
            results = []
        
        if results:
            pipeline.save_results_json(results, "folder_results.json")
            print(f"\n✅ Processed {len(results)} images successfully")
    
    else:
        print("\n❌ No valid image path found in configuration!")
        print("\nPlease update the Config class with either:")
        print("  - IMAGE_FOLDER_PATH = r'path/to/your/image/folder'")
        print("  - SINGLE_IMAGE_PATH = r'path/to/your/image.jpg'")
    
    print("\n" + "="*70)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("="*70)