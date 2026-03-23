"""
YOLOv8 Inference Script
Use this in VS Code with your trained model
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Class names (26 classes)
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'traffic light', 'stop sign', 'bench', 'bird', 'cat', 'dog', 'horse', 'cow',
    'elephant', 'bottle', 'cup', 'bowl', 'pizza', 'cake', 'chair', 'couch',
    'potted plant', 'bed'
]

# Load your trained model
model = YOLO('best_model.pt')

def detect_objects(image_path, confidence=0.25):
    """
    Detect objects in an image
    """
    print(f"\n🔍 Processing: {image_path}")
    
    # Run detection
    results = model(image_path, conf=confidence)
    
    # Show results
    for r in results:
        # Plot image with boxes
        im_array = r.plot()
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.title(f'Detection Results (Confidence: {confidence})')
        plt.show()
        
        # Print detections
        print(f"\n📊 Found {len(r.boxes)} objects:")
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"  {i+1}. {class_name}: {confidence:.2f}  [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    return results

# Example usage
if __name__ == "__main__":
    # Test on an image
    detect_objects("test_image.jpg", confidence=0.25)
