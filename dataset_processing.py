import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from tqdm import tqdm
import albumentations as A
import yaml

class DatasetProcessor:
    def __init__(self, classification_dir, detection_dir=None, output_dir='./processed'):
        """
        Initialize the processor
        """
        self.class_dir = Path(classification_dir)
        self.det_dir = Path(detection_dir) if detection_dir else None
        self.output = Path(output_dir)
        
        # Create output directories
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'analysis').mkdir(exist_ok=True)
        (self.output / 'classification_processed').mkdir(exist_ok=True)
        (self.output / 'detection_processed').mkdir(exist_ok=True)
        (self.output / 'augmented').mkdir(exist_ok=True)
        
        # Get classes
        self.classes = self._get_classes()
        print(f"Found {len(self.classes)} classes: {self.classes}")
        
        # Setup augmentations
        self._setup_augmentations()
    
    def _get_classes(self):
        """Get class names from classification dataset"""
        train_dir = self.class_dir / 'train'
        if train_dir.exists():
            classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
            return sorted(classes)
        return []
    
    def _setup_augmentations(self):
        """Setup augmentation pipelines"""
        # Classification augmentations
        self.cls_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.CenterCrop(height=200, width=200, p=0.5),  
    A.Resize(height=224, width=224, p=1.0),    
])
        
        # Detection augmentations
        self.det_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))
    
    def analyze_datasets(self):
        """Perform EDA on both datasets"""
        print("\n=== DATASET ANALYSIS ===\n")
        
        # Analyze classification
        print("1. Classification Dataset:")
        cls_stats = self._analyze_classification()
        
        # Analyze detection if exists
        det_stats = None
        if self.det_dir and self.det_dir.exists():
            print("\n2. Detection Dataset:")
            det_stats = self._analyze_detection()
        
        # Create visualizations
        self._create_visualizations(cls_stats, det_stats)
        
        # Save stats
        stats = {'classification': cls_stats}
        if det_stats:
            stats['detection'] = det_stats
        
        with open(self.output / 'analysis' / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        return stats
    
    def _analyze_classification(self):
        """Analyze classification dataset"""
        stats = {
            'total_images': 0,
            'split_counts': {},
            'class_counts': {},
            'image_sizes': []
        }
        
        for split in ['train', 'val', 'test']:
            split_dir = self.class_dir / split
            if not split_dir.exists():
                print(f"  {split}: not found")
                continue
            
            split_total = 0
            class_counts = {}
            
            for cls in self.classes:
                cls_dir = split_dir / cls
                if cls_dir.exists():
                    images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
                    count = len(images)
                    class_counts[cls] = count
                    split_total += count
                    
                    # Sample image sizes
                    for img_path in images[:2]:
                        try:
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                stats['image_sizes'].append(img.shape[:2])
                        except:
                            pass
            
            stats['split_counts'][split] = split_total
            stats['class_counts'][split] = class_counts
            stats['total_images'] += split_total
        
        print(f"  Total images: {stats['total_images']}")
        print(f"  Train: {stats['split_counts'].get('train', 0)}")
        print(f"  Validation: {stats['split_counts'].get('val', 0)}")
        print(f"  Test: {stats['split_counts'].get('test', 0)}")
        
        return stats
    
    def _analyze_detection(self):
        """Analyze detection dataset"""
        stats = {
            'total_images': 0,
            'total_objects': 0,
            'objects_per_image': [],
            'class_distribution': {},
            'image_sizes': [],
            'bbox_sizes': []
        }
        
        # Check structure
        images_dir = self.det_dir / 'images'
        labels_dir = self.det_dir / 'labels'
        
        if not images_dir.exists():
            print("  Error: 'images' folder not found!")
            return stats
        
        # Get images
        images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f"  Found {len(images)} images")
        
        # Parse a sample of images
        sample_size = min(100, len(images))
        print(f"  Analyzing {sample_size} sample images...")
        
        for img_path in tqdm(images[:sample_size], desc="  Analyzing"):
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                stats['image_sizes'].append((w, h))
                stats['total_images'] += 1
                
                # Check for corresponding label
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    obj_count = len(lines)
                    stats['total_objects'] += obj_count
                    stats['objects_per_image'].append(obj_count)
                    
                    # Parse each object
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            
                            # Update class distribution
                            stats['class_distribution'][class_id] = stats['class_distribution'].get(class_id, 0) + 1
                            
                            # Calculate bbox size in pixels
                            bbox_w = width * w
                            bbox_h = height * h
                            stats['bbox_sizes'].append((bbox_w, bbox_h))
                
            except Exception as e:
                continue
        
        # Calculate statistics
        if stats['image_sizes']:
            sizes = np.array(stats['image_sizes'])
            stats['avg_image_size'] = [float(np.mean(sizes[:, 0])), float(np.mean(sizes[:, 1]))]
        
        if stats['objects_per_image']:
            stats['avg_objects_per_image'] = float(np.mean(stats['objects_per_image']))
        
        if stats['bbox_sizes']:
            bbox_sizes = np.array(stats['bbox_sizes'])
            stats['avg_bbox_size'] = [float(np.mean(bbox_sizes[:, 0])), float(np.mean(bbox_sizes[:, 1]))]
        
        print(f"  Total objects: {stats['total_objects']}")
        print(f"  Avg objects per image: {stats.get('avg_objects_per_image', 0):.2f}")
        print(f"  Classes found: {list(stats['class_distribution'].keys())}")
        
        return stats
    
    def _create_visualizations(self, cls_stats, det_stats):
        """Create visualizations"""
        # Determine how many plots we need
        num_plots = 2 if cls_stats else 0
        if det_stats and det_stats['total_images'] > 0:
            num_plots += 2
        
        if num_plots == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        plot_idx = 0
        
        # Plot 1: Classification Class Distribution
        if cls_stats and 'train' in cls_stats['class_counts']:
            ax = axes[plot_idx]
            train_counts = cls_stats['class_counts']['train']
            classes = list(train_counts.keys())[:10]
            counts = [train_counts[cls] for cls in classes]
            
            ax.barh(range(len(classes)), counts, color='steelblue')
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes)
            ax.set_title('Classification: Top 10 Classes (Train)')
            ax.set_xlabel('Number of Images')
            plot_idx += 1
        
        # Plot 2: Classification Split Distribution
        if cls_stats and cls_stats['split_counts']:
            ax = axes[plot_idx]
            split_counts = cls_stats['split_counts']
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            ax.pie(split_counts.values(), labels=split_counts.keys(), 
                  autopct='%1.1f%%', colors=colors[:len(split_counts)])
            ax.set_title('Classification: Split Distribution')
            plot_idx += 1
        
        # Plot 3: Detection Objects per Image
        if det_stats and det_stats['objects_per_image']:
            ax = axes[plot_idx]
            objects_per_image = det_stats['objects_per_image']
            ax.hist(objects_per_image, bins=20, edgecolor='black', alpha=0.7)
            ax.set_title('Detection: Objects per Image')
            ax.set_xlabel('Number of Objects')
            ax.set_ylabel('Frequency')
            plot_idx += 1
        
        # Plot 4: Detection Class Distribution
        if det_stats and det_stats['class_distribution']:
            ax = axes[plot_idx]
            class_dist = det_stats['class_distribution']
            classes = sorted(class_dist.keys())[:10]
            counts = [class_dist[cls] for cls in classes]
            
            ax.bar(range(len(classes)), counts, color='lightcoral', edgecolor='black')
            ax.set_title('Detection: Top 10 Classes')
            ax.set_xlabel('Class ID')
            ax.set_ylabel('Number of Objects')
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels([f'Class {c}' for c in classes])
            plot_idx += 1
        
        # Hide unused plots
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output / 'analysis' / 'dataset_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualizations saved to: {self.output / 'analysis' / 'dataset_analysis.png'}")
    
    def preprocess_classification(self, target_size=224):
        """Preprocess classification data"""
        print("\n=== PREPROCESSING CLASSIFICATION DATA ===\n")
        
        for split in ['train', 'val', 'test']:
            src_dir = self.class_dir / split
            dst_dir = self.output / 'classification_processed' / split
            
            if not src_dir.exists():
                print(f"  Skipping {split} - not found")
                continue
            
            print(f"  Processing {split}...")
            
            for cls in tqdm(self.classes, desc=f"  Classes"):
                src_cls_dir = src_dir / cls
                dst_cls_dir = dst_dir / cls
                
                if not src_cls_dir.exists():
                    continue
                
                dst_cls_dir.mkdir(parents=True, exist_ok=True)
                
                # Process images
                images = list(src_cls_dir.glob('*.jpg')) + list(src_cls_dir.glob('*.png'))
                
                for img_path in images:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                        
                        img_resized = cv2.resize(img, (target_size, target_size))
                        dst_path = dst_cls_dir / f"{img_path.stem}_processed.jpg"
                        cv2.imwrite(str(dst_path), img_resized)
                        
                    except:
                        continue
        
        print(f"\nClassification preprocessing complete!")
        print(f"Saved to: {self.output / 'classification_processed'}")
    
    def preprocess_detection(self):
        """Preprocess detection data - create train/val/test splits"""
        if not self.det_dir or not self.det_dir.exists():
            print("\nDetection dataset not found!")
            return
        
        print("\n=== PREPROCESSING DETECTION DATA ===\n")
        
        # Check structure
        images_dir = self.det_dir / 'images'
        labels_dir = self.det_dir / 'labels'
        
        if not images_dir.exists():
            print("  Error: 'images' folder not found!")
            return
        
        # Get all images
        images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f"  Found {len(images)} images")
        
        if len(images) == 0:
            print("  No images found!")
            return
        
        # Create splits (70% train, 15% val, 15% test)
        np.random.shuffle(images)
        train_count = int(0.7 * len(images))
        val_count = int(0.15 * len(images))
        
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        print(f"  Splitting: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
        
        # Process each split
        for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            dst_img_dir = self.output / 'detection_processed' / split_name / 'images'
            dst_lbl_dir = self.output / 'detection_processed' / split_name / 'labels'
            
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Processing {split_name} split...")
            
            for img_path in tqdm(split_images, desc=f"    {split_name}"):
                try:
                    # Copy image
                    shutil.copy2(img_path, dst_img_dir / img_path.name)
                    
                    # Copy corresponding label if exists
                    label_path = labels_dir / f"{img_path.stem}.txt"
                    if label_path.exists():
                        shutil.copy2(label_path, dst_lbl_dir / label_path.name)
                    
                except:
                    continue
        
        # Copy data.yaml if exists
        yaml_src = self.det_dir / 'data.yaml'
        yaml_dst = self.output / 'detection_processed' / 'data.yaml'
        if yaml_src.exists():
            shutil.copy2(yaml_src, yaml_dst)
            print(f"  Copied data.yaml configuration")
        
        print(f"\nDetection preprocessing complete!")
        print(f"Saved to: {self.output / 'detection_processed'}")
    
    def augment_classification(self):
        """Augment classification training data"""
        print("\n=== AUGMENTING CLASSIFICATION DATA ===\n")
        
        src_dir = self.class_dir / 'train'
        dst_dir = self.output / 'augmented' / 'classification' / 'train'
        
        if not src_dir.exists():
            print("Training directory not found!")
            return
        
        print("Creating 2 augmented versions per image...")
        
        total_original = 0
        total_augmented = 0
        
        for cls in tqdm(self.classes, desc="Processing classes"):
            src_cls_dir = src_dir / cls
            dst_cls_dir = dst_dir / cls
            
            if not src_cls_dir.exists():
                continue
            
            dst_cls_dir.mkdir(parents=True, exist_ok=True)
            
            images = list(src_cls_dir.glob('*.jpg')) + list(src_cls_dir.glob('*.png'))
            total_original += len(images)
            
            for img_path in images:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Save original
                    original_dst = dst_cls_dir / f"{img_path.stem}_original.jpg"
                    cv2.imwrite(str(original_dst), img)
                    
                    # Create augmented versions
                    for i in range(2):
                        augmented = self.cls_aug(image=img_rgb)
                        aug_img = augmented['image']
                        
                        aug_dst = dst_cls_dir / f"{img_path.stem}_aug{i+1}.jpg"
                        cv2.imwrite(str(aug_dst), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                        total_augmented += 1
                        
                except:
                    continue
        
        # Copy validation and test sets
        for split in ['val', 'test']:
            src_split = self.class_dir / split
            dst_split = self.output / 'augmented' / 'classification' / split
            
            if src_split.exists():
                shutil.copytree(src_split, dst_split, dirs_exist_ok=True)
        
        print(f"\nOriginal images: {total_original}")
        print(f"Augmented images created: {total_augmented}")
        print(f"Total after augmentation: {total_original + total_augmented}")
        print(f"Results saved to: {self.output / 'augmented' / 'classification'}")
    
    def augment_detection(self):
        """Augment detection training data"""
        if not self.det_dir or not self.det_dir.exists():
            print("\nDetection dataset not found!")
            return
        
        print("\n=== AUGMENTING DETECTION DATA ===\n")
        
        # Use the processed detection data if available
        src_img_dir = self.output / 'detection_processed' / 'train' / 'images'
        src_lbl_dir = self.output / 'detection_processed' / 'train' / 'labels'
        
        if not (src_img_dir.exists() and src_lbl_dir.exists()):
            print("Processed detection data not found. Running preprocessing first...")
            self.preprocess_detection()
            src_img_dir = self.output / 'detection_processed' / 'train' / 'images'
            src_lbl_dir = self.output / 'detection_processed' / 'train' / 'labels'
        
        dst_img_dir = self.output / 'augmented' / 'detection' / 'train' / 'images'
        dst_lbl_dir = self.output / 'augmented' / 'detection' / 'train' / 'labels'
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Get images
        images = list(src_img_dir.glob('*.jpg')) + list(src_img_dir.glob('*.png'))
        print(f"Processing {len(images)} training images...")
        
        for img_path in tqdm(images[:20], desc="Augmenting"):  # Process first 20
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Load labels
                label_path = src_lbl_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue
                
                bboxes = []
                class_ids = []
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_c, y_c, bw, bh = map(float, parts[1:])
                            
                            # Convert to [x_min, y_min, x_max, y_max]
                            x_min = x_c - bw/2
                            y_min = y_c - bh/2
                            x_max = x_c + bw/2
                            y_max = y_c + bh/2
                            
                            bboxes.append([x_min, y_min, x_max, y_max])
                            class_ids.append(class_id)
                
                if not bboxes:
                    continue
                
                # Save original
                shutil.copy2(img_path, dst_img_dir / img_path.name)
                shutil.copy2(label_path, dst_lbl_dir / label_path.name)
                
                # Create augmented version
                try:
                    augmented = self.det_aug(
                        image=img_rgb,
                        bboxes=bboxes,
                        class_ids=class_ids
                    )
                    
                    aug_img = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_ids = augmented['class_ids']
                    
                    if aug_bboxes:
                        # Save augmented image
                        aug_img_name = f"{img_path.stem}_aug.jpg"
                        cv2.imwrite(str(dst_img_dir / aug_img_name), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                        
                        # Save augmented labels
                        aug_label_name = f"{img_path.stem}_aug.txt"
                        with open(dst_lbl_dir / aug_label_name, 'w') as f:
                            for bbox, cls_id in zip(aug_bboxes, aug_class_ids):
                                x_min, y_min, x_max, y_max = bbox
                                bw = x_max - x_min
                                bh = y_max - y_min
                                x_c = x_min + bw/2
                                y_c = y_min + bh/2
                                
                                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
                                
                except Exception as e:
                    continue
                    
            except Exception as e:
                continue
        
        # Copy val and test splits without augmentation
        for split in ['val', 'test']:
            src_split_img = self.output / 'detection_processed' / split / 'images'
            src_split_lbl = self.output / 'detection_processed' / split / 'labels'
            dst_split_img = self.output / 'augmented' / 'detection' / split / 'images'
            dst_split_lbl = self.output / 'augmented' / 'detection' / split / 'labels'
            
            if src_split_img.exists():
                shutil.copytree(src_split_img, dst_split_img, dirs_exist_ok=True)
            
            if src_split_lbl.exists():
                shutil.copytree(src_split_lbl, dst_split_lbl, dirs_exist_ok=True)
        
        print(f"\nDetection augmentation complete!")
        print(f"Results saved to: {self.output / 'augmented' / 'detection'}")
    
    def run_pipeline(self):
        """Run complete pipeline"""
        print("="*60)
        print("DATASET PROCESSING PIPELINE")
        print("="*60)
        
        # Step 1: EDA
        print("\n[1/4] Performing EDA...")
        self.analyze_datasets()
        
        # Step 2: Preprocessing
        print("\n[2/4] Preprocessing data...")
        self.preprocess_classification(target_size=224)
        
        if self.det_dir and self.det_dir.exists():
            self.preprocess_detection()
        
        # Step 3: Augmentation
        print("\n[3/4] Applying data augmentation...")
        self.augment_classification()
        
        if self.det_dir and self.det_dir.exists():
            self.augment_detection()
        
        # Step 4: Summary
        print("\n[4/4] Generating summary...")
        self._generate_summary()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
    
    def _generate_summary(self):
        """Generate summary file"""
        summary = {
            'classification': {
                'num_classes': len(self.classes),
                'classes': self.classes
            },
            'output_directories': {}
        }
        
        # Count files
        for item in self.output.iterdir():
            if item.is_dir():
                file_count = 0
                for root, dirs, files in os.walk(item):
                    file_count += len(files)
                summary['output_directories'][item.name] = file_count
        
        # Save summary
        with open(self.output / 'processing_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nProcessing summary saved to: {self.output / 'processing_summary.json'}")
        print(f"\nOutput directories created:")
        
        for dir_name, count in summary['output_directories'].items():
            print(f"  {dir_name}/ - {count} files")
        
        # Show detection info
        if self.det_dir and self.det_dir.exists():
            det_path = self.output / 'detection_processed'
            if det_path.exists():
                print(f"\nDetection dataset processed:")
                for split in ['train', 'val', 'test']:
                    img_dir = det_path / split / 'images'
                    lbl_dir = det_path / split / 'labels'
                    
                    if img_dir.exists():
                        images = len(list(img_dir.glob('*')))
                        labels = len(list(lbl_dir.glob('*.txt'))) if lbl_dir.exists() else 0
                        print(f"  {split}: {images} images, {labels} labels")

# Main execution
if __name__ == "__main__":
    # Your paths
    CLASSIFICATION_PATH = r"C:\Users\Satish Kumar\Desktop\pythoncode - guvi\Smartvision_AI\smartvision_dataset\classification"
    DETECTION_PATH = r"C:\Users\Satish Kumar\Desktop\pythoncode - guvi\Smartvision_AI\smartvision_dataset\detection"
    
    # Initialize processor
    processor = DatasetProcessor(
        classification_dir=CLASSIFICATION_PATH,
        detection_dir=DETECTION_PATH,
        output_dir='./dataset_processing_output'
    )
    
    # Run pipeline
    processor.run_pipeline()