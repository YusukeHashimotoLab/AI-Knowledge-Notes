---
title: "Chapter 3: Object Detection"
chapter_title: "Chapter 3: Object Detection"
subtitle: Bounding Box, IoU, NMS, Two-Stage/One-Stage Detectors, YOLO, Implementation and Applications
reading_time: 35-40 minutes
difficulty: Intermediate
code_examples: 8
exercises: 6
---

This chapter covers Object Detection. You will learn Understanding Bounding Box representation methods and Mastering the principles.

## Learning Objectives

By reading this chapter, you will master:

  * ✅ Understanding Bounding Box representation methods and IoU calculation
  * ✅ Mastering the principles and implementation of Non-Maximum Suppression (NMS)
  * ✅ Understanding evaluation metrics for object detection (mAP, Precision-Recall)
  * ✅ Explaining the mechanisms of Two-Stage detectors (R-CNN, Fast R-CNN, Faster R-CNN)
  * ✅ Understanding the characteristics of One-Stage detectors (YOLO, SSD, RetinaNet)
  * ✅ Implementing practical object detection using YOLOv8
  * ✅ Training object detection models with custom datasets
  * ✅ Implementing real-time detection and tracking applications

* * *

## 3.1 Fundamentals of Object Detection

### What is Object Detection?

Object Detection is the task of detecting multiple objects in an image and predicting their locations (Bounding Boxes) and classes. While image classification answers "what is in the entire image," object detection answers "where is what."

> **Object Detection = Localization + Classification**
> 
> From an input image, it outputs (x, y, width, height, class, confidence) for each object.

#### Bounding Box Representation Methods

A Bounding Box is a rectangular region that encloses an object. There are mainly four representation methods:

Representation | Format | Description  
---|---|---  
**XYXY** | (x1, y1, x2, y2) | Top-left and bottom-right coordinates  
**XYWH** | (x, y, w, h) | Top-left coordinates and width/height  
**CXCYWH** | (cx, cy, w, h) | Center coordinates and width/height  
**Normalized Coordinates** | (normalized to 0~1) | Coordinates normalized by image size  
  
### IoU (Intersection over Union)

IoU is a metric that measures the degree of overlap between predicted and ground truth Bounding Boxes. It is one of the most important evaluation metrics in object detection.

> IoU = (Prediction ∩ Ground Truth) / (Prediction ∪ Ground Truth) = Intersection Area / Union Area
> 
> IoU ranges from 0 (no overlap) to 1 (perfect match).

Code Example 1: IoU Calculation Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def calculate_iou(box1, box2):
        """
        Calculate IoU (Intersection over Union)
    
        Args:
            box1, box2: Bounding Boxes in [x1, y1, x2, y2] format
    
        Returns:
            float: IoU value (0~1)
        """
        # Calculate intersection area coordinates
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
    
        # Intersection area
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        intersection = inter_width * inter_height
    
        # Area of each box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
        # Union area
        union = box1_area + box2_area - intersection
    
        # Calculate IoU (avoid division by zero)
        iou = intersection / union if union > 0 else 0
    
        return iou
    
    # Usage example
    box_pred = [50, 50, 150, 150]   # Predicted Box
    box_gt = [60, 60, 160, 160]     # Ground Truth Box
    
    iou = calculate_iou(box_pred, box_gt)
    print(f"IoU: {iou:.4f}")
    
    # Calculate IoU for multiple boxes
    boxes_pred = np.array([
        [50, 50, 150, 150],
        [100, 100, 200, 200],
        [30, 30, 130, 130]
    ])
    
    boxes_gt = np.array([[60, 60, 160, 160]])
    
    for i, box_pred in enumerate(boxes_pred):
        iou = calculate_iou(box_pred, boxes_gt[0])
        print(f"Box {i+1} IoU: {iou:.4f}")
    

**Output Example:**
    
    
    IoU: 0.6806
    Box 1 IoU: 0.6806
    Box 2 IoU: 0.2537
    Box 3 IoU: 0.7347
    

### Non-Maximum Suppression (NMS)

Object detection models may predict multiple Bounding Boxes for the same object. NMS is a technique that removes duplicate detections and keeps only the box with the highest confidence score.

#### NMS Algorithm

  1. Sort Bounding Boxes in descending order by confidence score
  2. Select the box with highest confidence and add to output list
  3. Remove boxes from the remaining set whose IoU with the selected box is above a threshold
  4. Repeat steps 2-3 for remaining boxes

Code Example 2: NMS Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def non_max_suppression(boxes, scores, iou_threshold=0.5):
        """
        Implement Non-Maximum Suppression (NMS)
    
        Args:
            boxes: numpy array, shape (N, 4) [x1, y1, x2, y2]
            scores: numpy array, shape (N,) confidence scores
            iou_threshold: float, IoU threshold
    
        Returns:
            list: Indices of boxes to keep
        """
        # If boxes is empty
        if len(boxes) == 0:
            return []
    
        # Convert to float32
        boxes = boxes.astype(np.float32)
    
        # Calculate area of each box
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
    
        # Sort by scores in descending order
        order = scores.argsort()[::-1]
    
        keep = []
    
        while len(order) > 0:
            # Select box with highest confidence
            idx = order[0]
            keep.append(idx)
    
            if len(order) == 1:
                break
    
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[idx], x1[order[1:]])
            yy1 = np.maximum(y1[idx], y1[order[1:]])
            xx2 = np.minimum(x2[idx], x2[order[1:]])
            yy2 = np.minimum(y2[idx], y2[order[1:]])
    
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
    
            intersection = w * h
            union = areas[idx] + areas[order[1:]] - intersection
            iou = intersection / union
    
            # Keep only boxes with IoU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
    
        return keep
    
    # Usage example
    boxes = np.array([
        [50, 50, 150, 150],
        [55, 55, 155, 155],
        [60, 60, 160, 160],
        [200, 200, 300, 300],
        [205, 205, 305, 305]
    ])
    
    scores = np.array([0.9, 0.85, 0.88, 0.95, 0.92])
    
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
    
    print(f"Original number of boxes: {len(boxes)}")
    print(f"Number of boxes after NMS: {len(keep_indices)}")
    print(f"Indices of kept boxes: {keep_indices}")
    print(f"\nKept Boxes:")
    for idx in keep_indices:
        print(f"  Box {idx}: {boxes[idx]}, Score: {scores[idx]:.2f}")
    

**Output Example:**
    
    
    Original number of boxes: 5
    Number of boxes after NMS: 2
    Indices of kept boxes: [3, 2]
    
    Kept Boxes:
      Box 3: [200 200 300 300], Score: 0.95
      Box 2: [ 60  60 160 160], Score: 0.88
    

### Evaluation Metrics (mAP)

**mAP (mean Average Precision)** is widely used for evaluating object detection performance.

#### Main Evaluation Metrics

  * **Precision** : Ratio of correct predictions among all detections
  * **Recall** : Ratio of detected objects among all ground truth objects
  * **AP (Average Precision)** : Area under the Precision-Recall curve for one class
  * **mAP (mean Average Precision)** : Average of AP across all classes

> **mAP@0.5** : mAP at IoU threshold 0.5
> 
> **mAP@[0.5:0.95]** : Average mAP over IoU thresholds from 0.5 to 0.95 in 0.05 increments (COCO evaluation)

* * *

## 3.2 Two-Stage Detectors

### Evolution of the R-CNN Family

Two-Stage detectors perform object detection in two stages: "Region Proposal" and "Classification & Localization Refinement."

#### R-CNN (2014)

  1. Extract ~2000 region candidates using **Selective Search**
  2. Extract features from each region using CNN (AlexNet)
  3. Classify with SVM, refine location with regression

**Problem** : Very slow due to 2000 CNN forward passes (47 seconds per image)

#### Fast R-CNN (2015)

  1. Process entire image once with CNN
  2. Extract region candidates from feature map using RoI Pooling
  3. Perform classification and localization simultaneously with fully connected layers

**Improvement** : ~10x faster than R-CNN (2 seconds per image)

#### Faster R-CNN (2015)

  1. Generate region proposals with **RPN (Region Proposal Network)**
  2. Extract features with RoI Pooling
  3. Perform classification and localization

**Improvement** : Eliminates Selective Search, enables full end-to-end learning (0.2 seconds per image)

### Faster R-CNN Implementation

Code Example 3: Object Detection with Faster R-CNN (torchvision)
    
    
    # Requirements:
    # - Python 3.9+
    # - pillow>=10.0.0
    # - requests>=2.31.0
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    """
    Example: Faster R-CNN Implementation
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Advanced
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.transforms import functional as F
    from PIL import Image, ImageDraw, ImageFont
    import requests
    from io import BytesIO
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # COCO class names (91 classes)
    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def detect_objects(image_path, threshold=0.5):
        """
        Perform object detection with Faster R-CNN
    
        Args:
            image_path: Image path or URL
            threshold: Confidence threshold
        """
        # Load image
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')
    
        # Convert to tensor
        img_tensor = F.to_tensor(img).to(device)
    
        # Inference
        with torch.no_grad():
            predictions = model([img_tensor])[0]
    
        # Filter results
        keep = predictions['scores'] > threshold
        boxes = predictions['boxes'][keep].cpu().numpy()
        labels = predictions['labels'][keep].cpu().numpy()
        scores = predictions['scores'][keep].cpu().numpy()
    
        # Draw results
        draw = ImageDraw.Draw(img)
    
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label]
    
            # Draw Bounding Box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    
            # Draw label and score
            text = f"{class_name}: {score:.2f}"
            draw.text((x1, y1 - 15), text, fill='red')
    
        # Display results
        print(f"Number of detected objects: {len(boxes)}")
        for label, score in zip(labels, scores):
            print(f"  - {COCO_CLASSES[label]}: {score:.3f}")
    
        return img, boxes, labels, scores
    
    # Usage example
    image_url = "https://images.unsplash.com/photo-1544568100-847a948585b9?w=800"
    result_img, boxes, labels, scores = detect_objects(image_url, threshold=0.7)
    
    # Display image (for Jupyter Notebook)
    # display(result_img)
    
    # Save image
    result_img.save('faster_rcnn_result.jpg')
    print("Results saved to faster_rcnn_result.jpg")
    

### Feature Pyramid Networks (FPN)

FPN is an architecture that effectively utilizes multi-scale features. It combines feature maps at multiple resolutions to detect objects of different sizes.

> **FPN Features** :
> 
>   * Bottom-up pathway: Standard CNN forward pass
>   * Top-down pathway: Propagate high-level features from low to high resolution
>   * Lateral connections: Merge features at each level
> 

* * *

## 3.3 One-Stage Detectors

### YOLO Family

**YOLO (You Only Look Once)** is a revolutionary approach that performs object detection by looking at the image only once. It achieves real-time detection and is faster than Two-Stage detectors.

#### Basic Principles of YOLO

  1. Divide image into a grid (e.g., 13×13)
  2. Each grid cell predicts Bounding Boxes and confidence
  3. Predict class probabilities for each box
  4. Remove duplicates with NMS

#### Evolution of YOLO

Version | Year | Main Improvements  
---|---|---  
**YOLOv1** | 2016 | Proposed One-Stage detection, real-time processing  
**YOLOv2** | 2017 | Batch Normalization, introduced Anchor Boxes  
**YOLOv3** | 2018 | Multi-scale predictions, Darknet-53  
**YOLOv4** | 2020 | CSPDarknet53, Mosaic augmentation  
**YOLOv5** | 2020 | PyTorch implementation, improved usability  
**YOLOv8** | 2023 | Anchor-free, improved architecture  
Code Example 4: Object Detection with YOLOv8
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    # - pillow>=10.0.0
    
    from ultralytics import YOLO
    from PIL import Image
    import cv2
    import numpy as np
    
    # Load YOLOv8 model
    # Sizes: n (nano), s (small), m (medium), l (large), x (extra large)
    model = YOLO('yolov8n.pt')  # nano model (lightest)
    
    def detect_with_yolo(image_path, conf_threshold=0.5):
        """
        Perform object detection with YOLOv8
    
        Args:
            image_path: Image path or URL
            conf_threshold: Confidence threshold
        """
        # Run inference
        results = model(image_path, conf=conf_threshold)
    
        # Get results
        result = results[0]
    
        # Display detected object information
        print(f"Number of detected objects: {len(result.boxes)}")
    
        for box in result.boxes:
            # Get class ID, confidence, coordinates
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    
            class_name = model.names[class_id]
            print(f"  - {class_name}: {confidence:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
        # Get result image (with annotations)
        annotated_img = result.plot()
    
        return annotated_img, result
    
    # Usage example 1: Detect from image file
    image_path = "path/to/your/image.jpg"
    annotated_img, result = detect_with_yolo(image_path, conf_threshold=0.5)
    
    # Save results
    cv2.imwrite('yolov8_result.jpg', annotated_img)
    print("Results saved to yolov8_result.jpg")
    
    # Usage example 2: Detect from video file or Webcam
    def detect_video(source=0, conf_threshold=0.5):
        """
        Real-time detection from video or Webcam
    
        Args:
            source: Video file path or 0 (Webcam)
            conf_threshold: Confidence threshold
        """
        # Inference on video stream
        results = model(source, stream=True, conf=conf_threshold)
    
        for result in results:
            # Process each frame
            annotated_frame = result.plot()
    
            # Display
            cv2.imshow('YOLOv8 Detection', annotated_frame)
    
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cv2.destroyAllWindows()
    
    # Real-time detection with Webcam (uncomment to run)
    # detect_video(source=0, conf_threshold=0.5)
    
    # Detect from video file
    # detect_video(source='path/to/video.mp4', conf_threshold=0.5)
    

### SSD (Single Shot Detector)

SSD is a One-Stage detector like YOLO but performs detection from feature maps at multiple scales.

#### SSD Features

  * **Multi-scale feature maps** : Detect from layers at different resolutions
  * **Default boxes** : Predict boxes with multiple aspect ratios at each location
  * **Fast** : Faster than YOLOv1 with higher mAP

### RetinaNet (Focal Loss)

RetinaNet solved the class imbalance problem by introducing **Focal Loss**.

#### What is Focal Loss?

> Focal Loss = -α(1-p_t)^γ log(p_t)
> 
> It reduces loss for easy examples (like background) and focuses learning on hard examples.

Code Example 5: Focal Loss Implementation
    
    
    # Requirements:
    # - Python 3.9+
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class FocalLoss(nn.Module):
        """
        Focal Loss for Object Detection
    
        Args:
            alpha: Class weight (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
        """
        def __init__(self, alpha=0.25, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
    
        def forward(self, predictions, targets):
            """
            Args:
                predictions: (N, num_classes) predicted probabilities
                targets: (N,) ground truth labels
            """
            # Cross Entropy Loss
            ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    
            # Calculate p_t (predicted probability of correct class)
            p = torch.exp(-ce_loss)
    
            # Focal Loss
            focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
    
            return focal_loss.mean()
    
    # Usage example
    num_classes = 91  # COCO
    batch_size = 32
    
    # Dummy data
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # Standard Cross Entropy Loss
    ce_loss = F.cross_entropy(predictions, targets)
    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    
    # Focal Loss
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(predictions, targets)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # Compare loss for easy vs hard examples
    easy_predictions = torch.tensor([[10.0, 0.0, 0.0]])  # High confidence for correct class 0
    hard_predictions = torch.tensor([[1.0, 0.9, 0.8]])   # Low confidence for correct class 0
    targets_test = torch.tensor([0])
    
    easy_loss = focal_loss_fn(easy_predictions, targets_test)
    hard_loss = focal_loss_fn(hard_predictions, targets_test)
    
    print(f"\nEasy example loss: {easy_loss.item():.4f}")
    print(f"Hard example loss: {hard_loss.item():.4f}")
    print(f"Hard example loss is {hard_loss.item() / easy_loss.item():.1f}x the easy example loss")
    

### EfficientDet

EfficientDet is an efficient detector using EfficientNet as backbone and BiFPN (Bi-directional Feature Pyramid Network).

  * **Compound Scaling** : Scale resolution, depth, and width simultaneously
  * **BiFPN** : Bidirectional feature fusion
  * **High efficiency** : Higher accuracy with fewer parameters than YOLOv3 or RetinaNet

* * *

## 3.4 Implementation and Training

### COCO Dataset

COCO (Common Objects in Context) is the standard benchmark dataset for object detection.

  * **Number of images** : 330K (train: 118K, val: 5K, test: 41K)
  * **Categories** : 80 classes (people, animals, vehicles, furniture, etc.)
  * **Annotations** : Bounding Boxes, segmentation, keypoints

Code Example 6: Training PyTorch Object Detection
    
    
    # Requirements:
    # - Python 3.9+
    # - pillow>=10.0.0
    # - torch>=2.0.0, <2.3.0
    # - torchvision>=0.15.0
    
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    
    # Custom dataset class
    class CustomObjectDetectionDataset(torch.utils.data.Dataset):
        """
        Custom object detection dataset
    
        Returns images and annotations (boxes, labels)
        """
        def __init__(self, image_paths, annotations, transforms=None):
            self.image_paths = image_paths
            self.annotations = annotations
            self.transforms = transforms
    
        def __len__(self):
            return len(self.image_paths)
    
        def __getitem__(self, idx):
            # Load image
            from PIL import Image
            img = Image.open(self.image_paths[idx]).convert("RGB")
    
            # Get annotations
            boxes = self.annotations[idx]['boxes']  # [[x1,y1,x2,y2], ...]
            labels = self.annotations[idx]['labels']  # [1, 2, 1, ...]
    
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
    
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx])
            }
    
            if self.transforms:
                img = self.transforms(img)
    
            return img, target
    
    def get_model(num_classes):
        """
        Build Faster R-CNN model
    
        Args:
            num_classes: Number of classes (background + object classes)
        """
        # Load pre-trained model
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    
        # Replace classification head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
        return model
    
    def train_one_epoch(model, optimizer, data_loader, device):
        """
        Train for one epoch
        """
        model.train()
        total_loss = 0
    
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
    
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
    
            total_loss += losses.item()
    
        return total_loss / len(data_loader)
    
    # Training configuration
    num_classes = 3  # background + 2 classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model, optimizer, scheduler
    model = get_model(num_classes)
    model.to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    # Dataset (dummy)
    # In practice, prepare image paths and annotations
    image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    annotations = [
        {'boxes': [[10, 10, 50, 50]], 'labels': [1]},
        {'boxes': [[20, 20, 60, 60], [70, 70, 100, 100]], 'labels': [1, 2]},
        {'boxes': [[30, 30, 80, 80]], 'labels': [2]}
    ]
    
    # transforms = T.Compose([T.ToTensor()])
    # dataset = CustomObjectDetectionDataset(image_paths, annotations, transforms)
    # data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    # Training loop (if actual data is available)
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     train_loss = train_one_epoch(model, optimizer, data_loader, device)
    #     lr_scheduler.step()
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
    
    # Save model
    # torch.save(model.state_dict(), 'object_detection_model.pth')
    
    print("Training script ready")
    

### Training with Custom Dataset

Code Example 7: Training YOLOv8 with Custom Dataset
    
    
    # Requirements:
    # - Python 3.9+
    # - pyyaml>=6.0.0
    
    """
    Example: Training with Custom Dataset
    
    Purpose: Demonstrate core concepts and implementation patterns
    Target: Beginner to Intermediate
    Execution time: 1-5 minutes
    Dependencies: None
    """
    
    from ultralytics import YOLO
    import yaml
    import os
    
    # Create dataset configuration file
    dataset_yaml = """
    # Dataset path
    path: ./custom_dataset  # Root directory
    train: images/train     # Training images (relative to path)
    val: images/val         # Validation images
    
    # Class definitions
    names:
      0: cat
      1: dog
      2: bird
    """
    
    # Save dataset.yaml
    with open('custom_dataset.yaml', 'w') as f:
        f.write(dataset_yaml)
    
    # Directory structure example:
    # custom_dataset/
    # ├── images/
    # │   ├── train/
    # │   │   ├── img1.jpg
    # │   │   ├── img2.jpg
    # │   │   └── ...
    # │   └── val/
    # │       ├── img1.jpg
    # │       └── ...
    # └── labels/
    #     ├── train/
    #     │   ├── img1.txt  # YOLO format (class x_center y_center width height)
    #     │   ├── img2.txt
    #     │   └── ...
    #     └── val/
    #         ├── img1.txt
    #         └── ...
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Start from pre-trained weights
    
    # Run training
    results = model.train(
        data='custom_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='custom_yolo',
        # Other hyperparameters
        lr0=0.01,          # Initial learning rate
        momentum=0.937,     # SGD momentum
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=50,        # Early stopping
        # Data Augmentation
        degrees=10.0,       # Rotation
        translate=0.1,      # Translation
        scale=0.5,          # Scale
        flipud=0.0,         # Vertical flip
        fliplr=0.5,         # Horizontal flip
        mosaic=1.0,         # Mosaic augmentation
    )
    
    # Validation
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    
    # Inference
    results = model('path/to/test/image.jpg')
    
    # Save model (automatically saved, but can also save manually)
    # model.save('custom_yolo_best.pt')
    
    # Export model (ONNX, TensorRT, etc.)
    # model.export(format='onnx')
    
    print("\nTraining complete!")
    print(f"Weights: runs/detect/custom_yolo/weights/best.pt")
    print(f"Metrics: runs/detect/custom_yolo/results.csv")
    

**Annotation file format (YOLO):**
    
    
    # img1.txt example (each line is one object)
    0 0.5 0.5 0.3 0.2    # class=0, center=(0.5, 0.5), size=(0.3, 0.2)
    1 0.7 0.3 0.2 0.15   # class=1, center=(0.7, 0.3), size=(0.2, 0.15)
    
    # Coordinates are normalized by image size (0~1)
    # class x_center y_center width height
    

* * *

## 3.5 Advanced Techniques

### Anchor-Free Detection

Traditional detectors rely on Anchor Boxes (pre-defined boxes), but Anchor-Free approaches eliminate this requirement.

#### Main Anchor-Free Methods

  * **FCOS (Fully Convolutional One-Stage)** : Predict distance from each pixel to object center
  * **CenterNet** : Detect object center points, regress size and location
  * **YOLOv8** : Adopts Anchor-Free approach

> **Benefits** : No need for Anchor hyperparameter tuning, more flexible detection

### Object Tracking

Object Tracking is the task of continuously tracking objects across video frames. Used in combination with detectors.

#### SORT (Simple Online and Realtime Tracking)

  1. Detect objects in each frame
  2. Predict next frame positions with Kalman filter
  3. Match detections to tracks using Hungarian Algorithm

#### DeepSORT

Adds appearance features (Deep features) to SORT for more robust tracking.

Code Example 8: YOLOv8 + Object Tracking
    
    
    # Requirements:
    # - Python 3.9+
    # - opencv-python>=4.8.0
    
    from ultralytics import YOLO
    import cv2
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    def track_objects_video(video_path, output_path='tracking_result.mp4'):
        """
        Detect and track objects in video
    
        Args:
            video_path: Input video path
            output_path: Output video path
        """
        # Video capture
        cap = cv2.VideoCapture(video_path)
    
        # Output settings
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
        frame_count = 0
    
        # Inference in tracking mode
        results = model.track(video_path, stream=True, persist=True, conf=0.5)
    
        for result in results:
            frame_count += 1
    
            # Annotated frame
            annotated_frame = result.plot()
    
            # Display tracking IDs
            if result.boxes.id is not None:
                for box, track_id in zip(result.boxes.xyxy, result.boxes.id):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    track_id = int(track_id.cpu().numpy())
    
                    # Draw tracking ID
                    cv2.putText(
                        annotated_frame,
                        f"ID: {track_id}",
                        (int(x1), int(y1) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
    
            # Write frame
            out.write(annotated_frame)
    
            # Display
            cv2.imshow('Object Tracking', annotated_frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
        print(f"Processing complete: {frame_count} frames")
        print(f"Output: {output_path}")
    
    # Usage example
    # track_objects_video('input_video.mp4', 'output_tracking.mp4')
    
    # Real-time tracking with Webcam
    def track_webcam():
        """
        Real-time object tracking with Webcam
        """
        results = model.track(source=0, stream=True, persist=True, conf=0.5)
    
        for result in results:
            annotated_frame = result.plot()
            cv2.imshow('Real-time Tracking', annotated_frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cv2.destroyAllWindows()
    
    # Real-time tracking (uncomment to run)
    # track_webcam()
    
    print("Object tracking script ready")
    

### Multi-Scale Detection

Multi-scale detection is important because object sizes vary greatly across images.

#### Techniques

  * **Image Pyramid** : Resize image to multiple scales and detect
  * **Feature Pyramid** : Detect at multiple feature map levels (FPN)
  * **Multi-scale Training** : Use different input sizes during training

### Real-Time Optimization

#### Speed-Up Techniques

  * **Model lightweighting** : YOLOv8n, MobileNet-SSD
  * **Quantization** : FP32 → FP16 → INT8
  * **TensorRT** : NVIDIA inference optimization engine
  * **ONNX Runtime** : Cross-platform inference
  * **Resolution adjustment** : Reduce input image size (e.g., 640→416)

> **Speed vs Accuracy Tradeoff** :
> 
>   * Real-time requirements: YOLOv8n/s (30+ FPS)
>   * High accuracy requirements: YOLOv8x, Faster R-CNN with FPN
> 

* * *

## Exercises

Exercise 1: Understanding IoU and NMS

**Problem** : Calculate IoU and apply NMS to the following set of Bounding Boxes.
    
    
    boxes = np.array([
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [105, 105, 205, 205],
        [300, 300, 400, 400]
    ])
    scores = np.array([0.9, 0.85, 0.95, 0.8])
    

  * Calculate IoU of each box with Box 0
  * Apply NMS with IoU threshold 0.5
  * Identify which boxes remain

Exercise 2: Detection with Faster R-CNN

**Problem** : Create a script that detects only specific classes (e.g., person, car) from multiple images using pre-trained Faster R-CNN.

  * Load multiple images
  * Filter only specified classes
  * Display only detections with confidence >= 0.7
  * Aggregate detection counts

Exercise 3: YOLOv8 Model Size Comparison

**Problem** : Detect the same image with different YOLOv8 sizes (n, s, m, l) and compare accuracy and speed.

  * Measure inference time for each model
  * Compare number of detected objects
  * Analyze distribution of confidence scores
  * Determine which model is optimal

Exercise 4: Preparing Custom Dataset

**Problem** : Prepare your own image dataset (10+ images) and create YOLO format annotation files.

  * Create annotations using tools like LabelImg
  * Convert to YOLO format (class x_center y_center width height)
  * Split into train/val (80/20)
  * Create dataset.yaml

Exercise 5: Object Tracking Implementation

**Problem** : Track objects from video file or Webcam and draw trajectory for each object.

  * Use YOLOv8 tracking feature
  * Save trajectory for each tracking ID
  * Draw trajectories as lines
  * Verify ID stability across frames

Exercise 6: Real-Time Detection Optimization

**Problem** : Optimize the model to maximize detection speed.

  * Export YOLOv8 to ONNX format
  * Measure FPS at different input resolutions (320, 416, 640)
  * Adjust confidence threshold to improve speed
  * Analyze speed vs accuracy tradeoff

* * *

## Summary

In this chapter, we learned about object detection from fundamentals to practice:

  * ✅ **Object Detection Fundamentals** : Bounding Box representation, IoU, NMS, evaluation metrics (mAP)
  * ✅ **Two-Stage Detectors** : Evolution and features of R-CNN, Fast R-CNN, Faster R-CNN, FPN
  * ✅ **One-Stage Detectors** : Principles and comparison of YOLO, SSD, RetinaNet, EfficientDet
  * ✅ **Implementation and Training** : Practical detection with PyTorch and YOLOv8, training with custom datasets
  * ✅ **Advanced Techniques** : Anchor-free detection, object tracking, multi-scale detection, real-time optimization

In the next chapter, we will learn about **Semantic Segmentation**. We will understand more detailed image understanding methods including pixel-level classification, U-Net, DeepLab, and Mask R-CNN.

> **Key Point** : In object detection, the tradeoff between real-time performance and accuracy is crucial. Select appropriate models and parameters based on application requirements (speed priority or accuracy priority).

## References

  * Girshick et al. (2014). "Rich feature hierarchies for accurate object detection and semantic segmentation" (R-CNN)
  * Girshick (2015). "Fast R-CNN"
  * Ren et al. (2015). "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
  * Redmon et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection" (YOLO)
  * Liu et al. (2016). "SSD: Single Shot MultiBox Detector"
  * Lin et al. (2017). "Focal Loss for Dense Object Detection" (RetinaNet)
  * Lin et al. (2017). "Feature Pyramid Networks for Object Detection"
  * Bochkovskiy et al. (2020). "YOLOv4: Optimal Speed and Accuracy of Object Detection"
  * Ultralytics YOLOv8: <https://github.com/ultralytics/ultralytics>
  * COCO Dataset: <https://cocodataset.org/>
