---
title: "Chapter 5: Computer Vision Applications"
chapter_title: "Chapter 5: Computer Vision Applications"
subtitle: Practical CV Techniques - From Face Recognition to Image Generation
reading_time: 35-40 minutes
difficulty: Intermediate to Advanced
code_examples: 10
exercises: 5
version: 1.0
created_at: "by:"
---

This chapter focuses on practical applications of Computer Vision Applications. You will learn and implement state-of-the-art face detection, Build pose estimation systems, and character recognition using OCR technology.

## Learning Objectives

By reading this chapter, you will master the following:

  * ✅ Understand and implement state-of-the-art face detection and recognition techniques
  * ✅ Build pose estimation systems
  * ✅ Implement character recognition using OCR technology
  * ✅ Apply image generation and editing techniques
  * ✅ Develop end-to-end CV applications
  * ✅ Optimize and deploy models

* * *

## 5.1 Face Recognition and Detection

### Evolution of Face Detection Technology

**Face Detection** is the technology to identify the location of faces in images. Modern main approaches:

Method | Features | Accuracy | Speed  
---|---|---|---  
**Haar Cascade** | Classical, lightweight | Low | Fast  
**HOG + SVM** | Feature-based | Medium | Medium  
**MTCNN** | Multi-stage CNN | High | Medium  
**RetinaFace** | Latest, with landmarks | Highest | GPU required  
  
### Face Detection with MTCNN
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Face Detection with MTCNN
    
    Purpose: Demonstrate data visualization techniques
    Target: Beginner to Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from mtcnn import MTCNN
    
    # Initialize MTCNN model
    detector = MTCNN()
    
    # Load image
    image = cv2.imread('group_photo.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Face detection
    detections = detector.detect_faces(image_rgb)
    
    print(f"=== Detection Results ===")
    print(f"Number of faces detected: {len(detections)}")
    
    # Draw detection results
    image_with_boxes = image_rgb.copy()
    
    for i, detection in enumerate(detections):
        x, y, width, height = detection['box']
        confidence = detection['confidence']
    
        # Bounding box
        cv2.rectangle(image_with_boxes,
                      (x, y), (x + width, y + height),
                      (0, 255, 0), 2)
    
        # Display confidence
        cv2.putText(image_with_boxes,
                    f'{confidence:.2f}',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    
        # Landmarks (eyes, nose, mouth)
        keypoints = detection['keypoints']
        for name, point in keypoints.items():
            cv2.circle(image_with_boxes, point, 2, (255, 0, 0), -1)
    
        print(f"\nFace {i+1}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {width} x {height}")
        print(f"  Confidence: {confidence:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(image_with_boxes)
    axes[1].set_title(f'Detection Results ({len(detections)} faces)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

**Output** :
    
    
    === Detection Results ===
    Number of faces detected: 3
    
    Face 1:
      Position: (120, 80)
      Size: 150 x 180
      Confidence: 0.998
    
    Face 2:
      Position: (350, 95)
      Size: 145 x 175
      Confidence: 0.995
    
    Face 3:
      Position: (570, 110)
      Size: 140 x 170
      Confidence: 0.992
    

### Face Recognition: DeepFace Library

**Face Recognition** is the technology to identify who the detected face is.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - opencv-python>=4.8.0
    
    from deepface import DeepFace
    import cv2
    import matplotlib.pyplot as plt
    
    # Face recognition models: VGG-Face, Facenet, ArcFace, Dlib, OpenFace, etc.
    model_name = 'Facenet'
    
    # Compare faces
    def compare_faces(img1_path, img2_path, model='Facenet'):
        """Compare two face images"""
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model,
            enforce_detection=True
        )
    
        return result
    
    # Execution example
    result = compare_faces('person1_photo1.jpg', 'person1_photo2.jpg')
    
    print("=== Face Recognition Results ===")
    print(f"Same person: {result['verified']}")
    print(f"Distance: {result['distance']:.4f}")
    print(f"Threshold: {result['threshold']:.4f}")
    print(f"Model: {result['model']}")
    
    if result['verified']:
        print("\n✓ Identified as the same person")
    else:
        print("\n✗ Identified as different people")
    
    # Extract face features
    def extract_face_embedding(img_path, model='Facenet'):
        """Extract feature vector from face image"""
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=model,
            enforce_detection=True
        )
        return np.array(embedding[0]['embedding'])
    
    # Get feature vectors
    embedding1 = extract_face_embedding('person1_photo1.jpg')
    embedding2 = extract_face_embedding('person1_photo2.jpg')
    
    print(f"\n=== Feature Vectors ===")
    print(f"Dimensions: {len(embedding1)}")
    print(f"Vector 1 (partial): {embedding1[:10]}")
    print(f"\nCosine similarity: {np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)):.4f}")
    

### Complete Face Recognition System
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    import cv2
    import numpy as np
    from mtcnn import MTCNN
    from deepface import DeepFace
    import os
    import pickle
    
    class FaceRecognitionSystem:
        """Face Recognition System"""
    
        def __init__(self, model_name='Facenet'):
            self.detector = MTCNN()
            self.model_name = model_name
            self.face_database = {}
    
        def register_face(self, name, image_path):
            """Register a face"""
            try:
                # Extract feature vector
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    enforce_detection=True
                )
    
                self.face_database[name] = np.array(embedding[0]['embedding'])
                print(f"✓ Registered {name}")
    
            except Exception as e:
                print(f"✗ Failed to register {name}: {str(e)}")
    
        def recognize_face(self, image_path, threshold=0.6):
            """Recognize a face"""
            try:
                # Extract feature vector
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    enforce_detection=True
                )
    
                query_embedding = np.array(embedding[0]['embedding'])
    
                # Compare with all faces in database
                min_distance = float('inf')
                best_match = None
    
                for name, db_embedding in self.face_database.items():
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(query_embedding - db_embedding)
    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = name
    
                # Threshold judgment
                if min_distance < threshold:
                    return best_match, min_distance
                else:
                    return "Unknown", min_distance
    
            except Exception as e:
                return f"Error: {str(e)}", None
    
        def save_database(self, filepath):
            """Save database"""
            with open(filepath, 'wb') as f:
                pickle.dump(self.face_database, f)
            print(f"✓ Database saved: {filepath}")
    
        def load_database(self, filepath):
            """Load database"""
            with open(filepath, 'rb') as f:
                self.face_database = pickle.load(f)
            print(f"✓ Database loaded: {filepath}")
    
    # System usage example
    system = FaceRecognitionSystem(model_name='Facenet')
    
    # Register faces
    system.register_face("Alice", "alice_1.jpg")
    system.register_face("Bob", "bob_1.jpg")
    system.register_face("Carol", "carol_1.jpg")
    
    # Recognize face
    test_image = "unknown_person.jpg"
    name, distance = system.recognize_face(test_image)
    
    print(f"\n=== Recognition Results ===")
    print(f"Recognized person: {name}")
    print(f"Distance: {distance:.4f}")
    
    # Save database
    system.save_database("face_database.pkl")
    

* * *

## 5.2 Pose Estimation

### What is Pose Estimation?

**Pose Estimation** is the technology to detect human body keypoints (joint positions) from images.
    
    
    ```mermaid
    graph TD
        A[Input Image] --> B[Pose Estimation Model]
        B --> C[Keypoint Detection]
        C --> D[Nose]
        C --> E[Eyes]
        C --> F[Shoulders]
        C --> G[Elbows]
        C --> H[Wrists]
        C --> I[Hips]
        C --> J[Knees]
        C --> K[Ankles]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#e8f5e9
        style F fill:#e8f5e9
        style G fill:#e8f5e9
        style H fill:#e8f5e9
        style I fill:#e8f5e9
        style J fill:#e8f5e9
        style K fill:#e8f5e9
    ```

### Pose Estimation with MediaPipe Pose
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Pose Estimation with MediaPipe Pose
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import mediapipe as mp
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Load image
    image = cv2.imread('person_standing.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Pose estimation
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
    
        results = pose.process(image_rgb)
    
        if results.pose_landmarks:
            print("=== Pose Estimation Results ===")
            print(f"Number of keypoints detected: {len(results.pose_landmarks.landmark)}")
    
            # Display coordinates of key points
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
    
            keypoints_of_interest = {
                0: 'Nose',
                11: 'Left Shoulder',
                12: 'Right Shoulder',
                13: 'Left Elbow',
                14: 'Right Elbow',
                23: 'Left Hip',
                24: 'Right Hip',
                25: 'Left Knee',
                26: 'Right Knee'
            }
    
            for idx, name in keypoints_of_interest.items():
                landmark = landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                visibility = landmark.visibility
    
                print(f"{name}: ({x}, {y}), Visibility: {visibility:.3f}")
    
            # Draw
            annotated_image = image_rgb.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
    
            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
            axes[0].imshow(image_rgb)
            axes[0].set_title('Original Image', fontsize=14)
            axes[0].axis('off')
    
            axes[1].imshow(annotated_image)
            axes[1].set_title('Pose Estimation Results', fontsize=14)
            axes[1].axis('off')
    
            plt.tight_layout()
            plt.show()
    
        else:
            print("No pose detected")
    

### Motion Recognition: Angle Calculation
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    
    import numpy as np
    
    def calculate_angle(point1, point2, point3):
        """Calculate angle from three points (in degrees)"""
        # Calculate vectors
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    
        # Calculate angle
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
        return np.degrees(angle)
    
    def detect_exercise(landmarks, image_shape):
        """Detect exercise motion"""
        h, w = image_shape[:2]
    
        # Get keypoint coordinates
        left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
        left_elbow = (int(landmarks[13].x * w), int(landmarks[13].y * h))
        left_wrist = (int(landmarks[15].x * w), int(landmarks[15].y * h))
    
        left_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
        left_knee = (int(landmarks[25].x * w), int(landmarks[25].y * h))
        left_ankle = (int(landmarks[27].x * w), int(landmarks[27].y * h))
    
        # Calculate angles
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    
        print(f"\n=== Joint Angles ===")
        print(f"Elbow angle: {elbow_angle:.1f}°")
        print(f"Knee angle: {knee_angle:.1f}°")
        print(f"Hip angle: {hip_angle:.1f}°")
    
        # Motion determination
        if knee_angle < 90:
            action = "Squat (down)"
        elif knee_angle > 160:
            action = "Squat (up) or standing"
        else:
            action = "Mid position"
    
        if 30 < elbow_angle < 90:
            arm_action = "Push-up (down)"
        elif elbow_angle > 160:
            arm_action = "Push-up (up)"
        else:
            arm_action = "Arm bending"
    
        return {
            'angles': {
                'elbow': elbow_angle,
                'knee': knee_angle,
                'hip': hip_angle
            },
            'leg_action': action,
            'arm_action': arm_action
        }
    
    # Usage example (continuation from previous code)
    if results.pose_landmarks:
        exercise_info = detect_exercise(results.pose_landmarks.landmark, image.shape)
    
        print(f"\n=== Motion Recognition ===")
        print(f"Lower body: {exercise_info['leg_action']}")
        print(f"Upper body: {exercise_info['arm_action']}")
    

### Real-time Pose Estimation
    
    
    # Requirements:
    # - Python 3.9+
    # - opencv-python>=4.8.0
    
    import cv2
    import mediapipe as mp
    import time
    
    class PoseEstimator:
        """Real-time Pose Estimation"""
    
        def __init__(self):
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
        def process_video(self, video_source=0):
            """Process video for pose estimation"""
            cap = cv2.VideoCapture(video_source)
    
            fps_time = 0
            frame_count = 0
    
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
    
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
                # Pose estimation
                results = self.pose.process(frame_rgb)
    
                # Draw
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
    
                # FPS calculation
                frame_count += 1
                if time.time() - fps_time > 1:
                    fps = frame_count / (time.time() - fps_time)
                    fps_time = time.time()
                    frame_count = 0
    
                    cv2.putText(frame, f'FPS: {fps:.1f}',
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                               1, (0, 255, 0), 2)
    
                cv2.imshow('Pose Estimation', frame)
    
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
    
            cap.release()
            cv2.destroyAllWindows()
    
    # Execution
    estimator = PoseEstimator()
    # For webcam: estimator.process_video(0)
    # For video file: estimator.process_video('video.mp4')
    

* * *

## 5.3 OCR (Optical Character Recognition)

### OCR Technology Overview

**OCR (Optical Character Recognition)** is the technology to recognize text from images and convert it to text.

Library | Features | Japanese Support | Accuracy  
---|---|---|---  
**Tesseract** | Open source, multi-language | ○ | Medium  
**EasyOCR** | Deep learning, easy | ◎ | High  
**PaddleOCR** | Fast, high accuracy | ◎ | Highest  
  
### Character Recognition with EasyOCR
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Character Recognition with EasyOCR
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import easyocr
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Initialize EasyOCR reader (Japanese and English)
    reader = easyocr.Reader(['ja', 'en'], gpu=True)
    
    # Load image
    image_path = 'japanese_text.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Execute OCR
    results = reader.readtext(image_path)
    
    print("=== OCR Results ===")
    print(f"Text regions detected: {len(results)}\n")
    
    # Draw results
    image_with_boxes = image_rgb.copy()
    
    for i, (bbox, text, confidence) in enumerate(results):
        # Bounding box vertices
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
    
        # Draw rectangle
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
    
        # Display text and confidence
        print(f"Region {i+1}:")
        print(f"  Text: {text}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Position: {top_left} - {bottom_right}\n")
    
        # Display text on image
        cv2.putText(image_with_boxes, text,
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(image_with_boxes)
    axes[1].set_title(f'OCR Results ({len(results)} texts)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Combine all text
    full_text = '\n'.join([text for _, text, _ in results])
    print("=== Extracted Full Text ===")
    print(full_text)
    

### High-accuracy Recognition with PaddleOCR
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - opencv-python>=4.8.0
    
    """
    Example: High-accuracy Recognition with PaddleOCR
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    from paddleocr import PaddleOCR
    import cv2
    import matplotlib.pyplot as plt
    
    # Initialize PaddleOCR (Japanese)
    ocr = PaddleOCR(lang='japan', use_angle_cls=True, use_gpu=True)
    
    # Load image
    image_path = 'document.jpg'
    
    # Execute OCR
    result = ocr.ocr(image_path, cls=True)
    
    print("=== PaddleOCR Results ===\n")
    
    # Process results
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_boxes = image_rgb.copy()
    
    for line in result[0]:
        bbox = line[0]
        text = line[1][0]
        confidence = line[1][1]
    
        # Draw bounding box
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(image_with_boxes, [points], True, (0, 255, 0), 2)
    
        print(f"Text: {text}")
        print(f"Confidence: {confidence:.3f}\n")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(image_with_boxes)
    plt.title('PaddleOCR Results', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

### OCR Processing for Receipts and Invoices
    
    
    # Requirements:
    # - Python 3.9+
    # - opencv-python>=4.8.0
    
    import easyocr
    import cv2
    import re
    from datetime import datetime
    
    class ReceiptOCR:
        """Receipt and invoice OCR processing"""
    
        def __init__(self, languages=['ja', 'en']):
            self.reader = easyocr.Reader(languages)
    
        def process_receipt(self, image_path):
            """Process receipt"""
            # Execute OCR
            results = self.reader.readtext(image_path)
    
            # Extract text only
            texts = [text for _, text, _ in results]
    
            # Extract information
            receipt_info = {
                'store_name': self._extract_store_name(texts),
                'date': self._extract_date(texts),
                'total_amount': self._extract_total(texts),
                'items': self._extract_items(texts)
            }
    
            return receipt_info
    
        def _extract_store_name(self, texts):
            """Extract store name (first line)"""
            return texts[0] if texts else "Unknown"
    
        def _extract_date(self, texts):
            """Extract date"""
            date_patterns = [
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
                r'\d{4}年\d{1,2}月\d{1,2}日'
            ]
    
            for text in texts:
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group()
            return "Unknown"
    
        def _extract_total(self, texts):
            """Extract total amount"""
            total_keywords = ['Total', 'Sum', 'TOTAL', '$', '¥']
    
            for i, text in enumerate(texts):
                for keyword in total_keywords:
                    if keyword in text:
                        # Search for amount pattern
                        amount_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)', texts[i])
                        if amount_match:
                            return int(amount_match.group().replace(',', ''))
            return 0
    
        def _extract_items(self, texts):
            """Extract item list"""
            items = []
    
            # Pattern for item lines (item name + amount)
            for text in texts:
                # Detect lines containing amounts
                if re.search(r'\d{1,3}(?:,\d{3})*', text):
                    items.append(text)
    
            return items[:10]  # Maximum 10 items
    
    # Usage example
    ocr_system = ReceiptOCR(languages=['ja', 'en'])
    receipt_info = ocr_system.process_receipt('receipt.jpg')
    
    print("=== Receipt Analysis Results ===")
    print(f"Store name: {receipt_info['store_name']}")
    print(f"Date: {receipt_info['date']}")
    print(f"Total amount: ${receipt_info['total_amount']:,}")
    print(f"\nItem list:")
    for i, item in enumerate(receipt_info['items'], 1):
        print(f"  {i}. {item}")
    

* * *

## 5.4 Image Generation and Editing

### Super-Resolution

**Super-resolution** is the technology to enhance low-resolution images to high-resolution.
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    """
    Example: Super-resolutionis the technology to enhance low-resolution 
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from cv2 import dnn_superres
    
    # Initialize super-resolution model
    sr = dnn_superres.DnnSuperResImpl_create()
    
    # Load ESPCN model (4x upscaling)
    model_path = "ESPCN_x4.pb"
    sr.readModel(model_path)
    sr.setModel("espcn", 4)
    
    # Load low-resolution image
    low_res_image = cv2.imread('low_resolution.jpg')
    
    print(f"=== Super-resolution Processing ===")
    print(f"Original image size: {low_res_image.shape[:2]}")
    
    # Execute super-resolution
    high_res_image = sr.upsample(low_res_image)
    
    print(f"Processed size: {high_res_image.shape[:2]}")
    
    # Comparison with bicubic interpolation
    bicubic_image = cv2.resize(low_res_image,
                               (low_res_image.shape[1] * 4, low_res_image.shape[0] * 4),
                               interpolation=cv2.INTER_CUBIC)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image (Low Resolution)', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(bicubic_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Bicubic Interpolation', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(high_res_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Super-resolution (ESPCN)', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Image quality evaluation
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    # If reference image exists
    if 'ground_truth.jpg' in os.listdir():
        gt_image = cv2.imread('ground_truth.jpg')
    
        psnr_bicubic = psnr(gt_image, bicubic_image)
        psnr_sr = psnr(gt_image, high_res_image)
    
        ssim_bicubic = ssim(gt_image, bicubic_image, multichannel=True)
        ssim_sr = ssim(gt_image, high_res_image, multichannel=True)
    
        print(f"\n=== Image Quality Evaluation ===")
        print(f"PSNR - Bicubic: {psnr_bicubic:.2f} dB")
        print(f"PSNR - Super-resolution: {psnr_sr:.2f} dB")
        print(f"SSIM - Bicubic: {ssim_bicubic:.4f}")
        print(f"SSIM - Super-resolution: {ssim_sr:.4f}")
    

### Background Removal
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    # - pillow>=10.0.0
    
    """
    Example: Background Removal
    
    Purpose: Demonstrate data visualization techniques
    Target: Intermediate
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import cv2
    import numpy as np
    from rembg import remove
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Load image
    input_path = 'person.jpg'
    input_image = Image.open(input_path)
    
    print("=== Background Removal Processing ===")
    print(f"Original image size: {input_image.size}")
    
    # Remove background
    output_image = remove(input_image)
    
    print("✓ Background removal complete")
    
    # Convert to NumPy array
    input_array = np.array(input_image)
    output_array = np.array(output_image)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(input_array)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # After background removal (transparent background)
    axes[1].imshow(output_array)
    axes[1].set_title('Background Removed', fontsize=14)
    axes[1].axis('off')
    
    # Composite with new background
    # Create green background
    green_background = np.zeros_like(output_array)
    green_background[:, :, 1] = 255  # Green channel
    green_background[:, :, 3] = 255  # Alpha channel
    
    # Alpha blending
    alpha = output_array[:, :, 3:4] / 255.0
    composited = (output_array[:, :, :3] * alpha +
                  green_background[:, :, :3] * (1 - alpha)).astype(np.uint8)
    
    axes[2].imshow(composited)
    axes[2].set_title('Composite with New Background', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save
    output_image.save('output_no_bg.png')
    print("✓ Result saved: output_no_bg.png")
    

### Neural Style Transfer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - pillow>=10.0.0
    # - tensorflow>=2.13.0, <2.16.0
    
    import tensorflow as tf
    import tensorflow_hub as hub
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    def load_image(image_path, max_dim=512):
        """Load and preprocess image"""
        img = Image.open(image_path)
        img = img.convert('RGB')
    
        # Resize
        scale = max_dim / max(img.size)
        new_size = tuple([int(dim * scale) for dim in img.size])
        img = img.resize(new_size, Image.LANCZOS)
    
        # Convert to NumPy array
        img = np.array(img)
        img = img[np.newaxis, :]
    
        return img
    
    def style_transfer(content_path, style_path):
        """Execute style transfer"""
        print("=== Neural Style Transfer ===")
    
        # Load model
        print("Loading model...")
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    
        # Load images
        content_image = load_image(content_path)
        style_image = load_image(style_path)
    
        print(f"Content image: {content_image.shape}")
        print(f"Style image: {style_image.shape}")
    
        # Execute style transfer
        print("Executing style transfer...")
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    
        return content_image[0], style_image[0], stylized_image.numpy()[0]
    
    # Execution
    content_img, style_img, stylized_img = style_transfer(
        'content.jpg',  # Content image
        'style.jpg'     # Style image (e.g., Van Gogh painting)
    )
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(content_img.astype(np.uint8))
    axes[0].set_title('Content Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(style_img.astype(np.uint8))
    axes[1].set_title('Style Image', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(stylized_img)
    axes[2].set_title('Style Transfer Result', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Style transfer complete")
    

* * *

## 5.5 End-to-End Project

### Multi-task CV Application
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    import cv2
    import numpy as np
    from mtcnn import MTCNN
    import mediapipe as mp
    import easyocr
    from deepface import DeepFace
    
    class MultiTaskCVSystem:
        """Multi-task Computer Vision System"""
    
        def __init__(self):
            # Initialize each module
            self.face_detector = MTCNN()
            self.pose_estimator = mp.solutions.pose.Pose()
            self.ocr_reader = easyocr.Reader(['ja', 'en'])
            self.mp_drawing = mp.solutions.drawing_utils
    
            print("✓ Multi-task CV system initialized")
    
        def process_image(self, image_path, tasks=['face', 'pose', 'ocr']):
            """Process image"""
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
            results = {}
    
            # Face detection
            if 'face' in tasks:
                print("Executing face detection...")
                faces = self.face_detector.detect_faces(image_rgb)
                results['faces'] = faces
                print(f"  ✓ Detected {len(faces)} face(s)")
    
            # Pose estimation
            if 'pose' in tasks:
                print("Executing pose estimation...")
                pose_results = self.pose_estimator.process(image_rgb)
                results['pose'] = pose_results
                if pose_results.pose_landmarks:
                    print(f"  ✓ Pose detected")
                else:
                    print(f"  ✗ No pose detected")
    
            # OCR
            if 'ocr' in tasks:
                print("Executing OCR...")
                ocr_results = self.ocr_reader.readtext(image_path)
                results['ocr'] = ocr_results
                print(f"  ✓ Detected {len(ocr_results)} text region(s)")
    
            return image_rgb, results
    
        def visualize_results(self, image, results):
            """Visualize results"""
            output = image.copy()
    
            # Draw face detection results
            if 'faces' in results:
                for face in results['faces']:
                    x, y, w, h = face['box']
                    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(output, 'Face', (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
            # Draw pose estimation results
            if 'pose' in results and results['pose'].pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    output,
                    results['pose'].pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
    
            # Draw OCR results
            if 'ocr' in results:
                for bbox, text, conf in results['ocr']:
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    cv2.rectangle(output, top_left, bottom_right, (255, 0, 0), 2)
    
            return output
    
    # Usage example
    system = MultiTaskCVSystem()
    
    # Process image
    image, results = system.process_image('test_image.jpg',
                                          tasks=['face', 'pose', 'ocr'])
    
    # Visualize results
    output_image = system.visualize_results(image, results)
    
    # Display
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(output_image)
    axes[1].set_title('Processing Results (Face, Pose, Text)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

Due to character limit, I'll need to continue the translation in my next response. The file is very large (2643 lines). Let me create a comprehensive solution using a better approach.

### Model Optimization and Deployment

#### ONNX Conversion
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    import torch
    import torch.onnx
    import onnxruntime as ort
    import numpy as np
    
    # Convert PyTorch model to ONNX
    def convert_to_onnx(model, input_shape, output_path):
        """Convert PyTorch model to ONNX"""
        print("=== ONNX Conversion ===")
    
        # Dummy input
        dummy_input = torch.randn(input_shape)
    
        # Export to ONNX format
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
    
        print(f"✓ ONNX model saved: {output_path}")
    
    # ONNX model inference
    class ONNXInference:
        """ONNX Runtime Inference"""
    
        def __init__(self, model_path):
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
    
            print(f"✓ ONNX model loaded: {model_path}")
    
        def predict(self, input_data):
            """Execute inference"""
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            return outputs[0]
    
    # Speed benchmark
    import time
    
    def benchmark_model(model, input_data, num_iterations=100):
        """Measure model inference speed"""
        print(f"\n=== Benchmark ({num_iterations} iterations) ===")
    
        # Warmup
        for _ in range(10):
            _ = model.predict(input_data)
    
        # Measurement
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model.predict(input_data)
        end_time = time.time()
    
        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time
    
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"FPS: {fps:.1f}")
    
        return avg_time, fps
    
    # Usage example
    # Assuming PyTorch model
    # model = YourPyTorchModel()
    # convert_to_onnx(model, (1, 3, 224, 224), 'model.onnx')
    
    # ONNX inference
    onnx_model = ONNXInference('model.onnx')
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    avg_time, fps = benchmark_model(onnx_model, test_input)
    

### Edge Device Optimization
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - tensorflow>=2.13.0, <2.16.0
    
    import tensorflow as tf
    import numpy as np
    
    class ModelOptimizer:
        """Model Optimization Tool"""
    
        @staticmethod
        def quantize_model(model_path, output_path):
            """Reduce model size with quantization (INT8)"""
            print("=== Model Quantization ===")
    
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
            # Quantization settings
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
    
            # Convert
            tflite_model = converter.convert()
    
            # Save
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
    
            # Size comparison
            import os
            original_size = os.path.getsize(model_path) / (1024 * 1024)
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    
            print(f"Original size: {original_size:.2f} MB")
            print(f"Quantized size: {quantized_size:.2f} MB")
            print(f"Compression rate: {(1 - quantized_size/original_size)*100:.1f}%")
    
            return output_path
    
        @staticmethod
        def prune_model(model, target_sparsity=0.5):
            """Remove unnecessary weights with pruning"""
            import tensorflow_model_optimization as tfmot
    
            print(f"=== Model Pruning (Target: {target_sparsity*100}% sparse) ===")
    
            # Pruning settings
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=target_sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }
    
            # Apply pruning
            model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
                model, **pruning_params
            )
    
            print("✓ Pruning settings applied")
    
            return model_for_pruning
    
    # Device-specific optimization strategies
    optimization_strategies = {
        'Mobile': {
            'quantization': 'INT8',
            'target_size': '< 10 MB',
            'target_fps': '> 30',
            'framework': 'TFLite'
        },
        'Edge (Raspberry Pi)': {
            'quantization': 'INT8',
            'target_size': '< 50 MB',
            'target_fps': '> 10',
            'framework': 'TFLite or ONNX'
        },
        'Cloud': {
            'quantization': 'FP16 or FP32',
            'target_size': 'Any',
            'target_fps': '> 100',
            'framework': 'TensorRT or ONNX'
        }
    }
    
    print("\n=== Device-specific Optimization Strategies ===")
    for device, strategy in optimization_strategies.items():
        print(f"\n{device}:")
        for key, value in strategy.items():
            print(f"  {key}: {value}")
    

* * *

## 5.6 Chapter Summary

### What We Learned

  1. **Face Recognition and Detection**

     * High-accuracy face detection with MTCNN and RetinaFace
     * Building face recognition systems using DeepFace
     * Face database management and matching
  2. **Pose Estimation**

     * Keypoint detection with MediaPipe Pose
     * Joint angle calculation and motion recognition
     * Real-time pose estimation implementation
  3. **OCR Technology**

     * Character recognition with EasyOCR and PaddleOCR
     * Multi-language support including Japanese
     * Structured data extraction from receipts and documents
  4. **Image Generation and Editing**

     * Image quality improvement with super-resolution
     * Background removal and alpha blending
     * Artistic transformation with Neural Style Transfer
  5. **End-to-End Development**

     * Multi-task CV system integration
     * ONNX conversion and model optimization
     * Deployment to edge devices

### Key Points for Practical Application

Task | Recommended Models | Considerations  
---|---|---  
**Face Detection** | MTCNN, RetinaFace | Privacy protection, consent  
**Face Recognition** | FaceNet, ArcFace | Security, misrecognition risks  
**Pose Estimation** | MediaPipe, OpenPose | Lighting and occlusion handling  
**OCR** | PaddleOCR, EasyOCR | Font and layout diversity  
**Image Generation** | GANs, Diffusion | Ethical use, copyright  
  
### Next Steps

To further your learning:

  * 3D reconstruction and SLAM technology
  * Video analysis and object tracking
  * CV for autonomous driving
  * Medical image analysis
  * Generative AI (Stable Diffusion, DALL-E)

* * *

## Exercises

### Exercise 1 (Difficulty: medium)

Explain the differences between MTCNN and RetinaFace, and describe which situations each should be used in.

Sample Answer

**Answer** :

**MTCNN (Multi-task Cascaded Convolutional Networks)** :

  * Architecture: 3-stage cascaded CNN (P-Net, R-Net, O-Net)
  * Function: Face detection + 5-point landmarks (eyes, nose, mouth corners)
  * Speed: Medium (CPU compatible)
  * Accuracy: High (especially strong with small faces)

**RetinaFace** :

  * Architecture: Single-stage detector (RetinaNet-based)
  * Function: Face detection + 5-point landmarks + dense 3D face mesh
  * Speed: GPU required, somewhat slow
  * Accuracy: Highest (strong against occlusion and angle variations)

**Usage Guidelines** :

Scenario | Recommendation | Reason  
---|---|---  
Real-time processing (CPU) | MTCNN | Lightweight and fast  
High accuracy required | RetinaFace | Latest technology, high robustness  
Small face detection | MTCNN | Multi-scale support  
Face orientation estimation needed | RetinaFace | Provides 3D information  
Mobile devices | MTCNN | Low computational cost  
  
### Exercise 2 (Difficulty: medium)

Implement a squat form checking system using MediaPipe Pose. Include functionality to count when the knee angle goes below 90 degrees and determine whether the form is correct.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    import cv2
    import mediapipe as mp
    import numpy as np
    
    class SquatFormChecker:
        """Squat Form Checker"""
    
        def __init__(self):
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
            self.squat_count = 0
            self.is_down = False
            self.form_feedback = []
    
        def calculate_angle(self, p1, p2, p3):
            """Calculate angle from 3 points"""
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
    
        def check_form(self, landmarks, h, w):
            """Check form"""
            # Get keypoints
            left_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
            left_knee = (int(landmarks[25].x * w), int(landmarks[25].y * h))
            left_ankle = (int(landmarks[27].x * w), int(landmarks[27].y * h))
    
            left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
    
            # Knee angle
            knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
    
            # Hip-shoulder angle (posture check)
            hip_shoulder_vertical = abs(left_hip[0] - left_shoulder[0])
    
            # Form determination
            self.form_feedback = []
    
            # Knee angle check
            if knee_angle < 90:
                self.form_feedback.append("✓ Sufficient depth")
                if not self.is_down:
                    self.is_down = True
            else:
                if self.is_down and knee_angle > 160:
                    self.squat_count += 1
                    self.is_down = False
    
            # Posture check
            if hip_shoulder_vertical > 50:
                self.form_feedback.append("⚠ Upper body leaning too far forward")
            else:
                self.form_feedback.append("✓ Good upper body posture")
    
            # Knee position check (not going past toes)
            if left_knee[0] > left_ankle[0] + 20:
                self.form_feedback.append("⚠ Knees going past toes")
            else:
                self.form_feedback.append("✓ Good knee position")
    
            return knee_angle
    
        def process_frame(self, frame):
            """Process frame"""
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
    
            if results.pose_landmarks:
                # Draw
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
    
                # Form check
                h, w, _ = frame.shape
                knee_angle = self.check_form(
                    results.pose_landmarks.landmark, h, w
                )
    
                # Display information
                cv2.putText(frame, f'Count: {self.squat_count}',
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)
    
                cv2.putText(frame, f'Knee Angle: {knee_angle:.1f}',
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 0), 2)
    
                # Display feedback
                y_offset = 120
                for feedback in self.form_feedback:
                    cv2.putText(frame, feedback,
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 255), 2)
                    y_offset += 30
    
            return frame
    
    # Usage example
    checker = SquatFormChecker()
    cap = cv2.VideoCapture('squat_video.mp4')
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        frame = checker.process_frame(frame)
        cv2.imshow('Squat Form Checker', frame)
    
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== Final Results ===")
    print(f"Total squat count: {checker.squat_count}")
    

### Exercise 3 (Difficulty: hard)

Process the same image using multiple OCR libraries (Tesseract, EasyOCR, PaddleOCR) and compare their accuracy and speed. Use an image containing Japanese text.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - opencv-python>=4.8.0
    
    import time
    import cv2
    import pytesseract
    import easyocr
    from paddleocr import PaddleOCR
    import matplotlib.pyplot as plt
    
    class OCRComparison:
        """OCR Library Comparison"""
    
        def __init__(self, image_path):
            self.image_path = image_path
            self.image = cv2.imread(image_path)
            self.results = {}
    
        def test_tesseract(self):
            """Tesseract OCR"""
            print("=== Tesseract OCR ===")
            start = time.time()
    
            # Japanese + English settings
            text = pytesseract.image_to_string(
                self.image,
                lang='jpn+eng'
            )
    
            elapsed = time.time() - start
    
            self.results['Tesseract'] = {
                'text': text,
                'time': elapsed,
                'char_count': len(text)
            }
    
            print(f"Processing time: {elapsed:.3f}s")
            print(f"Character count: {len(text)}")
            print(f"Text:\n{text}\n")
    
        def test_easyocr(self):
            """EasyOCR"""
            print("=== EasyOCR ===")
            start = time.time()
    
            reader = easyocr.Reader(['ja', 'en'], gpu=True)
            results = reader.readtext(self.image_path)
    
            elapsed = time.time() - start
    
            text = '\n'.join([item[1] for item in results])
    
            self.results['EasyOCR'] = {
                'text': text,
                'time': elapsed,
                'char_count': len(text),
                'regions': len(results)
            }
    
            print(f"Processing time: {elapsed:.3f}s")
            print(f"Detected regions: {len(results)}")
            print(f"Character count: {len(text)}")
            print(f"Text:\n{text}\n")
    
        def test_paddleocr(self):
            """PaddleOCR"""
            print("=== PaddleOCR ===")
            start = time.time()
    
            ocr = PaddleOCR(lang='japan', use_angle_cls=True)
            results = ocr.ocr(self.image_path, cls=True)
    
            elapsed = time.time() - start
    
            text = '\n'.join([line[1][0] for line in results[0]])
    
            self.results['PaddleOCR'] = {
                'text': text,
                'time': elapsed,
                'char_count': len(text),
                'regions': len(results[0])
            }
    
            print(f"Processing time: {elapsed:.3f}s")
            print(f"Detected regions: {len(results[0])}")
            print(f"Character count: {len(text)}")
            print(f"Text:\n{text}\n")
    
        def compare_all(self):
            """Compare all OCR"""
            self.test_tesseract()
            self.test_easyocr()
            self.test_paddleocr()
    
            # Display comparison results
            print("\n=== Comparison Results ===")
            print(f"{'Library':<15} {'Time(s)':<15} {'Chars':<10} {'Regions':<10}")
            print("-" * 50)
    
            for name, result in self.results.items():
                regions = result.get('regions', 'N/A')
                print(f"{name:<15} {result['time']:<15.3f} {result['char_count']:<10} {regions:<10}")
    
            # Visualize
            self.visualize_comparison()
    
        def visualize_comparison(self):
            """Visualize comparison results"""
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
            # Processing time comparison
            names = list(self.results.keys())
            times = [self.results[name]['time'] for name in names]
    
            axes[0].bar(names, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0].set_ylabel('Processing Time (seconds)')
            axes[0].set_title('Processing Time Comparison', fontsize=14)
            axes[0].grid(True, alpha=0.3)
    
            # Character count comparison
            char_counts = [self.results[name]['char_count'] for name in names]
    
            axes[1].bar(names, char_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[1].set_ylabel('Character Count')
            axes[1].set_title('Character Count Comparison', fontsize=14)
            axes[1].grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
    # Execute
    comparison = OCRComparison('japanese_document.jpg')
    comparison.compare_all()
    

**Expected Output** :
    
    
    === Comparison Results ===
    Library         Time(s)         Chars      Regions
    --------------------------------------------------
    Tesseract       2.341          156        N/A
    EasyOCR         4.567          162        12
    PaddleOCR       1.823          165        15
    
    Conclusion:
    - Speed: PaddleOCR > Tesseract > EasyOCR
    - Accuracy: PaddleOCR ≈ EasyOCR > Tesseract (for Japanese)
    - Recommendation: PaddleOCR for high accuracy, Tesseract for balanced approach
    

### Exercise 4 (Difficulty: hard)

Convert a CNN model trained in PyTorch to ONNX format and compare inference speeds. Additionally, apply quantization and evaluate the trade-off between accuracy and speed.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - matplotlib>=3.7.0
    # - numpy>=1.24.0, <2.0.0
    # - torch>=2.0.0, <2.3.0
    
    """
    Example: Convert a CNN model trained in PyTorch to ONNX format and co
    
    Purpose: Demonstrate data visualization techniques
    Target: Advanced
    Execution time: 2-5 seconds
    Dependencies: None
    """
    
    import torch
    import torch.nn as nn
    import torch.onnx
    import onnxruntime as ort
    import numpy as np
    import time
    
    # Sample CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 56 * 56, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
    
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 56 * 56)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Prepare model
    model = SimpleCNN()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("=== PyTorch Model ===")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
    
    # 1. Measure PyTorch inference speed
    def benchmark_pytorch(model, input_data, iterations=100):
        """Benchmark PyTorch model"""
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(input_data)
    
            # Measure
            start = time.time()
            for _ in range(iterations):
                _ = model(input_data)
            elapsed = time.time() - start
    
        return elapsed / iterations
    
    pytorch_time = benchmark_pytorch(model, dummy_input)
    print(f"PyTorch inference time: {pytorch_time*1000:.2f} ms")
    
    # 2. ONNX conversion
    onnx_path = 'model.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )
    print(f"\n✓ ONNX model saved: {onnx_path}")
    
    # 3. ONNX Runtime inference
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    def benchmark_onnx(session, input_data, iterations=100):
        """Benchmark ONNX Runtime"""
        # Warmup
        for _ in range(10):
            _ = session.run([output_name], {input_name: input_data})
    
        # Measure
        start = time.time()
        for _ in range(iterations):
            _ = session.run([output_name], {input_name: input_data})
        elapsed = time.time() - start
    
        return elapsed / iterations
    
    dummy_input_np = dummy_input.numpy()
    onnx_time = benchmark_onnx(ort_session, dummy_input_np)
    print(f"ONNX inference time: {onnx_time*1000:.2f} ms")
    
    # 4. Quantization (Dynamic Quantization)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    pytorch_quantized_time = benchmark_pytorch(quantized_model, dummy_input)
    print(f"\nQuantized PyTorch inference time: {pytorch_quantized_time*1000:.2f} ms")
    
    # 5. Accuracy comparison
    test_input = torch.randn(10, 3, 224, 224)
    
    with torch.no_grad():
        output_original = model(test_input)
        output_quantized = quantized_model(test_input)
    
    # Accuracy difference
    diff = torch.abs(output_original - output_quantized).mean()
    print(f"\nOutput difference from quantization: {diff:.6f}")
    
    # 6. Model size comparison
    import os
    
    def get_model_size(filepath):
        """Get model size"""
        return os.path.getsize(filepath) / (1024 * 1024)
    
    # Save PyTorch models
    torch.save(model.state_dict(), 'model_original.pth')
    torch.save(quantized_model.state_dict(), 'model_quantized.pth')
    
    original_size = get_model_size('model_original.pth')
    quantized_size = get_model_size('model_quantized.pth')
    onnx_size = get_model_size(onnx_path)
    
    print(f"\n=== Model Size Comparison ===")
    print(f"Original: {original_size:.2f} MB")
    print(f"Quantized: {quantized_size:.2f} MB ({quantized_size/original_size*100:.1f}%)")
    print(f"ONNX: {onnx_size:.2f} MB")
    
    # 7. Comprehensive comparison
    print("\n=== Comprehensive Comparison ===")
    results = [
        ('PyTorch (FP32)', pytorch_time*1000, original_size, 1.0),
        ('PyTorch (INT8)', pytorch_quantized_time*1000, quantized_size, diff.item()),
        ('ONNX Runtime', onnx_time*1000, onnx_size, 0.0)
    ]
    
    print(f"{'Model':<20} {'Inference(ms)':<15} {'Size(MB)':<15} {'Error':<10}")
    print("-" * 65)
    for name, latency, size, error in results:
        print(f"{name:<20} {latency:<15.2f} {size:<15.2f} {error:<10.6f}")
    
    # Visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Inference time
    names = ['PyTorch\n(FP32)', 'PyTorch\n(INT8)', 'ONNX']
    times = [r[1] for r in results]
    axes[0].bar(names, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_ylabel('Inference Time (ms)')
    axes[0].set_title('Inference Speed Comparison', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Model size
    sizes = [r[2] for r in results]
    axes[1].bar(names, sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_ylabel('Model Size (MB)')
    axes[1].set_title('Model Size Comparison', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### Exercise 5 (Difficulty: hard)

Extend the multi-task CV system to simultaneously execute (1) face detection, (2) pose estimation, and (3) OCR from real-time video, and create an application that integrates and displays the results.

Sample Answer
    
    
    # Requirements:
    # - Python 3.9+
    # - numpy>=1.24.0, <2.0.0
    # - opencv-python>=4.8.0
    
    import cv2
    import numpy as np
    from mtcnn import MTCNN
    import mediapipe as mp
    import easyocr
    import threading
    import queue
    import time
    
    class RealtimeMultiTaskCV:
        """Real-time Multi-task CV System"""
    
        def __init__(self):
            # Initialize each model
            self.face_detector = MTCNN()
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.ocr_reader = easyocr.Reader(['ja', 'en'], gpu=True)
    
            # Queues to store results
            self.face_queue = queue.Queue(maxsize=1)
            self.pose_queue = queue.Queue(maxsize=1)
            self.ocr_queue = queue.Queue(maxsize=1)
    
            # Latest results
            self.latest_faces = []
            self.latest_pose = None
            self.latest_ocr = []
    
            # Frame skip settings (reduce heavy processing)
            self.face_skip = 5
            self.ocr_skip = 30
            self.frame_count = 0
    
            print("✓ Multi-task CV system initialized")
    
        def detect_faces_async(self, frame):
            """Face detection (asynchronous)"""
            def worker():
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = self.face_detector.detect_faces(rgb)
    
                    if not self.face_queue.full():
                        self.face_queue.put(faces)
                except Exception as e:
                    print(f"Face detection error: {e}")
    
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
    
        def estimate_pose_async(self, frame):
            """Pose estimation (asynchronous)"""
            def worker():
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb)
    
                    if not self.pose_queue.full():
                        self.pose_queue.put(results)
                except Exception as e:
                    print(f"Pose estimation error: {e}")
    
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
    
        def detect_text_async(self, frame):
            """OCR (asynchronous)"""
            def worker():
                try:
                    # Save as temporary file (EasyOCR constraint)
                    temp_path = 'temp_frame.jpg'
                    cv2.imwrite(temp_path, frame)
                    results = self.ocr_reader.readtext(temp_path)
    
                    if not self.ocr_queue.full():
                        self.ocr_queue.put(results)
                except Exception as e:
                    print(f"OCR error: {e}")
    
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
    
        def update_results(self):
            """Get latest results from queues"""
            try:
                if not self.face_queue.empty():
                    self.latest_faces = self.face_queue.get_nowait()
            except queue.Empty:
                pass
    
            try:
                if not self.pose_queue.empty():
                    self.latest_pose = self.pose_queue.get_nowait()
            except queue.Empty:
                pass
    
            try:
                if not self.ocr_queue.empty():
                    self.latest_ocr = self.ocr_queue.get_nowait()
            except queue.Empty:
                pass
    
        def draw_results(self, frame):
            """Draw results"""
            output = frame.copy()
    
            # Face detection results
            for face in self.latest_faces:
                x, y, w, h = face['box']
                confidence = face['confidence']
    
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output, f'Face: {confidence:.2f}',
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
    
                # Landmarks
                for point in face['keypoints'].values():
                    cv2.circle(output, point, 2, (255, 0, 0), -1)
    
            # Pose estimation results
            if self.latest_pose and self.latest_pose.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    output,
                    self.latest_pose.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
    
            # OCR results
            for bbox, text, conf in self.latest_ocr:
                if conf > 0.5:
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
    
                    cv2.rectangle(output, top_left, bottom_right, (255, 0, 0), 2)
                    cv2.putText(output, text[:20],
                               (top_left[0], top_left[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (255, 0, 0), 2)
    
            return output
    
        def add_info_panel(self, frame):
            """Add information panel"""
            h, w = frame.shape[:2]
    
            # Semi-transparent panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
            # Text information
            info = [
                f'Faces: {len(self.latest_faces)}',
                f'Pose: {"Detected" if self.latest_pose and self.latest_pose.pose_landmarks else "None"}',
                f'Text Regions: {len(self.latest_ocr)}',
                f'Frame: {self.frame_count}'
            ]
    
            y_offset = 40
            for line in info:
                cv2.putText(frame, line, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (255, 255, 255), 2)
                y_offset += 30
    
            return frame
    
        def process_video(self, source=0):
            """Process video"""
            cap = cv2.VideoCapture(source)
    
            fps_time = time.time()
            fps = 0
    
            print("Processing started (press 'q' to quit)")
    
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
    
                self.frame_count += 1
    
                # Face detection (frame skip)
                if self.frame_count % self.face_skip == 0:
                    self.detect_faces_async(frame.copy())
    
                # Pose estimation (every frame)
                self.estimate_pose_async(frame.copy())
    
                # OCR (large skip)
                if self.frame_count % self.ocr_skip == 0:
                    self.detect_text_async(frame.copy())
    
                # Update results
                self.update_results()
    
                # Draw
                output = self.draw_results(frame)
                output = self.add_info_panel(output)
    
                # FPS calculation
                if time.time() - fps_time > 1:
                    fps = self.frame_count / (time.time() - fps_time)
                    fps_time = time.time()
                    self.frame_count = 0
    
                cv2.putText(output, f'FPS: {fps:.1f}',
                           (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
    
                cv2.imshow('Multi-Task CV System', output)
    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
            cap.release()
            cv2.destroyAllWindows()
    
            print("\n=== Processing Complete ===")
            print(f"Total frames: {self.frame_count}")
    
    # Execute
    system = RealtimeMultiTaskCV()
    
    # Process from webcam
    system.process_video(0)
    
    # Or video file
    # system.process_video('video.mp4')
    

**System Features** :

  * Ensures real-time performance with asynchronous processing
  * Reduces heavy processing (OCR) through frame skipping
  * Parallel execution using multithreading
  * Integrated visualization

* * *

## References

  1. Zhang, K., et al. (2016). "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks." _IEEE Signal Processing Letters_.
  2. Deng, J., et al. (2020). "RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild." _CVPR_.
  3. Cao, Z., et al. (2017). "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." _CVPR_.
  4. Bazarevsky, V., et al. (2020). "BlazePose: On-device Real-time Body Pose tracking." _arXiv preprint_.
  5. Baek, J., et al. (2019). "Character Region Awareness for Text Detection." _CVPR_.
  6. Gatys, L. A., et al. (2016). "Image Style Transfer Using Convolutional Neural Networks." _CVPR_.
  7. Ledig, C., et al. (2017). "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network." _CVPR_.
  8. Serengil, S. I., & Ozpinar, A. (2020). "LightFace: A Hybrid Deep Face Recognition Framework." _Innovations in Intelligent Systems and Applications Conference_.
