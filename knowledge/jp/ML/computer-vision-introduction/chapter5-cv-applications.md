---
title: 第5章：コンピュータビジョン応用
chapter_title: 第5章：コンピュータビジョン応用
subtitle: 実践的なCV技術 - 顔認識から画像生成まで
reading_time: 35-40分
difficulty: 中級～上級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 顔検出・認識の最新技術を理解し実装できる
  * ✅ 姿勢推定システムを構築できる
  * ✅ OCR技術で文字認識を実装できる
  * ✅ 画像生成・編集技術を応用できる
  * ✅ エンドツーエンドのCVアプリケーションを開発できる
  * ✅ モデルを最適化しデプロイできる

* * *

## 5.1 顔認識・検出

### 顔検出技術の進化

**顔検出（Face Detection）** は、画像から顔の位置を特定する技術です。現代の主要な手法：

手法 | 特徴 | 精度 | 速度  
---|---|---|---  
**Haar Cascade** | 古典的、軽量 | 低 | 高速  
**HOG + SVM** | 特徴ベース | 中 | 中速  
**MTCNN** | 多段階CNN | 高 | 中速  
**RetinaFace** | 最新、ランドマーク付き | 最高 | GPU必須  
  
### MTCNNによる顔検出
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from mtcnn import MTCNN
    
    # MTCNNモデルの初期化
    detector = MTCNN()
    
    # 画像の読み込み
    image = cv2.imread('group_photo.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 顔検出
    detections = detector.detect_faces(image_rgb)
    
    print(f"=== 検出結果 ===")
    print(f"検出された顔の数: {len(detections)}")
    
    # 検出結果の描画
    image_with_boxes = image_rgb.copy()
    
    for i, detection in enumerate(detections):
        x, y, width, height = detection['box']
        confidence = detection['confidence']
    
        # バウンディングボックス
        cv2.rectangle(image_with_boxes,
                      (x, y), (x + width, y + height),
                      (0, 255, 0), 2)
    
        # 信頼度の表示
        cv2.putText(image_with_boxes,
                    f'{confidence:.2f}',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
    
        # ランドマーク（目、鼻、口）
        keypoints = detection['keypoints']
        for name, point in keypoints.items():
            cv2.circle(image_with_boxes, point, 2, (255, 0, 0), -1)
    
        print(f"\n顔 {i+1}:")
        print(f"  位置: ({x}, {y})")
        print(f"  サイズ: {width} x {height}")
        print(f"  信頼度: {confidence:.3f}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('元画像', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(image_with_boxes)
    axes[1].set_title(f'検出結果 ({len(detections)}個の顔)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

**出力** ：
    
    
    === 検出結果 ===
    検出された顔の数: 3
    
    顔 1:
      位置: (120, 80)
      サイズ: 150 x 180
      信頼度: 0.998
    
    顔 2:
      位置: (350, 95)
      サイズ: 145 x 175
      信頼度: 0.995
    
    顔 3:
      位置: (570, 110)
      サイズ: 140 x 170
      信頼度: 0.992
    

### 顔認識: DeepFaceライブラリ

**顔認識（Face Recognition）** は、検出された顔が誰であるかを識別する技術です。
    
    
    from deepface import DeepFace
    import cv2
    import matplotlib.pyplot as plt
    
    # 顔認識モデル: VGG-Face, Facenet, ArcFace, Dlib, OpenFace など
    model_name = 'Facenet'
    
    # 顔の比較
    def compare_faces(img1_path, img2_path, model='Facenet'):
        """2つの顔画像を比較"""
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model,
            enforce_detection=True
        )
    
        return result
    
    # 実行例
    result = compare_faces('person1_photo1.jpg', 'person1_photo2.jpg')
    
    print("=== 顔認識結果 ===")
    print(f"同一人物: {result['verified']}")
    print(f"距離: {result['distance']:.4f}")
    print(f"閾値: {result['threshold']:.4f}")
    print(f"モデル: {result['model']}")
    
    if result['verified']:
        print("\n✓ 同一人物と判定されました")
    else:
        print("\n✗ 別人と判定されました")
    
    # 顔の特徴抽出
    def extract_face_embedding(img_path, model='Facenet'):
        """顔画像から特徴ベクトルを抽出"""
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name=model,
            enforce_detection=True
        )
        return np.array(embedding[0]['embedding'])
    
    # 特徴ベクトルの取得
    embedding1 = extract_face_embedding('person1_photo1.jpg')
    embedding2 = extract_face_embedding('person1_photo2.jpg')
    
    print(f"\n=== 特徴ベクトル ===")
    print(f"次元数: {len(embedding1)}")
    print(f"ベクトル1の一部: {embedding1[:10]}")
    print(f"\nコサイン類似度: {np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)):.4f}")
    

### 完全な顔認識システム
    
    
    import cv2
    import numpy as np
    from mtcnn import MTCNN
    from deepface import DeepFace
    import os
    import pickle
    
    class FaceRecognitionSystem:
        """顔認識システム"""
    
        def __init__(self, model_name='Facenet'):
            self.detector = MTCNN()
            self.model_name = model_name
            self.face_database = {}
    
        def register_face(self, name, image_path):
            """顔を登録"""
            try:
                # 特徴ベクトルを抽出
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    enforce_detection=True
                )
    
                self.face_database[name] = np.array(embedding[0]['embedding'])
                print(f"✓ {name} を登録しました")
    
            except Exception as e:
                print(f"✗ {name} の登録に失敗: {str(e)}")
    
        def recognize_face(self, image_path, threshold=0.6):
            """顔を認識"""
            try:
                # 特徴ベクトルを抽出
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    enforce_detection=True
                )
    
                query_embedding = np.array(embedding[0]['embedding'])
    
                # データベース内の全ての顔と比較
                min_distance = float('inf')
                best_match = None
    
                for name, db_embedding in self.face_database.items():
                    # ユークリッド距離を計算
                    distance = np.linalg.norm(query_embedding - db_embedding)
    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = name
    
                # 閾値判定
                if min_distance < threshold:
                    return best_match, min_distance
                else:
                    return "Unknown", min_distance
    
            except Exception as e:
                return f"Error: {str(e)}", None
    
        def save_database(self, filepath):
            """データベースを保存"""
            with open(filepath, 'wb') as f:
                pickle.dump(self.face_database, f)
            print(f"✓ データベースを保存: {filepath}")
    
        def load_database(self, filepath):
            """データベースを読み込み"""
            with open(filepath, 'rb') as f:
                self.face_database = pickle.load(f)
            print(f"✓ データベースを読み込み: {filepath}")
    
    # システムの使用例
    system = FaceRecognitionSystem(model_name='Facenet')
    
    # 顔を登録
    system.register_face("Alice", "alice_1.jpg")
    system.register_face("Bob", "bob_1.jpg")
    system.register_face("Carol", "carol_1.jpg")
    
    # 顔を認識
    test_image = "unknown_person.jpg"
    name, distance = system.recognize_face(test_image)
    
    print(f"\n=== 認識結果 ===")
    print(f"認識された人物: {name}")
    print(f"距離: {distance:.4f}")
    
    # データベースを保存
    system.save_database("face_database.pkl")
    

* * *

## 5.2 姿勢推定

### 姿勢推定とは

**姿勢推定（Pose Estimation）** は、画像から人体のキーポイント（関節位置）を検出する技術です。
    
    
    ```mermaid
    graph TD
        A[入力画像] --> B[姿勢推定モデル]
        B --> C[キーポイント検出]
        C --> D[鼻]
        C --> E[目]
        C --> F[肩]
        C --> G[肘]
        C --> H[手首]
        C --> I[腰]
        C --> J[膝]
        C --> K[足首]
    
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

### MediaPipe Poseによる姿勢推定
    
    
    import cv2
    import mediapipe as mp
    import numpy as np
    import matplotlib.pyplot as plt
    
    # MediaPipe Poseの初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # 画像の読み込み
    image = cv2.imread('person_standing.jpg')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 姿勢推定
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
    
        results = pose.process(image_rgb)
    
        if results.pose_landmarks:
            print("=== 姿勢推定結果 ===")
            print(f"検出されたキーポイント数: {len(results.pose_landmarks.landmark)}")
    
            # 主要なキーポイントの座標を表示
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
    
            keypoints_of_interest = {
                0: '鼻',
                11: '左肩',
                12: '右肩',
                13: '左肘',
                14: '右肘',
                23: '左腰',
                24: '右腰',
                25: '左膝',
                26: '右膝'
            }
    
            for idx, name in keypoints_of_interest.items():
                landmark = landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                visibility = landmark.visibility
    
                print(f"{name}: ({x}, {y}), 可視性: {visibility:.3f}")
    
            # 描画
            annotated_image = image_rgb.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
    
            # 可視化
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
            axes[0].imshow(image_rgb)
            axes[0].set_title('元画像', fontsize=14)
            axes[0].axis('off')
    
            axes[1].imshow(annotated_image)
            axes[1].set_title('姿勢推定結果', fontsize=14)
            axes[1].axis('off')
    
            plt.tight_layout()
            plt.show()
    
        else:
            print("姿勢が検出されませんでした")
    

### 動作認識: 角度計算
    
    
    import numpy as np
    
    def calculate_angle(point1, point2, point3):
        """3点から角度を計算（度数法）"""
        # ベクトルの計算
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    
        # 角度の計算
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
        return np.degrees(angle)
    
    def detect_exercise(landmarks, image_shape):
        """運動動作を検出"""
        h, w = image_shape[:2]
    
        # キーポイントの座標を取得
        left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
        left_elbow = (int(landmarks[13].x * w), int(landmarks[13].y * h))
        left_wrist = (int(landmarks[15].x * w), int(landmarks[15].y * h))
    
        left_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
        left_knee = (int(landmarks[25].x * w), int(landmarks[25].y * h))
        left_ankle = (int(landmarks[27].x * w), int(landmarks[27].y * h))
    
        # 角度を計算
        elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    
        print(f"\n=== 関節角度 ===")
        print(f"肘の角度: {elbow_angle:.1f}°")
        print(f"膝の角度: {knee_angle:.1f}°")
        print(f"腰の角度: {hip_angle:.1f}°")
    
        # 動作判定
        if knee_angle < 90:
            action = "スクワット（下）"
        elif knee_angle > 160:
            action = "スクワット（上）または立位"
        else:
            action = "中間姿勢"
    
        if 30 < elbow_angle < 90:
            arm_action = "腕立て伏せ（下）"
        elif elbow_angle > 160:
            arm_action = "腕立て伏せ（上）"
        else:
            arm_action = "腕曲げ中"
    
        return {
            'angles': {
                'elbow': elbow_angle,
                'knee': knee_angle,
                'hip': hip_angle
            },
            'leg_action': action,
            'arm_action': arm_action
        }
    
    # 使用例（前のコードの続き）
    if results.pose_landmarks:
        exercise_info = detect_exercise(results.pose_landmarks.landmark, image.shape)
    
        print(f"\n=== 動作認識 ===")
        print(f"下半身: {exercise_info['leg_action']}")
        print(f"上半身: {exercise_info['arm_action']}")
    

### リアルタイム姿勢推定
    
    
    import cv2
    import mediapipe as mp
    import time
    
    class PoseEstimator:
        """リアルタイム姿勢推定"""
    
        def __init__(self):
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
        def process_video(self, video_source=0):
            """ビデオから姿勢推定"""
            cap = cv2.VideoCapture(video_source)
    
            fps_time = 0
            frame_count = 0
    
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
    
                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
                # 姿勢推定
                results = self.pose.process(frame_rgb)
    
                # 描画
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
    
                # FPS計算
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
    
    # 実行
    estimator = PoseEstimator()
    # Webカメラの場合: estimator.process_video(0)
    # 動画ファイルの場合: estimator.process_video('video.mp4')
    

* * *

## 5.3 OCR（光学文字認識）

### OCR技術の概要

**OCR（Optical Character Recognition）** は、画像から文字を認識してテキストに変換する技術です。

ライブラリ | 特徴 | 日本語対応 | 精度  
---|---|---|---  
**Tesseract** | オープンソース、多言語 | ○ | 中  
**EasyOCR** | ディープラーニング、簡単 | ◎ | 高  
**PaddleOCR** | 高速、高精度 | ◎ | 最高  
  
### EasyOCRによる文字認識
    
    
    import easyocr
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    # EasyOCRリーダーの初期化（日本語と英語）
    reader = easyocr.Reader(['ja', 'en'], gpu=True)
    
    # 画像の読み込み
    image_path = 'japanese_text.jpg'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # OCR実行
    results = reader.readtext(image_path)
    
    print("=== OCR結果 ===")
    print(f"検出されたテキスト領域: {len(results)}個\n")
    
    # 結果の描画
    image_with_boxes = image_rgb.copy()
    
    for i, (bbox, text, confidence) in enumerate(results):
        # バウンディングボックスの頂点
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
    
        # 矩形を描画
        cv2.rectangle(image_with_boxes, top_left, bottom_right, (0, 255, 0), 2)
    
        # テキストと信頼度を表示
        print(f"領域 {i+1}:")
        print(f"  テキスト: {text}")
        print(f"  信頼度: {confidence:.3f}")
        print(f"  位置: {top_left} - {bottom_right}\n")
    
        # 画像上にテキストを表示
        cv2.putText(image_with_boxes, text,
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(image_rgb)
    axes[0].set_title('元画像', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(image_with_boxes)
    axes[1].set_title(f'OCR結果 ({len(results)}個のテキスト)', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 全テキストを結合
    full_text = '\n'.join([text for _, text, _ in results])
    print("=== 抽出された全テキスト ===")
    print(full_text)
    

### PaddleOCRによる高精度認識
    
    
    from paddleocr import PaddleOCR
    import cv2
    import matplotlib.pyplot as plt
    
    # PaddleOCRの初期化（日本語）
    ocr = PaddleOCR(lang='japan', use_angle_cls=True, use_gpu=True)
    
    # 画像の読み込み
    image_path = 'document.jpg'
    
    # OCR実行
    result = ocr.ocr(image_path, cls=True)
    
    print("=== PaddleOCR結果 ===\n")
    
    # 結果の処理
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_with_boxes = image_rgb.copy()
    
    for line in result[0]:
        bbox = line[0]
        text = line[1][0]
        confidence = line[1][1]
    
        # バウンディングボックスを描画
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(image_with_boxes, [points], True, (0, 255, 0), 2)
    
        print(f"テキスト: {text}")
        print(f"信頼度: {confidence:.3f}\n")
    
    # 可視化
    plt.figure(figsize=(12, 8))
    plt.imshow(image_with_boxes)
    plt.title('PaddleOCR結果', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    

### レシート・領収書のOCR処理
    
    
    import easyocr
    import cv2
    import re
    from datetime import datetime
    
    class ReceiptOCR:
        """レシート・領収書のOCR処理"""
    
        def __init__(self, languages=['ja', 'en']):
            self.reader = easyocr.Reader(languages)
    
        def process_receipt(self, image_path):
            """レシートを処理"""
            # OCR実行
            results = self.reader.readtext(image_path)
    
            # テキストのみを抽出
            texts = [text for _, text, _ in results]
    
            # 情報を抽出
            receipt_info = {
                'store_name': self._extract_store_name(texts),
                'date': self._extract_date(texts),
                'total_amount': self._extract_total(texts),
                'items': self._extract_items(texts)
            }
    
            return receipt_info
    
        def _extract_store_name(self, texts):
            """店舗名を抽出（最初の行）"""
            return texts[0] if texts else "不明"
    
        def _extract_date(self, texts):
            """日付を抽出"""
            date_patterns = [
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
                r'\d{4}年\d{1,2}月\d{1,2}日'
            ]
    
            for text in texts:
                for pattern in date_patterns:
                    match = re.search(pattern, text)
                    if match:
                        return match.group()
            return "不明"
    
        def _extract_total(self, texts):
            """合計金額を抽出"""
            total_keywords = ['合計', '計', 'TOTAL', '¥']
    
            for i, text in enumerate(texts):
                for keyword in total_keywords:
                    if keyword in text:
                        # 金額パターンを検索
                        amount_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)', texts[i])
                        if amount_match:
                            return int(amount_match.group().replace(',', ''))
            return 0
    
        def _extract_items(self, texts):
            """商品リストを抽出"""
            items = []
    
            # 商品行のパターン（商品名 + 金額）
            for text in texts:
                # 金額を含む行を検出
                if re.search(r'\d{1,3}(?:,\d{3})*', text):
                    items.append(text)
    
            return items[:10]  # 最大10件
    
    # 使用例
    ocr_system = ReceiptOCR(languages=['ja', 'en'])
    receipt_info = ocr_system.process_receipt('receipt.jpg')
    
    print("=== レシート解析結果 ===")
    print(f"店舗名: {receipt_info['store_name']}")
    print(f"日付: {receipt_info['date']}")
    print(f"合計金額: ¥{receipt_info['total_amount']:,}")
    print(f"\n商品リスト:")
    for i, item in enumerate(receipt_info['items'], 1):
        print(f"  {i}. {item}")
    

* * *

## 5.4 画像生成・編集

### 超解像（Super-Resolution）

**超解像** は、低解像度画像を高解像度化する技術です。
    
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from cv2 import dnn_superres
    
    # 超解像モデルの初期化
    sr = dnn_superres.DnnSuperResImpl_create()
    
    # ESPCN モデルを読み込み（4倍拡大）
    model_path = "ESPCN_x4.pb"
    sr.readModel(model_path)
    sr.setModel("espcn", 4)
    
    # 低解像度画像の読み込み
    low_res_image = cv2.imread('low_resolution.jpg')
    
    print(f"=== 超解像処理 ===")
    print(f"元の画像サイズ: {low_res_image.shape[:2]}")
    
    # 超解像実行
    high_res_image = sr.upsample(low_res_image)
    
    print(f"処理後のサイズ: {high_res_image.shape[:2]}")
    
    # バイキュービック補間との比較
    bicubic_image = cv2.resize(low_res_image,
                               (low_res_image.shape[1] * 4, low_res_image.shape[0] * 4),
                               interpolation=cv2.INTER_CUBIC)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('元画像（低解像度）', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(bicubic_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('バイキュービック補間', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(high_res_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title('超解像（ESPCN）', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 画質評価
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    
    # 参照画像がある場合
    if 'ground_truth.jpg' in os.listdir():
        gt_image = cv2.imread('ground_truth.jpg')
    
        psnr_bicubic = psnr(gt_image, bicubic_image)
        psnr_sr = psnr(gt_image, high_res_image)
    
        ssim_bicubic = ssim(gt_image, bicubic_image, multichannel=True)
        ssim_sr = ssim(gt_image, high_res_image, multichannel=True)
    
        print(f"\n=== 画質評価 ===")
        print(f"PSNR - バイキュービック: {psnr_bicubic:.2f} dB")
        print(f"PSNR - 超解像: {psnr_sr:.2f} dB")
        print(f"SSIM - バイキュービック: {ssim_bicubic:.4f}")
        print(f"SSIM - 超解像: {ssim_sr:.4f}")
    

### 背景除去（Background Removal）
    
    
    import cv2
    import numpy as np
    from rembg import remove
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # 画像の読み込み
    input_path = 'person.jpg'
    input_image = Image.open(input_path)
    
    print("=== 背景除去処理 ===")
    print(f"元の画像サイズ: {input_image.size}")
    
    # 背景除去
    output_image = remove(input_image)
    
    print("✓ 背景除去完了")
    
    # NumPy配列に変換
    input_array = np.array(input_image)
    output_array = np.array(output_image)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 元画像
    axes[0].imshow(input_array)
    axes[0].set_title('元画像', fontsize=14)
    axes[0].axis('off')
    
    # 背景除去後（透明背景）
    axes[1].imshow(output_array)
    axes[1].set_title('背景除去後', fontsize=14)
    axes[1].axis('off')
    
    # 新しい背景と合成
    # 緑色の背景を作成
    green_background = np.zeros_like(output_array)
    green_background[:, :, 1] = 255  # 緑チャンネル
    green_background[:, :, 3] = 255  # アルファチャンネル
    
    # アルファブレンディング
    alpha = output_array[:, :, 3:4] / 255.0
    composited = (output_array[:, :, :3] * alpha +
                  green_background[:, :, :3] * (1 - alpha)).astype(np.uint8)
    
    axes[2].imshow(composited)
    axes[2].set_title('新しい背景と合成', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 保存
    output_image.save('output_no_bg.png')
    print("✓ 結果を保存: output_no_bg.png")
    

### スタイル転送（Neural Style Transfer）
    
    
    import tensorflow as tf
    import tensorflow_hub as hub
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    
    def load_image(image_path, max_dim=512):
        """画像を読み込み、前処理"""
        img = Image.open(image_path)
        img = img.convert('RGB')
    
        # リサイズ
        scale = max_dim / max(img.size)
        new_size = tuple([int(dim * scale) for dim in img.size])
        img = img.resize(new_size, Image.LANCZOS)
    
        # NumPy配列に変換
        img = np.array(img)
        img = img[np.newaxis, :]
    
        return img
    
    def style_transfer(content_path, style_path):
        """スタイル転送を実行"""
        print("=== Neural Style Transfer ===")
    
        # モデルの読み込み
        print("モデルを読み込んでいます...")
        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    
        # 画像の読み込み
        content_image = load_image(content_path)
        style_image = load_image(style_path)
    
        print(f"コンテンツ画像: {content_image.shape}")
        print(f"スタイル画像: {style_image.shape}")
    
        # スタイル転送の実行
        print("スタイル転送を実行中...")
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    
        return content_image[0], style_image[0], stylized_image.numpy()[0]
    
    # 実行
    content_img, style_img, stylized_img = style_transfer(
        'content.jpg',  # コンテンツ画像
        'style.jpg'     # スタイル画像（例：ゴッホの絵画）
    )
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(content_img.astype(np.uint8))
    axes[0].set_title('コンテンツ画像', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(style_img.astype(np.uint8))
    axes[1].set_title('スタイル画像', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(stylized_img)
    axes[2].set_title('スタイル転送結果', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✓ スタイル転送完了")
    

* * *

## 5.5 エンドツーエンドプロジェクト

### マルチタスクCVアプリケーション
    
    
    import cv2
    import numpy as np
    from mtcnn import MTCNN
    import mediapipe as mp
    import easyocr
    from deepface import DeepFace
    
    class MultiTaskCVSystem:
        """マルチタスクコンピュータビジョンシステム"""
    
        def __init__(self):
            # 各モジュールの初期化
            self.face_detector = MTCNN()
            self.pose_estimator = mp.solutions.pose.Pose()
            self.ocr_reader = easyocr.Reader(['ja', 'en'])
            self.mp_drawing = mp.solutions.drawing_utils
    
            print("✓ マルチタスクCVシステムを初期化しました")
    
        def process_image(self, image_path, tasks=['face', 'pose', 'ocr']):
            """画像を処理"""
            # 画像の読み込み
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
            results = {}
    
            # 顔検出
            if 'face' in tasks:
                print("顔検出を実行中...")
                faces = self.face_detector.detect_faces(image_rgb)
                results['faces'] = faces
                print(f"  ✓ {len(faces)}個の顔を検出")
    
            # 姿勢推定
            if 'pose' in tasks:
                print("姿勢推定を実行中...")
                pose_results = self.pose_estimator.process(image_rgb)
                results['pose'] = pose_results
                if pose_results.pose_landmarks:
                    print(f"  ✓ 姿勢を検出")
                else:
                    print(f"  ✗ 姿勢が検出されませんでした")
    
            # OCR
            if 'ocr' in tasks:
                print("OCRを実行中...")
                ocr_results = self.ocr_reader.readtext(image_path)
                results['ocr'] = ocr_results
                print(f"  ✓ {len(ocr_results)}個のテキスト領域を検出")
    
            return image_rgb, results
    
        def visualize_results(self, image, results):
            """結果を可視化"""
            output = image.copy()
    
            # 顔検出結果の描画
            if 'faces' in results:
                for face in results['faces']:
                    x, y, w, h = face['box']
                    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(output, 'Face', (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
            # 姿勢推定結果の描画
            if 'pose' in results and results['pose'].pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    output,
                    results['pose'].pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )
    
            # OCR結果の描画
            if 'ocr' in results:
                for bbox, text, conf in results['ocr']:
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    cv2.rectangle(output, top_left, bottom_right, (255, 0, 0), 2)
    
            return output
    
    # 使用例
    system = MultiTaskCVSystem()
    
    # 画像を処理
    image, results = system.process_image('test_image.jpg',
                                          tasks=['face', 'pose', 'ocr'])
    
    # 結果を可視化
    output_image = system.visualize_results(image, results)
    
    # 表示
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('元画像', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(output_image)
    axes[1].set_title('処理結果（顔・姿勢・テキスト）', fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### モデルの最適化とデプロイ

#### ONNXへの変換
    
    
    import torch
    import torch.onnx
    import onnxruntime as ort
    import numpy as np
    
    # PyTorchモデルをONNXに変換
    def convert_to_onnx(model, input_shape, output_path):
        """PyTorchモデルをONNXに変換"""
        print("=== ONNX変換 ===")
    
        # ダミー入力
        dummy_input = torch.randn(input_shape)
    
        # ONNX形式でエクスポート
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
    
        print(f"✓ ONNXモデルを保存: {output_path}")
    
    # ONNXモデルの推論
    class ONNXInference:
        """ONNX Runtime推論"""
    
        def __init__(self, model_path):
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
    
            print(f"✓ ONNXモデルを読み込み: {model_path}")
    
        def predict(self, input_data):
            """推論を実行"""
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            return outputs[0]
    
    # 速度ベンチマーク
    import time
    
    def benchmark_model(model, input_data, num_iterations=100):
        """モデルの推論速度を測定"""
        print(f"\n=== ベンチマーク ({num_iterations}回) ===")
    
        # ウォームアップ
        for _ in range(10):
            _ = model.predict(input_data)
    
        # 測定
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model.predict(input_data)
        end_time = time.time()
    
        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time
    
        print(f"平均推論時間: {avg_time*1000:.2f} ms")
        print(f"FPS: {fps:.1f}")
    
        return avg_time, fps
    
    # 使用例
    # PyTorchモデルを想定
    # model = YourPyTorchModel()
    # convert_to_onnx(model, (1, 3, 224, 224), 'model.onnx')
    
    # ONNX推論
    onnx_model = ONNXInference('model.onnx')
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    avg_time, fps = benchmark_model(onnx_model, test_input)
    

### エッジデバイス最適化
    
    
    import tensorflow as tf
    import numpy as np
    
    class ModelOptimizer:
        """モデル最適化ツール"""
    
        @staticmethod
        def quantize_model(model_path, output_path):
            """量子化（INT8）でモデルサイズを削減"""
            print("=== モデル量子化 ===")
    
            # TFLiteコンバーターの作成
            converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    
            # 量子化設定
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
    
            # 変換
            tflite_model = converter.convert()
    
            # 保存
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
    
            # サイズ比較
            import os
            original_size = os.path.getsize(model_path) / (1024 * 1024)
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    
            print(f"元のサイズ: {original_size:.2f} MB")
            print(f"量子化後: {quantized_size:.2f} MB")
            print(f"圧縮率: {(1 - quantized_size/original_size)*100:.1f}%")
    
            return output_path
    
        @staticmethod
        def prune_model(model, target_sparsity=0.5):
            """プルーニングで不要な重みを削除"""
            import tensorflow_model_optimization as tfmot
    
            print(f"=== モデルプルーニング (目標: {target_sparsity*100}%疎) ===")
    
            # プルーニング設定
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=target_sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }
    
            # プルーニング適用
            model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
                model, **pruning_params
            )
    
            print("✓ プルーニング設定を適用")
    
            return model_for_pruning
    
    # デバイス別の最適化戦略
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
            'target_size': '任意',
            'target_fps': '> 100',
            'framework': 'TensorRT or ONNX'
        }
    }
    
    print("\n=== デバイス別最適化戦略 ===")
    for device, strategy in optimization_strategies.items():
        print(f"\n{device}:")
        for key, value in strategy.items():
            print(f"  {key}: {value}")
    

* * *

## 5.6 本章のまとめ

### 学んだこと

  1. **顔認識・検出**

     * MTCNN、RetinaFaceによる高精度な顔検出
     * DeepFaceを使った顔認識システム構築
     * 顔データベースの管理と照合
  2. **姿勢推定**

     * MediaPipe Poseによるキーポイント検出
     * 関節角度の計算と動作認識
     * リアルタイム姿勢推定の実装
  3. **OCR技術**

     * EasyOCR、PaddleOCRによる文字認識
     * 日本語を含む多言語対応
     * レシート・文書の構造化データ抽出
  4. **画像生成・編集**

     * 超解像による画質向上
     * 背景除去とアルファブレンディング
     * Neural Style Transferによる芸術的変換
  5. **エンドツーエンド開発**

     * マルチタスクCVシステムの統合
     * ONNX変換とモデル最適化
     * エッジデバイスへのデプロイ

### 実用化のポイント

タスク | 推奨モデル | 注意点  
---|---|---  
**顔検出** | MTCNN, RetinaFace | プライバシー保護、同意取得  
**顔認識** | FaceNet, ArcFace | セキュリティ、誤認識リスク  
**姿勢推定** | MediaPipe, OpenPose | 照明・遮蔽への対応  
**OCR** | PaddleOCR, EasyOCR | フォント・レイアウトの多様性  
**画像生成** | GANs, Diffusion | 倫理的使用、著作権  
  
### 次のステップ

さらに学習を進めるために：

  * 3D再構成とSLAM技術
  * ビデオ解析と物体追跡
  * 自動運転のためのCV
  * 医療画像解析
  * 生成AI（Stable Diffusion、DALL-E）

* * *

## 演習問題

### 問題1（難易度：medium）

MTCNNとRetinaFaceの違いを説明し、それぞれをどのような場面で使うべきか述べてください。

解答例

**解答** ：

**MTCNN（Multi-task Cascaded Convolutional Networks）** ：

  * 構造: 3段階のカスケードCNN（P-Net, R-Net, O-Net）
  * 機能: 顔検出 + 5点ランドマーク（目、鼻、口角）
  * 速度: 中速（CPU可）
  * 精度: 高（特に小さい顔に強い）

**RetinaFace** ：

  * 構造: シングルステージ検出器（RetinaNetベース）
  * 機能: 顔検出 + 5点ランドマーク + 密な3D顔メッシュ
  * 速度: GPU必須、やや遅い
  * 精度: 最高（遮蔽や角度変化に強い）

**使い分け** ：

場面 | 推奨 | 理由  
---|---|---  
リアルタイム処理（CPU） | MTCNN | 軽量で高速  
高精度が必須 | RetinaFace | 最新技術、頑健性高  
小さい顔の検出 | MTCNN | マルチスケール対応  
顔の向き推定も必要 | RetinaFace | 3D情報を提供  
モバイルデバイス | MTCNN | 計算コスト低  
  
### 問題2（難易度：medium）

MediaPipe Poseを使って、スクワットのフォームチェックシステムを実装してください。膝の角度が90度以下になったらカウントし、正しいフォームかどうかを判定する機能を含めてください。

解答例
    
    
    import cv2
    import mediapipe as mp
    import numpy as np
    
    class SquatFormChecker:
        """スクワットフォームチェッカー"""
    
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
            """3点から角度を計算"""
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
    
        def check_form(self, landmarks, h, w):
            """フォームをチェック"""
            # キーポイントを取得
            left_hip = (int(landmarks[23].x * w), int(landmarks[23].y * h))
            left_knee = (int(landmarks[25].x * w), int(landmarks[25].y * h))
            left_ankle = (int(landmarks[27].x * w), int(landmarks[27].y * h))
    
            left_shoulder = (int(landmarks[11].x * w), int(landmarks[11].y * h))
    
            # 膝の角度
            knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
    
            # 腰-肩の角度（姿勢チェック）
            hip_shoulder_vertical = abs(left_hip[0] - left_shoulder[0])
    
            # フォーム判定
            self.form_feedback = []
    
            # 膝の角度チェック
            if knee_angle < 90:
                self.form_feedback.append("✓ 十分な深さ")
                if not self.is_down:
                    self.is_down = True
            else:
                if self.is_down and knee_angle > 160:
                    self.squat_count += 1
                    self.is_down = False
    
            # 姿勢チェック
            if hip_shoulder_vertical > 50:
                self.form_feedback.append("⚠ 上体が前傾しすぎ")
            else:
                self.form_feedback.append("✓ 上体の姿勢良好")
    
            # 膝の位置チェック（つま先より前に出ていないか）
            if left_knee[0] > left_ankle[0] + 20:
                self.form_feedback.append("⚠ 膝がつま先より前に出ています")
            else:
                self.form_feedback.append("✓ 膝の位置良好")
    
            return knee_angle
    
        def process_frame(self, frame):
            """フレームを処理"""
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
    
            if results.pose_landmarks:
                # 描画
                self.mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
    
                # フォームチェック
                h, w, _ = frame.shape
                knee_angle = self.check_form(
                    results.pose_landmarks.landmark, h, w
                )
    
                # 情報を表示
                cv2.putText(frame, f'Count: {self.squat_count}',
                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2)
    
                cv2.putText(frame, f'Knee Angle: {knee_angle:.1f}',
                           (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (255, 255, 0), 2)
    
                # フィードバック表示
                y_offset = 120
                for feedback in self.form_feedback:
                    cv2.putText(frame, feedback,
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 255), 2)
                    y_offset += 30
    
            return frame
    
    # 使用例
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
    
    print(f"\n=== 最終結果 ===")
    print(f"総スクワット回数: {checker.squat_count}")
    

### 問題3（難易度：hard）

複数のOCRライブラリ（Tesseract、EasyOCR、PaddleOCR）を使って、同じ画像を処理し、精度と速度を比較してください。日本語を含む画像を使用してください。

解答例
    
    
    import time
    import cv2
    import pytesseract
    import easyocr
    from paddleocr import PaddleOCR
    import matplotlib.pyplot as plt
    
    class OCRComparison:
        """OCRライブラリの比較"""
    
        def __init__(self, image_path):
            self.image_path = image_path
            self.image = cv2.imread(image_path)
            self.results = {}
    
        def test_tesseract(self):
            """Tesseract OCR"""
            print("=== Tesseract OCR ===")
            start = time.time()
    
            # 日本語+英語の設定
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
    
            print(f"処理時間: {elapsed:.3f}秒")
            print(f"抽出文字数: {len(text)}")
            print(f"テキスト:\n{text}\n")
    
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
    
            print(f"処理時間: {elapsed:.3f}秒")
            print(f"検出領域: {len(results)}")
            print(f"抽出文字数: {len(text)}")
            print(f"テキスト:\n{text}\n")
    
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
    
            print(f"処理時間: {elapsed:.3f}秒")
            print(f"検出領域: {len(results[0])}")
            print(f"抽出文字数: {len(text)}")
            print(f"テキスト:\n{text}\n")
    
        def compare_all(self):
            """すべてのOCRを比較"""
            self.test_tesseract()
            self.test_easyocr()
            self.test_paddleocr()
    
            # 比較結果を表示
            print("\n=== 比較結果 ===")
            print(f"{'ライブラリ':<15} {'処理時間(秒)':<15} {'文字数':<10} {'領域数':<10}")
            print("-" * 50)
    
            for name, result in self.results.items():
                regions = result.get('regions', 'N/A')
                print(f"{name:<15} {result['time']:<15.3f} {result['char_count']:<10} {regions:<10}")
    
            # グラフ化
            self.visualize_comparison()
    
        def visualize_comparison(self):
            """比較結果を可視化"""
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
            # 処理時間の比較
            names = list(self.results.keys())
            times = [self.results[name]['time'] for name in names]
    
            axes[0].bar(names, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0].set_ylabel('処理時間（秒）')
            axes[0].set_title('処理時間の比較', fontsize=14)
            axes[0].grid(True, alpha=0.3)
    
            # 抽出文字数の比較
            char_counts = [self.results[name]['char_count'] for name in names]
    
            axes[1].bar(names, char_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[1].set_ylabel('抽出文字数')
            axes[1].set_title('抽出文字数の比較', fontsize=14)
            axes[1].grid(True, alpha=0.3)
    
            plt.tight_layout()
            plt.show()
    
    # 実行
    comparison = OCRComparison('japanese_document.jpg')
    comparison.compare_all()
    

**期待される出力** ：
    
    
    === 比較結果 ===
    ライブラリ      処理時間(秒)    文字数     領域数
    --------------------------------------------------
    Tesseract       2.341          156        N/A
    EasyOCR         4.567          162        12
    PaddleOCR       1.823          165        15
    
    結論:
    - 速度: PaddleOCR > Tesseract > EasyOCR
    - 精度: PaddleOCR ≈ EasyOCR > Tesseract（日本語）
    - 推奨: 高精度が必要ならPaddleOCR、バランス重視ならTesseract
    

### 問題4（難易度：hard）

PyTorchで訓練したCNNモデルをONNX形式に変換し、推論速度を比較してください。さらに量子化も適用し、精度と速度のトレードオフを評価してください。

解答例
    
    
    import torch
    import torch.nn as nn
    import torch.onnx
    import onnxruntime as ort
    import numpy as np
    import time
    
    # サンプルCNNモデル
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
    
    # モデルの準備
    model = SimpleCNN()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print("=== PyTorchモデル ===")
    print(f"パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 1. PyTorchでの推論速度測定
    def benchmark_pytorch(model, input_data, iterations=100):
        """PyTorchモデルのベンチマーク"""
        with torch.no_grad():
            # ウォームアップ
            for _ in range(10):
                _ = model(input_data)
    
            # 測定
            start = time.time()
            for _ in range(iterations):
                _ = model(input_data)
            elapsed = time.time() - start
    
        return elapsed / iterations
    
    pytorch_time = benchmark_pytorch(model, dummy_input)
    print(f"PyTorch推論時間: {pytorch_time*1000:.2f} ms")
    
    # 2. ONNX変換
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
    print(f"\n✓ ONNXモデルを保存: {onnx_path}")
    
    # 3. ONNX Runtime推論
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    def benchmark_onnx(session, input_data, iterations=100):
        """ONNX Runtimeのベンチマーク"""
        # ウォームアップ
        for _ in range(10):
            _ = session.run([output_name], {input_name: input_data})
    
        # 測定
        start = time.time()
        for _ in range(iterations):
            _ = session.run([output_name], {input_name: input_data})
        elapsed = time.time() - start
    
        return elapsed / iterations
    
    dummy_input_np = dummy_input.numpy()
    onnx_time = benchmark_onnx(ort_session, dummy_input_np)
    print(f"ONNX推論時間: {onnx_time*1000:.2f} ms")
    
    # 4. 量子化（Dynamic Quantization）
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    pytorch_quantized_time = benchmark_pytorch(quantized_model, dummy_input)
    print(f"\n量子化PyTorch推論時間: {pytorch_quantized_time*1000:.2f} ms")
    
    # 5. 精度比較
    test_input = torch.randn(10, 3, 224, 224)
    
    with torch.no_grad():
        output_original = model(test_input)
        output_quantized = quantized_model(test_input)
    
    # 精度の差
    diff = torch.abs(output_original - output_quantized).mean()
    print(f"\n量子化による出力の差: {diff:.6f}")
    
    # 6. モデルサイズ比較
    import os
    
    def get_model_size(filepath):
        """モデルサイズを取得"""
        return os.path.getsize(filepath) / (1024 * 1024)
    
    # PyTorchモデルを保存
    torch.save(model.state_dict(), 'model_original.pth')
    torch.save(quantized_model.state_dict(), 'model_quantized.pth')
    
    original_size = get_model_size('model_original.pth')
    quantized_size = get_model_size('model_quantized.pth')
    onnx_size = get_model_size(onnx_path)
    
    print(f"\n=== モデルサイズ比較 ===")
    print(f"オリジナル: {original_size:.2f} MB")
    print(f"量子化: {quantized_size:.2f} MB ({quantized_size/original_size*100:.1f}%)")
    print(f"ONNX: {onnx_size:.2f} MB")
    
    # 7. 総合比較
    print("\n=== 総合比較 ===")
    results = [
        ('PyTorch (FP32)', pytorch_time*1000, original_size, 1.0),
        ('PyTorch (INT8)', pytorch_quantized_time*1000, quantized_size, diff.item()),
        ('ONNX Runtime', onnx_time*1000, onnx_size, 0.0)
    ]
    
    print(f"{'モデル':<20} {'推論時間(ms)':<15} {'サイズ(MB)':<15} {'誤差':<10}")
    print("-" * 65)
    for name, latency, size, error in results:
        print(f"{name:<20} {latency:<15.2f} {size:<15.2f} {error:<10.6f}")
    
    # グラフ化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 推論時間
    names = ['PyTorch\n(FP32)', 'PyTorch\n(INT8)', 'ONNX']
    times = [r[1] for r in results]
    axes[0].bar(names, times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_ylabel('推論時間（ms）')
    axes[0].set_title('推論速度の比較', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # モデルサイズ
    sizes = [r[2] for r in results]
    axes[1].bar(names, sizes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_ylabel('モデルサイズ（MB）')
    axes[1].set_title('モデルサイズの比較', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

### 問題5（難易度：hard）

マルチタスクCVシステムを拡張し、リアルタイム動画から（1）顔検出、（2）姿勢推定、（3）OCRを同時に実行し、結果を統合して表示するアプリケーションを作成してください。

解答例
    
    
    import cv2
    import numpy as np
    from mtcnn import MTCNN
    import mediapipe as mp
    import easyocr
    import threading
    import queue
    import time
    
    class RealtimeMultiTaskCV:
        """リアルタイムマルチタスクCVシステム"""
    
        def __init__(self):
            # 各モデルの初期化
            self.face_detector = MTCNN()
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.ocr_reader = easyocr.Reader(['ja', 'en'], gpu=True)
    
            # 結果を保存するキュー
            self.face_queue = queue.Queue(maxsize=1)
            self.pose_queue = queue.Queue(maxsize=1)
            self.ocr_queue = queue.Queue(maxsize=1)
    
            # 最新の結果
            self.latest_faces = []
            self.latest_pose = None
            self.latest_ocr = []
    
            # フレームスキップ設定（重い処理を間引く）
            self.face_skip = 5
            self.ocr_skip = 30
            self.frame_count = 0
    
            print("✓ マルチタスクCVシステムを初期化")
    
        def detect_faces_async(self, frame):
            """顔検出（非同期）"""
            def worker():
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = self.face_detector.detect_faces(rgb)
    
                    if not self.face_queue.full():
                        self.face_queue.put(faces)
                except Exception as e:
                    print(f"顔検出エラー: {e}")
    
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
    
        def estimate_pose_async(self, frame):
            """姿勢推定（非同期）"""
            def worker():
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.pose.process(rgb)
    
                    if not self.pose_queue.full():
                        self.pose_queue.put(results)
                except Exception as e:
                    print(f"姿勢推定エラー: {e}")
    
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
    
        def detect_text_async(self, frame):
            """OCR（非同期）"""
            def worker():
                try:
                    # 一時ファイルとして保存（EasyOCRの制約）
                    temp_path = 'temp_frame.jpg'
                    cv2.imwrite(temp_path, frame)
                    results = self.ocr_reader.readtext(temp_path)
    
                    if not self.ocr_queue.full():
                        self.ocr_queue.put(results)
                except Exception as e:
                    print(f"OCRエラー: {e}")
    
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
    
        def update_results(self):
            """キューから最新の結果を取得"""
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
            """結果を描画"""
            output = frame.copy()
    
            # 顔検出結果
            for face in self.latest_faces:
                x, y, w, h = face['box']
                confidence = face['confidence']
    
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(output, f'Face: {confidence:.2f}',
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
    
                # ランドマーク
                for point in face['keypoints'].values():
                    cv2.circle(output, point, 2, (255, 0, 0), -1)
    
            # 姿勢推定結果
            if self.latest_pose and self.latest_pose.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    output,
                    self.latest_pose.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
    
            # OCR結果
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
            """情報パネルを追加"""
            h, w = frame.shape[:2]
    
            # 半透明パネル
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
            # テキスト情報
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
            """ビデオを処理"""
            cap = cv2.VideoCapture(source)
    
            fps_time = time.time()
            fps = 0
    
            print("処理を開始（'q'で終了）")
    
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
    
                self.frame_count += 1
    
                # 顔検出（フレームスキップ）
                if self.frame_count % self.face_skip == 0:
                    self.detect_faces_async(frame.copy())
    
                # 姿勢推定（毎フレーム）
                self.estimate_pose_async(frame.copy())
    
                # OCR（大幅にスキップ）
                if self.frame_count % self.ocr_skip == 0:
                    self.detect_text_async(frame.copy())
    
                # 結果を更新
                self.update_results()
    
                # 描画
                output = self.draw_results(frame)
                output = self.add_info_panel(output)
    
                # FPS計算
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
    
            print("\n=== 処理完了 ===")
            print(f"総フレーム数: {self.frame_count}")
    
    # 実行
    system = RealtimeMultiTaskCV()
    
    # Webカメラから処理
    system.process_video(0)
    
    # または動画ファイル
    # system.process_video('video.mp4')
    

**システムの特徴** ：

  * 非同期処理でリアルタイム性を確保
  * 重い処理（OCR）はフレームスキップで軽量化
  * マルチスレッドで並列実行
  * 統合されたビジュアライゼーション

* * *

## 参考文献

  1. Zhang, K., et al. (2016). "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks." _IEEE Signal Processing Letters_.
  2. Deng, J., et al. (2020). "RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild." _CVPR_.
  3. Cao, Z., et al. (2017). "Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields." _CVPR_.
  4. Bazarevsky, V., et al. (2020). "BlazePose: On-device Real-time Body Pose tracking." _arXiv preprint_.
  5. Baek, J., et al. (2019). "Character Region Awareness for Text Detection." _CVPR_.
  6. Gatys, L. A., et al. (2016). "Image Style Transfer Using Convolutional Neural Networks." _CVPR_.
  7. Ledig, C., et al. (2017). "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network." _CVPR_.
  8. Serengil, S. I., & Ozpinar, A. (2020). "LightFace: A Hybrid Deep Face Recognition Framework." _Innovations in Intelligent Systems and Applications Conference_.
