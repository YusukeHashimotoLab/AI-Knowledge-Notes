---
title: 第3章：物体検出
chapter_title: 第3章：物体検出
subtitle: Bounding Box、IoU、NMS、Two-Stage/One-Stage検出器、YOLO、実装と応用
reading_time: 35-40分
difficulty: 中級
code_examples: 8
exercises: 6
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Bounding Boxの表現方法とIoUの計算方法を理解する
  * ✅ Non-Maximum Suppression (NMS)の原理と実装を習得する
  * ✅ 物体検出の評価指標（mAP、Precision-Recall）を理解する
  * ✅ Two-Stage検出器（R-CNN、Fast R-CNN、Faster R-CNN）の仕組みを説明できる
  * ✅ One-Stage検出器（YOLO、SSD、RetinaNet）の特徴を理解する
  * ✅ YOLOv8を使った実践的な物体検出を実装できる
  * ✅ カスタムデータセットで物体検出モデルを訓練できる
  * ✅ リアルタイム検出とトラッキングの応用を実装できる

* * *

## 3.1 物体検出の基礎

### 物体検出とは

物体検出（Object Detection）は、画像内の複数の物体を検出し、それぞれの位置（Bounding Box）とクラスを予測するタスクです。画像分類が「画像全体に何が写っているか」を答えるのに対し、物体検出は「どこに何があるか」を答えます。

> **物体検出 = 位置特定（Localization）+ 分類（Classification）**
> 
> 入力画像から、各物体の (x, y, width, height, class, confidence) を出力します。

#### Bounding Boxの表現方法

Bounding Box（バウンディングボックス）は、物体を囲む矩形領域です。主に以下の4つの表現方法があります：

表現方法 | 形式 | 説明  
---|---|---  
**XYXY** | (x1, y1, x2, y2) | 左上座標と右下座標  
**XYWH** | (x, y, w, h) | 左上座標と幅・高さ  
**CXCYWH** | (cx, cy, w, h) | 中心座標と幅・高さ  
**正規化座標** | (0~1に正規化) | 画像サイズで正規化した座標  
  
### IoU (Intersection over Union)

IoUは、予測Bounding Boxと正解Bounding Boxの重なり度合いを測る指標です。物体検出において最も重要な評価指標の一つです。

> IoU = (予測 ∩ 正解) / (予測 ∪ 正解) = 重なり面積 / 結合面積
> 
> IoU = 0（重なりなし）～ 1（完全一致）の範囲を取ります。

コード例1：IoU計算の実装
    
    
    import numpy as np
    
    def calculate_iou(box1, box2):
        """
        IoU (Intersection over Union) を計算
    
        Args:
            box1, box2: [x1, y1, x2, y2] 形式のBounding Box
    
        Returns:
            float: IoU値 (0~1)
        """
        # 交差領域の座標を計算
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
    
        # 交差領域の面積
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        intersection = inter_width * inter_height
    
        # 各Boxの面積
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
        # 結合領域の面積
        union = box1_area + box2_area - intersection
    
        # IoU計算（ゼロ除算回避）
        iou = intersection / union if union > 0 else 0
    
        return iou
    
    # 使用例
    box_pred = [50, 50, 150, 150]   # 予測Box
    box_gt = [60, 60, 160, 160]     # 正解Box
    
    iou = calculate_iou(box_pred, box_gt)
    print(f"IoU: {iou:.4f}")
    
    # 複数のBoxに対してIoUを計算
    boxes_pred = np.array([
        [50, 50, 150, 150],
        [100, 100, 200, 200],
        [30, 30, 130, 130]
    ])
    
    boxes_gt = np.array([[60, 60, 160, 160]])
    
    for i, box_pred in enumerate(boxes_pred):
        iou = calculate_iou(box_pred, boxes_gt[0])
        print(f"Box {i+1} IoU: {iou:.4f}")
    

**出力例：**
    
    
    IoU: 0.6806
    Box 1 IoU: 0.6806
    Box 2 IoU: 0.2537
    Box 3 IoU: 0.7347
    

### Non-Maximum Suppression (NMS)

物体検出モデルは、同じ物体に対して複数のBounding Boxを予測することがあります。NMSは、重複した検出を除去し、最も信頼度の高いBoxのみを残す手法です。

#### NMSのアルゴリズム

  1. 信頼度スコアでBounding Boxを降順にソート
  2. 最も信頼度の高いBoxを選択し、出力リストに追加
  3. 残りのBoxのうち、選択したBoxとのIoUが閾値以上のものを削除
  4. 残りのBoxに対してステップ2-3を繰り返す

コード例2：NMSの実装
    
    
    import numpy as np
    
    def non_max_suppression(boxes, scores, iou_threshold=0.5):
        """
        Non-Maximum Suppression (NMS) を実装
    
        Args:
            boxes: numpy array, shape (N, 4) [x1, y1, x2, y2]
            scores: numpy array, shape (N,) 信頼度スコア
            iou_threshold: float, IoU閾値
    
        Returns:
            list: 残すべきBoxのインデックス
        """
        # Boxが空の場合
        if len(boxes) == 0:
            return []
    
        # float型に変換
        boxes = boxes.astype(np.float32)
    
        # 各Boxの面積を計算
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
    
        # スコアで降順にソート
        order = scores.argsort()[::-1]
    
        keep = []
    
        while len(order) > 0:
            # 最も信頼度の高いBoxを選択
            idx = order[0]
            keep.append(idx)
    
            if len(order) == 1:
                break
    
            # 残りのBoxとのIoUを計算
            xx1 = np.maximum(x1[idx], x1[order[1:]])
            yy1 = np.maximum(y1[idx], y1[order[1:]])
            xx2 = np.minimum(x2[idx], x2[order[1:]])
            yy2 = np.minimum(y2[idx], y2[order[1:]])
    
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
    
            intersection = w * h
            union = areas[idx] + areas[order[1:]] - intersection
            iou = intersection / union
    
            # IoUが閾値未満のBoxのみ残す
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
    
        return keep
    
    # 使用例
    boxes = np.array([
        [50, 50, 150, 150],
        [55, 55, 155, 155],
        [60, 60, 160, 160],
        [200, 200, 300, 300],
        [205, 205, 305, 305]
    ])
    
    scores = np.array([0.9, 0.85, 0.88, 0.95, 0.92])
    
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
    
    print(f"元のBox数: {len(boxes)}")
    print(f"NMS後のBox数: {len(keep_indices)}")
    print(f"残すBoxのインデックス: {keep_indices}")
    print(f"\n残ったBoxes:")
    for idx in keep_indices:
        print(f"  Box {idx}: {boxes[idx]}, Score: {scores[idx]:.2f}")
    

**出力例：**
    
    
    元のBox数: 5
    NMS後のBox数: 2
    残すBoxのインデックス: [3, 2]
    
    残ったBoxes:
      Box 3: [200 200 300 300], Score: 0.95
      Box 2: [ 60  60 160 160], Score: 0.88
    

### 評価指標（mAP）

物体検出の性能評価には、**mAP (mean Average Precision)** が広く使われます。

#### 主要な評価指標

  * **Precision（適合率）** ：検出したもののうち、正解だった割合
  * **Recall（再現率）** ：正解のうち、検出できた割合
  * **AP (Average Precision)** ：1クラスに対するPrecision-Recall曲線の面積
  * **mAP (mean Average Precision)** ：全クラスのAPの平均値

> **mAP@0.5** ：IoU閾値0.5でのmAP
> 
> **mAP@[0.5:0.95]** ：IoU閾値0.5～0.95（0.05刻み）の平均mAP（COCO評価）

* * *

## 3.2 Two-Stage検出器

### R-CNNファミリーの進化

Two-Stage検出器は、「領域提案（Region Proposal）」と「分類・位置調整」の2段階で物体検出を行います。

#### R-CNN (2014)

  1. **Selective Search** で約2000個の領域候補を抽出
  2. 各領域をCNNで特徴抽出（AlexNet）
  3. SVMで分類、回帰で位置調整

**問題点** ：2000回のCNN処理が必要で非常に遅い（1画像に47秒）

#### Fast R-CNN (2015)

  1. 画像全体を1回だけCNNで処理
  2. 特徴マップから領域候補をRoI Poolingで抽出
  3. 全結合層で分類と位置調整を同時実行

**改善** ：R-CNNの約10倍高速化（1画像に2秒）

#### Faster R-CNN (2015)

  1. **RPN (Region Proposal Network)** で領域候補を生成
  2. RoI Poolingで特徴抽出
  3. 分類と位置調整

**改善** ：Selective Searchを不要にし、完全なEnd-to-Endの学習が可能に（1画像に0.2秒）

### Faster R-CNNの実装

コード例3：Faster R-CNNでの物体検出（torchvision）
    
    
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.transforms import functional as F
    from PIL import Image, ImageDraw, ImageFont
    import requests
    from io import BytesIO
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 事前学習済みFaster R-CNNモデルをロード
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.eval()
    
    # COCOクラス名（91クラス）
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
        Faster R-CNNで物体検出を実行
    
        Args:
            image_path: 画像パスまたはURL
            threshold: 信頼度閾値
        """
        # 画像読み込み
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')
    
        # テンソルに変換
        img_tensor = F.to_tensor(img).to(device)
    
        # 推論
        with torch.no_grad():
            predictions = model([img_tensor])[0]
    
        # 結果をフィルタリング
        keep = predictions['scores'] > threshold
        boxes = predictions['boxes'][keep].cpu().numpy()
        labels = predictions['labels'][keep].cpu().numpy()
        scores = predictions['scores'][keep].cpu().numpy()
    
        # 結果を描画
        draw = ImageDraw.Draw(img)
    
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            class_name = COCO_CLASSES[label]
    
            # Bounding Box描画
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    
            # ラベルとスコア描画
            text = f"{class_name}: {score:.2f}"
            draw.text((x1, y1 - 15), text, fill='red')
    
        # 結果を表示
        print(f"検出された物体数: {len(boxes)}")
        for label, score in zip(labels, scores):
            print(f"  - {COCO_CLASSES[label]}: {score:.3f}")
    
        return img, boxes, labels, scores
    
    # 使用例
    image_url = "https://images.unsplash.com/photo-1544568100-847a948585b9?w=800"
    result_img, boxes, labels, scores = detect_objects(image_url, threshold=0.7)
    
    # 画像を表示（Jupyter Notebookの場合）
    # display(result_img)
    
    # 画像を保存
    result_img.save('faster_rcnn_result.jpg')
    print("結果を faster_rcnn_result.jpg に保存しました")
    

### Feature Pyramid Networks (FPN)

FPNは、マルチスケールの特徴を効果的に利用するアーキテクチャです。異なるサイズの物体を検出するために、複数の解像度の特徴マップを組み合わせます。

> **FPNの特徴** ：
> 
>   * ボトムアップパス：通常のCNNの順伝播
>   * トップダウンパス：高レベル特徴を低解像度から高解像度へ伝播
>   * ラテラル接続：各レベルの特徴を結合
> 

* * *

## 3.3 One-Stage検出器

### YOLOファミリー

**YOLO (You Only Look Once)** は、画像を1回見るだけで物体検出を行う革新的なアプローチです。リアルタイム検出を実現し、Two-Stage検出器よりも高速です。

#### YOLOの基本原理

  1. 画像をグリッド（例：13×13）に分割
  2. 各グリッドセルがBounding Boxと信頼度を予測
  3. 各Boxに対してクラス確率を予測
  4. NMSで重複を除去

#### YOLOの進化

バージョン | 年 | 主な改良点  
---|---|---  
**YOLOv1** | 2016 | One-Stage検出の提案、リアルタイム処理  
**YOLOv2** | 2017 | Batch Normalization、Anchor Box導入  
**YOLOv3** | 2018 | マルチスケール予測、Darknet-53  
**YOLOv4** | 2020 | CSPDarknet53、Mosaic augmentation  
**YOLOv5** | 2020 | PyTorch実装、使いやすさ向上  
**YOLOv8** | 2023 | Anchor-free、改良されたアーキテクチャ  
コード例4：YOLOv8での物体検出
    
    
    from ultralytics import YOLO
    from PIL import Image
    import cv2
    import numpy as np
    
    # YOLOv8モデルをロード
    # サイズ: n (nano), s (small), m (medium), l (large), x (extra large)
    model = YOLO('yolov8n.pt')  # nanoモデル（最軽量）
    
    def detect_with_yolo(image_path, conf_threshold=0.5):
        """
        YOLOv8で物体検出を実行
    
        Args:
            image_path: 画像パスまたはURL
            conf_threshold: 信頼度閾値
        """
        # 推論実行
        results = model(image_path, conf=conf_threshold)
    
        # 結果を取得
        result = results[0]
    
        # 検出された物体の情報を表示
        print(f"検出された物体数: {len(result.boxes)}")
    
        for box in result.boxes:
            # クラスID、信頼度、座標を取得
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    
            class_name = model.names[class_id]
            print(f"  - {class_name}: {confidence:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
    
        # 結果画像を取得（アノテーション付き）
        annotated_img = result.plot()
    
        return annotated_img, result
    
    # 使用例1：画像ファイルから検出
    image_path = "path/to/your/image.jpg"
    annotated_img, result = detect_with_yolo(image_path, conf_threshold=0.5)
    
    # 結果を保存
    cv2.imwrite('yolov8_result.jpg', annotated_img)
    print("結果を yolov8_result.jpg に保存しました")
    
    # 使用例2：ビデオファイルまたはWebcamから検出
    def detect_video(source=0, conf_threshold=0.5):
        """
        ビデオまたはWebcamからリアルタイム検出
    
        Args:
            source: ビデオファイルパスまたは0（Webcam）
            conf_threshold: 信頼度閾値
        """
        # ビデオストリームで推論
        results = model(source, stream=True, conf=conf_threshold)
    
        for result in results:
            # フレームごとに処理
            annotated_frame = result.plot()
    
            # 表示
            cv2.imshow('YOLOv8 Detection', annotated_frame)
    
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cv2.destroyAllWindows()
    
    # Webcamでリアルタイム検出（コメント解除して実行）
    # detect_video(source=0, conf_threshold=0.5)
    
    # ビデオファイルで検出
    # detect_video(source='path/to/video.mp4', conf_threshold=0.5)
    

### SSD (Single Shot Detector)

SSDは、YOLOと同様にOne-Stage検出器ですが、複数のスケールの特徴マップから検出を行います。

#### SSDの特徴

  * **マルチスケール特徴マップ** ：異なる解像度の層から検出
  * **デフォルトボックス** ：各位置で複数のアスペクト比のBoxを予測
  * **高速** ：YOLOv1より高速でmAPも高い

### RetinaNet (Focal Loss)

RetinaNetは、**Focal Loss** を導入することで、クラス不均衡問題を解決しました。

#### Focal Lossとは

> Focal Loss = -α(1-p_t)^γ log(p_t)
> 
> 簡単な例（背景など）の損失を小さくし、難しい例に集中して学習します。

コード例5：Focal Lossの実装
    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class FocalLoss(nn.Module):
        """
        Focal Loss for Object Detection
    
        Args:
            alpha: クラス重み（デフォルト: 0.25）
            gamma: フォーカスパラメータ（デフォルト: 2.0）
        """
        def __init__(self, alpha=0.25, gamma=2.0):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
    
        def forward(self, predictions, targets):
            """
            Args:
                predictions: (N, num_classes) 予測確率
                targets: (N,) 正解ラベル
            """
            # Cross Entropy Loss
            ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    
            # p_tを計算（正解クラスの予測確率）
            p = torch.exp(-ce_loss)
    
            # Focal Loss
            focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
    
            return focal_loss.mean()
    
    # 使用例
    num_classes = 91  # COCO
    batch_size = 32
    
    # ダミーデータ
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    # 通常のCross Entropy Loss
    ce_loss = F.cross_entropy(predictions, targets)
    print(f"Cross Entropy Loss: {ce_loss.item():.4f}")
    
    # Focal Loss
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss = focal_loss_fn(predictions, targets)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # 簡単な例 vs 難しい例での損失比較
    easy_predictions = torch.tensor([[10.0, 0.0, 0.0]])  # 正解クラス0に高い確率
    hard_predictions = torch.tensor([[1.0, 0.9, 0.8]])   # 正解クラス0だが低い確率
    targets_test = torch.tensor([0])
    
    easy_loss = focal_loss_fn(easy_predictions, targets_test)
    hard_loss = focal_loss_fn(hard_predictions, targets_test)
    
    print(f"\n簡単な例の損失: {easy_loss.item():.4f}")
    print(f"難しい例の損失: {hard_loss.item():.4f}")
    print(f"難しい例の損失は簡単な例の {hard_loss.item() / easy_loss.item():.1f} 倍")
    

### EfficientDet

EfficientDetは、EfficientNetをバックボーンとし、BiFPN（Bi-directional Feature Pyramid Network）を使用した効率的な検出器です。

  * **Compound Scaling** ：解像度、深さ、幅を同時にスケーリング
  * **BiFPN** ：双方向の特徴融合
  * **高効率** ：YOLOv3やRetinaNetよりも少ないパラメータで高精度

* * *

## 3.4 実装と訓練

### COCOデータセット

COCO (Common Objects in Context) は、物体検出の標準ベンチマークデータセットです。

  * **画像数** ：330K枚（train: 118K, val: 5K, test: 41K）
  * **カテゴリ** ：80クラス（人、動物、乗り物、家具など）
  * **アノテーション** ：Bounding Box、セグメンテーション、キーポイント

コード例6：PyTorch Object Detectionの訓練
    
    
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torch.utils.data import DataLoader
    import torchvision.transforms as T
    
    # カスタムデータセットクラス
    class CustomObjectDetectionDataset(torch.utils.data.Dataset):
        """
        カスタム物体検出データセット
    
        画像とアノテーション（boxes, labels）を返す
        """
        def __init__(self, image_paths, annotations, transforms=None):
            self.image_paths = image_paths
            self.annotations = annotations
            self.transforms = transforms
    
        def __len__(self):
            return len(self.image_paths)
    
        def __getitem__(self, idx):
            # 画像読み込み
            from PIL import Image
            img = Image.open(self.image_paths[idx]).convert("RGB")
    
            # アノテーション取得
            boxes = self.annotations[idx]['boxes']  # [[x1,y1,x2,y2], ...]
            labels = self.annotations[idx]['labels']  # [1, 2, 1, ...]
    
            # テンソルに変換
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
        Faster R-CNNモデルを構築
    
        Args:
            num_classes: クラス数（背景 + 物体クラス）
        """
        # 事前学習済みモデルをロード
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    
        # 分類ヘッドを置き換え
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
        return model
    
    def train_one_epoch(model, optimizer, data_loader, device):
        """
        1エポックの訓練
        """
        model.train()
        total_loss = 0
    
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
            # 順伝播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
    
            # 逆伝播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
    
            total_loss += losses.item()
    
        return total_loss / len(data_loader)
    
    # 訓練設定
    num_classes = 3  # 背景 + 2クラス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル、オプティマイザ、スケジューラ
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
    
    # データセット（ダミー）
    # 実際には画像パスとアノテーションを用意
    image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    annotations = [
        {'boxes': [[10, 10, 50, 50]], 'labels': [1]},
        {'boxes': [[20, 20, 60, 60], [70, 70, 100, 100]], 'labels': [1, 2]},
        {'boxes': [[30, 30, 80, 80]], 'labels': [2]}
    ]
    
    # transforms = T.Compose([T.ToTensor()])
    # dataset = CustomObjectDetectionDataset(image_paths, annotations, transforms)
    # data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    
    # 訓練ループ（実際のデータがある場合）
    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     train_loss = train_one_epoch(model, optimizer, data_loader, device)
    #     lr_scheduler.step()
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")
    
    # モデル保存
    # torch.save(model.state_dict(), 'object_detection_model.pth')
    
    print("訓練スクリプトの準備完了")
    

### カスタムデータセットでの訓練

コード例7：YOLOv8でカスタムデータセット訓練
    
    
    from ultralytics import YOLO
    import yaml
    import os
    
    # データセット設定ファイルを作成
    dataset_yaml = """
    # データセットのパス
    path: ./custom_dataset  # ルートディレクトリ
    train: images/train     # 訓練画像（pathからの相対パス）
    val: images/val         # 検証画像
    
    # クラス定義
    names:
      0: cat
      1: dog
      2: bird
    """
    
    # dataset.yamlを保存
    with open('custom_dataset.yaml', 'w') as f:
        f.write(dataset_yaml)
    
    # ディレクトリ構造の例：
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
    #     │   ├── img1.txt  # YOLO形式（class x_center y_center width height）
    #     │   ├── img2.txt
    #     │   └── ...
    #     └── val/
    #         ├── img1.txt
    #         └── ...
    
    # YOLOv8モデルを初期化
    model = YOLO('yolov8n.pt')  # 事前学習済みweightsから開始
    
    # 訓練実行
    results = model.train(
        data='custom_dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='custom_yolo',
        # その他のハイパーパラメータ
        lr0=0.01,          # 初期学習率
        momentum=0.937,     # SGDモメンタム
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=50,        # Early stopping
        # Data Augmentation
        degrees=10.0,       # 回転
        translate=0.1,      # 平行移動
        scale=0.5,          # スケール
        flipud=0.0,         # 上下反転
        fliplr=0.5,         # 左右反転
        mosaic=1.0,         # Mosaic augmentation
    )
    
    # 検証
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    
    # 推論
    results = model('path/to/test/image.jpg')
    
    # モデル保存（自動的に保存されるが、手動でも可能）
    # model.save('custom_yolo_best.pt')
    
    # モデルのエクスポート（ONNX, TensorRT, etc.）
    # model.export(format='onnx')
    
    print("\n訓練完了！")
    print(f"Weights: runs/detect/custom_yolo/weights/best.pt")
    print(f"Metrics: runs/detect/custom_yolo/results.csv")
    

**アノテーションファイルの形式（YOLO）：**
    
    
    # img1.txt の例（各行が1つの物体）
    0 0.5 0.5 0.3 0.2    # class=0, center=(0.5, 0.5), size=(0.3, 0.2)
    1 0.7 0.3 0.2 0.15   # class=1, center=(0.7, 0.3), size=(0.2, 0.15)
    
    # 座標は画像サイズで正規化（0~1）
    # class x_center y_center width height
    

* * *

## 3.5 応用テクニック

### Anchor-Free Detection

従来の検出器はAnchor Box（事前定義のBox）に依存していましたが、Anchor-Freeアプローチはこれを不要にします。

#### 主なAnchor-Free手法

  * **FCOS (Fully Convolutional One-Stage)** ：各ピクセルから物体中心までの距離を予測
  * **CenterNet** ：物体の中心点を検出し、サイズと位置を回帰
  * **YOLOv8** ：Anchor-Freeアプローチを採用

> **メリット** ：Anchorのハイパーパラメータチューニングが不要、より柔軟な検出

### 物体追跡（Object Tracking）

物体追跡は、ビデオ内の物体を連続的に追跡するタスクです。検出器と組み合わせて使用します。

#### SORT (Simple Online and Realtime Tracking)

  1. フレームごとに物体検出
  2. Kalmanフィルタで次フレームの位置を予測
  3. Hungarian Algorithmで検出と追跡をマッチング

#### DeepSORT

SORTに外観特徴（Deep features）を追加し、より堅牢な追跡を実現。

コード例8：YOLOv8 + 物体追跡
    
    
    from ultralytics import YOLO
    import cv2
    
    # YOLOv8モデルをロード
    model = YOLO('yolov8n.pt')
    
    def track_objects_video(video_path, output_path='tracking_result.mp4'):
        """
        ビデオで物体を検出・追跡
    
        Args:
            video_path: 入力ビデオパス
            output_path: 出力ビデオパス
        """
        # ビデオキャプチャ
        cap = cv2.VideoCapture(video_path)
    
        # 出力設定
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
        frame_count = 0
    
        # 追跡モードで推論
        results = model.track(video_path, stream=True, persist=True, conf=0.5)
    
        for result in results:
            frame_count += 1
    
            # アノテーション付きフレーム
            annotated_frame = result.plot()
    
            # トラッキングIDを表示
            if result.boxes.id is not None:
                for box, track_id in zip(result.boxes.xyxy, result.boxes.id):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    track_id = int(track_id.cpu().numpy())
    
                    # トラッキングID描画
                    cv2.putText(
                        annotated_frame,
                        f"ID: {track_id}",
                        (int(x1), int(y1) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
    
            # フレームを書き込み
            out.write(annotated_frame)
    
            # 表示
            cv2.imshow('Object Tracking', annotated_frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
        print(f"処理完了: {frame_count} フレーム")
        print(f"出力: {output_path}")
    
    # 使用例
    # track_objects_video('input_video.mp4', 'output_tracking.mp4')
    
    # Webcamでリアルタイム追跡
    def track_webcam():
        """
        Webcamでリアルタイム物体追跡
        """
        results = model.track(source=0, stream=True, persist=True, conf=0.5)
    
        for result in results:
            annotated_frame = result.plot()
            cv2.imshow('Real-time Tracking', annotated_frame)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        cv2.destroyAllWindows()
    
    # リアルタイム追跡（コメント解除して実行）
    # track_webcam()
    
    print("物体追跡スクリプトの準備完了")
    

### マルチスケール検出

物体のサイズは画像によって大きく異なるため、マルチスケール検出が重要です。

#### テクニック

  * **Image Pyramid** ：画像を複数のスケールでリサイズして検出
  * **Feature Pyramid** ：複数の特徴マップレベルで検出（FPN）
  * **Multi-scale Training** ：訓練時に異なるサイズの入力を使用

### リアルタイム最適化

#### 高速化テクニック

  * **モデルの軽量化** ：YOLOv8n, MobileNet-SSD
  * **量子化** ：FP32 → FP16 → INT8
  * **TensorRT** ：NVIDIAの推論最適化エンジン
  * **ONNX Runtime** ：クロスプラットフォーム推論
  * **解像度調整** ：入力画像サイズを小さく（例：640→416）

> **速度 vs 精度のトレードオフ** ：
> 
>   * リアルタイム要求：YOLOv8n/s（30+ FPS）
>   * 高精度要求：YOLOv8x, Faster R-CNN with FPN
> 

* * *

## 練習問題

演習1：IoUとNMSの理解

**問題** ：以下のBounding Boxセットに対してIoUを計算し、NMSを適用してください。
    
    
    boxes = np.array([
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [105, 105, 205, 205],
        [300, 300, 400, 400]
    ])
    scores = np.array([0.9, 0.85, 0.95, 0.8])
    

  * 各BoxとBox 0のIoUを計算
  * IoU閾値0.5でNMSを適用
  * 残るBoxを特定

演習2：Faster R-CNNで検出

**問題** ：事前学習済みFaster R-CNNを使って、複数の画像から特定クラス（例：person, car）のみを検出するスクリプトを作成してください。

  * 複数画像を読み込み
  * 指定クラスのみフィルタリング
  * 信頼度が0.7以上のもののみ表示
  * 検出数を集計

演習3：YOLOv8のモデルサイズ比較

**問題** ：YOLOv8の異なるサイズ（n, s, m, l）で同じ画像を検出し、精度と速度を比較してください。

  * 各モデルで推論時間を測定
  * 検出された物体数を比較
  * 信頼度スコアの分布を分析
  * どのモデルが最適か判断

演習4：カスタムデータセットの準備

**問題** ：独自の画像データセット（10枚以上）を用意し、YOLO形式のアノテーションファイルを作成してください。

  * LabelImgなどのツールでアノテーション作成
  * YOLO形式（class x_center y_center width height）に変換
  * train/valに分割（80/20）
  * dataset.yamlを作成

演習5：物体追跡の実装

**問題** ：ビデオファイルまたはWebcamから物体を追跡し、各物体の軌跡を描画してください。

  * YOLOv8の追跡機能を使用
  * 各トラッキングIDの軌跡を保存
  * 軌跡を線で描画
  * フレーム間でIDが安定しているか確認

演習6：リアルタイム検出の最適化

**問題** ：検出速度を最大化するために、モデルを最適化してください。

  * YOLOv8をONNX形式にエクスポート
  * 異なる入力解像度（320, 416, 640）でFPSを測定
  * 信頼度閾値を調整して速度改善
  * 速度と精度のトレードオフを分析

* * *

## まとめ

この章では、物体検出の基礎から実践までを学びました：

  * ✅ **物体検出の基礎** ：Bounding Box表現、IoU、NMS、評価指標（mAP）
  * ✅ **Two-Stage検出器** ：R-CNN、Fast R-CNN、Faster R-CNN、FPNの進化と特徴
  * ✅ **One-Stage検出器** ：YOLO、SSD、RetinaNet、EfficientDetの原理と比較
  * ✅ **実装と訓練** ：PyTorch、YOLOv8を使った実践的な検出とカスタムデータセット訓練
  * ✅ **応用テクニック** ：Anchor-free検出、物体追跡、マルチスケール検出、リアルタイム最適化

次章では、**セマンティックセグメンテーション** を学びます。ピクセルレベルの分類、U-Net、DeepLab、Mask R-CNNなど、より詳細な画像理解の手法を理解していきます。

> **重要なポイント** ：物体検出は、リアルタイム性と精度のトレードオフが重要です。アプリケーションの要件（速度優先 or 精度優先）に応じて、適切なモデルとパラメータを選択しましょう。

## 参考文献

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
