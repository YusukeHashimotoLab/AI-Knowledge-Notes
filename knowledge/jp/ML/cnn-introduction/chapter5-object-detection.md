---
title: 第5章：物体検出入門
chapter_title: 第5章：物体検出入門
subtitle: 画像分類から物体検出へ - R-CNN、YOLO、そして最新手法
reading_time: 25-30分
difficulty: 中級〜上級
code_examples: 8
exercises: 4
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ 画像分類と物体検出の違いを理解し、適切なタスク定義ができる
  * ✅ Two-stage detector（R-CNN系）のアーキテクチャと進化を説明できる
  * ✅ One-stage detector（YOLO、SSD）の設計思想と利点を理解できる
  * ✅ IoU、NMS、mAPなどの評価指標を実装し解釈できる
  * ✅ PyTorchで物体検出モデルを実装・推論できる
  * ✅ COCO形式のデータセットで実践的な物体検出を実現できる

* * *

## 5.1 物体検出とは

### 画像認識タスクの種類

コンピュータビジョンにおける画像認識タスクは、目的に応じて主に3つに分類されます。
    
    
    ```mermaid
    graph LR
        A[画像認識タスク] --> B[Classification画像分類]
        A --> C[Detection物体検出]
        A --> D[Segmentationセグメンテーション]
    
        B --> B1["「この画像は何？」クラスラベルのみ"]
        C --> C1["「何が、どこに？」位置+クラスラベル"]
        D --> D1["「どのピクセルが何？」ピクセル単位の分類"]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#e8f5e9
        style D fill:#f3e5f5
    ```

タスク | 目的 | 出力 | 応用例  
---|---|---|---  
**Classification** | 画像全体のクラス分類 | クラスラベル（例: "猫"） | 画像検索、コンテンツフィルタリング  
**Detection** | 物体の位置とクラス特定 | Bounding Box + クラスラベル | 自動運転、監視カメラ、医療画像  
**Segmentation** | ピクセル単位の領域分割 | セグメンテーションマスク | 背景除去、3D再構成、医療診断  
  
### 物体検出の基本概念

#### Bounding Box（バウンディングボックス）

**Bounding Box** は、検出された物体を囲む矩形領域で、以下の情報を持ちます：

  * **座標表現** : $(x, y, w, h)$ または $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$
  * **クラスラベル** : 物体のカテゴリ（例: "person", "car"）
  * **信頼度スコア** : 検出の確信度 $[0, 1]$

    
    
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np
    
    def visualize_bounding_boxes(image_path, boxes, labels, scores, class_names):
        """
        Bounding Boxの可視化
    
        Args:
            image_path: 画像ファイルパス
            boxes: Bounding Box座標 [[x_min, y_min, x_max, y_max], ...]
            labels: クラスラベル [0, 1, 2, ...]
            scores: 信頼度スコア [0.95, 0.87, ...]
            class_names: クラス名のリスト ['person', 'car', ...]
        """
        # 画像の読み込み
        img = Image.open(image_path)
    
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(img)
    
        # 各Bounding Boxを描画
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    
        for box, label, score in zip(boxes, labels, scores):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
    
            # 矩形を描画
            color = colors[label % len(colors)]
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
    
            # ラベルとスコアを表示
            label_text = f'{class_names[label]}: {score:.2f}'
            ax.text(
                x_min, y_min - 5,
                label_text,
                bbox=dict(facecolor=color, alpha=0.7),
                fontsize=10, color='white'
            )
    
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    # 使用例
    # boxes = [[50, 50, 200, 300], [250, 100, 400, 350]]
    # labels = [0, 1]  # 0: person, 1: car
    # scores = [0.95, 0.87]
    # class_names = ['person', 'car', 'dog', 'cat']
    # visualize_bounding_boxes('sample.jpg', boxes, labels, scores, class_names)
    

#### IoU (Intersection over Union)

**IoU** は、2つのBounding Boxの重なり具合を測る指標で、物体検出の評価に不可欠です。

$$ \text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|A \cap B|}{|A \cup B|} $$
    
    
    ```mermaid
    graph LR
        A[予測Box] --> C[Intersection重なり領域]
        B[正解Box] --> C
        C --> D[Union和集合領域]
        D --> E[IoU = Intersection / Union]
    
        style A fill:#ffebee
        style B fill:#e8f5e9
        style C fill:#fff3e0
        style E fill:#e3f2fd
    ```
    
    
    def calculate_iou(box1, box2):
        """
        2つのBounding Box間のIoUを計算
    
        Args:
            box1, box2: [x_min, y_min, x_max, y_max]
    
        Returns:
            iou: IoU値 [0, 1]
        """
        # 交差領域の座標
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])
    
        # 交差領域の面積
        inter_width = max(0, x_max_inter - x_min_inter)
        inter_height = max(0, y_max_inter - y_min_inter)
        intersection = inter_width * inter_height
    
        # 各Boxの面積
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
        # 和集合の面積
        union = area1 + area2 - intersection
    
        # IoU計算（ゼロ除算を回避）
        iou = intersection / union if union > 0 else 0
    
        return iou
    
    # 使用例とテスト
    box1 = [50, 50, 150, 150]   # 正解Box
    box2 = [100, 100, 200, 200] # 予測Box（部分的重なり）
    box3 = [50, 50, 150, 150]   # 予測Box（完全一致）
    box4 = [200, 200, 300, 300] # 予測Box（重なりなし）
    
    print(f"部分的重なり IoU: {calculate_iou(box1, box2):.4f}")  # ~0.14
    print(f"完全一致 IoU: {calculate_iou(box1, box3):.4f}")      # 1.00
    print(f"重なりなし IoU: {calculate_iou(box1, box4):.4f}")    # 0.00
    
    # ベクトル化されたバッチIoU計算
    def batch_iou(boxes1, boxes2):
        """
        複数のBounding Box間のIoUを効率的に計算（PyTorchバージョン）
    
        Args:
            boxes1: Tensor of shape [N, 4]
            boxes2: Tensor of shape [M, 4]
    
        Returns:
            iou: Tensor of shape [N, M]
        """
        # 交差領域の計算
        x_min = torch.max(boxes1[:, None, 0], boxes2[:, 0])
        y_min = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x_max = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y_max = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
        inter_width = torch.clamp(x_max - x_min, min=0)
        inter_height = torch.clamp(y_max - y_min, min=0)
        intersection = inter_width * inter_height
    
        # 面積の計算
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
        # 和集合とIoU
        union = area1[:, None] + area2 - intersection
        iou = intersection / union
    
        return iou
    
    # 使用例
    boxes1 = torch.tensor([[50, 50, 150, 150], [100, 100, 200, 200]], dtype=torch.float32)
    boxes2 = torch.tensor([[50, 50, 150, 150], [200, 200, 300, 300]], dtype=torch.float32)
    
    iou_matrix = batch_iou(boxes1, boxes2)
    print("\nBatch IoU Matrix:")
    print(iou_matrix)
    # 出力:
    # tensor([[1.0000, 0.0000],
    #         [0.1429, 0.0000]])
    

> **IoUの判定基準** :
> 
>   * IoU ≥ 0.5: 通常、正解として扱われる（PASCAL VOC基準）
>   * IoU ≥ 0.75: 厳しい基準（COCO評価）
>   * IoU < 0.5: 誤検出として扱われる
> 

* * *

## 5.2 Two-Stage Detectors

### R-CNN系の進化

Two-stage detectorは、**①候補領域の提案** と**②物体のクラス分類** を2段階で行うアプローチです。
    
    
    ```mermaid
    graph LR
        A[入力画像] --> B[Stage 1Region Proposal]
        B --> C[候補領域~2000個]
        C --> D[Stage 2Classification]
        D --> E[最終検出結果Box + Class]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style D fill:#f3e5f5
        style E fill:#e8f5e9
    ```

#### R-CNN (2014)

**R-CNN (Regions with CNN features)** は、深層学習ベースの物体検出の先駆けです。

ステップ | 処理内容 | 特徴  
---|---|---  
**1\. Region Proposal** | Selective Searchで候補領域生成（~2000個） | 従来の画像処理手法  
**2\. CNN Feature Extraction** | 各領域をAlexNetで特徴抽出 | 2000回の順伝播が必要  
**3\. SVM Classification** | SVMでクラス分類 | CNNとは別に訓練  
**4\. Bounding Box Regression** | Box座標を微調整 | 精度向上  
  
**問題点** :

  * 推論が非常に遅い（1画像あたり47秒）
  * 訓練が複雑（3段階の別々の学習）
  * 特徴抽出の重複計算が多い

#### Fast R-CNN (2015)

**Fast R-CNN** は、R-CNNの計算効率を大幅に改善しました。
    
    
    ```mermaid
    graph LR
        A[入力画像] --> B[CNN特徴マップ]
        B --> C[RoI Pooling]
        D[RegionProposals] --> C
        C --> E[FC層]
        E --> F1[Softmaxクラス分類]
        E --> F2[RegressorBox回帰]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style F1 fill:#e8f5e9
        style F2 fill:#e8f5e9
    ```

**改善点** :

  * 画像全体で1回だけCNNを実行
  * RoI Poolingで候補領域から固定サイズの特徴を抽出
  * Multi-task Loss（分類 + Box回帰）でEnd-to-end学習
  * 推論速度: 47秒 → 2秒（23倍高速化）

#### Faster R-CNN (2016)

**Faster R-CNN** は、Region ProposalもCNNで学習可能にし、完全なEnd-to-end化を実現しました。
    
    
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    def create_faster_rcnn(num_classes, pretrained=True):
        """
        Faster R-CNNモデルの作成
    
        Args:
            num_classes: 検出するクラス数（背景を含む）
            pretrained: COCO事前学習済み重みを使用するか
    
        Returns:
            model: Faster R-CNNモデル
        """
        # COCO事前学習済みモデルのロード
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    
        # 分類器の置き換え（最終層のみ）
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
        return model
    
    # モデルの作成（例: COCO 80クラス + 背景）
    model = create_faster_rcnn(num_classes=91, pretrained=True)
    model.eval()
    
    print("Faster R-CNNモデルの構造:")
    print(f"- Backbone: ResNet-50 + FPN")
    print(f"- RPN: Region Proposal Network")
    print(f"- RoI Heads: Box Head + Class Predictor")
    
    # 推論の実行
    def run_faster_rcnn_inference(model, image_path, threshold=0.5):
        """
        Faster R-CNNによる物体検出推論
    
        Args:
            model: Faster R-CNNモデル
            image_path: 入力画像パス
            threshold: 検出スコアの閾値
    
        Returns:
            boxes, labels, scores: 検出結果
        """
        from PIL import Image
        from torchvision import transforms
    
        # 画像の読み込みと前処理
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    
        # 推論
        model.eval()
        with torch.no_grad():
            predictions = model(img_tensor)
    
        # 閾値以上のスコアの検出結果のみ抽出
        pred = predictions[0]
        keep = pred['scores'] > threshold
    
        boxes = pred['boxes'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
    
        print(f"\n検出された物体数: {len(boxes)}")
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            print(f"  {i+1}. Label: {label}, Score: {score:.3f}, Box: {box}")
    
        return boxes, labels, scores
    
    # COCO クラス名（一部）
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'
        # ... 全91クラス
    ]
    
    # 使用例
    # boxes, labels, scores = run_faster_rcnn_inference(model, 'test_image.jpg', threshold=0.7)
    # visualize_bounding_boxes('test_image.jpg', boxes, labels, scores, COCO_INSTANCE_CATEGORY_NAMES)
    

### Region Proposal Network (RPN)

**RPN** は、Faster R-CNNの核心技術で、候補領域を学習ベースで提案します。

> **RPNの仕組み** :
> 
>   1. 特徴マップの各位置に複数のAnchor Box（異なるサイズ・アスペクト比）を配置
>   2. 各Anchorに対して「物体らしさ（Objectness）」をスコアリング
>   3. Bounding Box の座標オフセットを回帰
>   4. 高スコアのProposalをRoI Poolingに渡す
> 

    
    
    class SimpleRPN(nn.Module):
        """
        簡略化されたRegion Proposal Network（教育目的）
        """
    
        def __init__(self, in_channels=512, num_anchors=9):
            """
            Args:
                in_channels: 入力特徴マップのチャンネル数
                num_anchors: 各位置のAnchor数（通常 3スケール × 3アスペクト比 = 9）
            """
            super(SimpleRPN, self).__init__()
    
            # 共有畳み込み層
            self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
    
            # Objectnessスコア（物体 or 背景の2クラス）
            self.cls_logits = nn.Conv2d(512, num_anchors * 2, kernel_size=1)
    
            # Bounding Box回帰（4座標 × num_anchors）
            self.bbox_pred = nn.Conv2d(512, num_anchors * 4, kernel_size=1)
    
        def forward(self, feature_map):
            """
            Args:
                feature_map: [B, C, H, W] 特徴マップ
    
            Returns:
                objectness: [B, num_anchors*2, H, W] 物体スコア
                bbox_deltas: [B, num_anchors*4, H, W] Box座標オフセット
            """
            # 共有特徴抽出
            x = torch.relu(self.conv(feature_map))
    
            # Objectness分類
            objectness = self.cls_logits(x)
    
            # Bounding Box回帰
            bbox_deltas = self.bbox_pred(x)
    
            return objectness, bbox_deltas
    
    # RPNの動作確認
    rpn = SimpleRPN(in_channels=512, num_anchors=9)
    feature_map = torch.randn(1, 512, 38, 38)  # 例: ResNetの特徴マップ
    
    objectness, bbox_deltas = rpn(feature_map)
    print(f"Objectness shape: {objectness.shape}")     # [1, 18, 38, 38]
    print(f"BBox Deltas shape: {bbox_deltas.shape}")   # [1, 36, 38, 38]
    print(f"Total Proposals: {38 * 38 * 9} anchors")   # 12,996個のAnchor
    

* * *

## 5.3 One-Stage Detectors

### YOLO (You Only Look Once)

**YOLO** は、物体検出を「回帰問題」として定式化し、単一のCNNで直接Bounding BoxとクラスをEnd-to-end予測します。
    
    
    ```mermaid
    graph LR
        A[入力画像448×448] --> B[CNN Backbone特徴抽出]
        B --> C[Grid分割7×7]
        C --> D[各セルで予測Box + Class]
        D --> E[NMS重複除去]
        E --> F[最終検出結果]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#ffe0b2
        style E fill:#e1bee7
        style F fill:#e8f5e9
    ```

#### YOLOの設計思想

  * **速度重視** : リアルタイム推論（45+ FPS）を目指す
  * **Global Context** : 画像全体を一度に見るため文脈理解が良い
  * **シンプル** : 複雑なパイプラインなし、End-to-end学習

    
    
    import torch
    import torch.nn as nn
    
    # YOLOv5の使用（Ultralytics実装）
    def load_yolov5(model_size='yolov5s', pretrained=True):
        """
        YOLOv5モデルのロード
    
        Args:
            model_size: モデルサイズ ('yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            pretrained: COCO事前学習済み重みを使用
    
        Returns:
            model: YOLOv5モデル
        """
        # PyTorch Hubからロード（Ultralytics実装）
        model = torch.hub.load('ultralytics/yolov5', model_size, pretrained=pretrained)
    
        return model
    
    # モデルのロード
    model = load_yolov5('yolov5s', pretrained=True)
    model.eval()
    
    print("YOLOv5s モデル情報:")
    print(f"- パラメータ数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"- 推論速度: ~140 FPS (GPU)")
    print(f"- 入力サイズ: 640×640 (デフォルト)")
    
    def run_yolo_inference(model, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        YOLOv5による物体検出推論
    
        Args:
            model: YOLOv5モデル
            image_path: 入力画像パス
            conf_threshold: 信頼度スコアの閾値
            iou_threshold: NMSのIoU閾値
    
        Returns:
            results: 検出結果（pandas DataFrame）
        """
        # 推論設定
        model.conf = conf_threshold
        model.iou = iou_threshold
    
        # 推論実行
        results = model(image_path)
    
        # 結果の表示
        results.print()  # コンソールに出力
    
        # 結果の可視化
        results.show()   # 画像表示
    
        # 結果をDataFrameで取得
        detections = results.pandas().xyxy[0]
    
        print(f"\n検出された物体数: {len(detections)}")
        print(detections)
    
        return results
    
    # 使用例
    # results = run_yolo_inference(model, 'test_image.jpg', conf_threshold=0.5)
    
    # バッチ推論
    def run_yolo_batch_inference(model, image_paths, save_dir='results/'):
        """
        複数画像のバッチ推論
    
        Args:
            model: YOLOv5モデル
            image_paths: 画像パスのリスト
            save_dir: 結果保存ディレクトリ
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
    
        # バッチ推論
        results = model(image_paths)
    
        # 結果を保存
        results.save(save_dir=save_dir)
    
        print(f"バッチ推論完了: {len(image_paths)}枚の画像")
        print(f"結果保存先: {save_dir}")
    
        return results
    
    # 使用例
    # image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']
    # batch_results = run_yolo_batch_inference(model, image_list)
    

#### YOLOのLoss関数

YOLOは3つの損失を組み合わせて学習します：

$$ \mathcal{L}_{\text{YOLO}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{obj}} \mathcal{L}_{\text{obj}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} $$

  * $\mathcal{L}_{\text{box}}$: Bounding Box座標の回帰損失（CIoU Loss）
  * $\mathcal{L}_{\text{obj}}$: Objectness（物体らしさ）の二値分類損失
  * $\mathcal{L}_{\text{cls}}$: クラス分類の多クラス損失

### SSD (Single Shot Detector)

**SSD** は、異なるスケールの特徴マップで検出を行い、速度と精度のバランスを取ります。

> **SSDの特徴** :
> 
>   * Multi-scale Feature Maps（複数解像度での検出）
>   * Default Boxes（Anchorに相当）を各特徴マップで使用
>   * YOLOより精度が高く、Faster R-CNNより速い
> 

    
    
    from torchvision.models.detection import ssd300_vgg16
    
    def create_ssd_model(num_classes=91, pretrained=True):
        """
        SSD300モデルの作成
    
        Args:
            num_classes: 検出クラス数
            pretrained: 事前学習済み重みを使用
    
        Returns:
            model: SSD300モデル
        """
        # SSD300 with VGG16 backbone
        model = ssd300_vgg16(pretrained=pretrained, num_classes=num_classes)
    
        return model
    
    # モデルのロード
    ssd_model = create_ssd_model(num_classes=91, pretrained=True)
    ssd_model.eval()
    
    print("SSD300モデル情報:")
    print(f"- 入力サイズ: 300×300")
    print(f"- Backbone: VGG16")
    print(f"- 特徴マップ: 6層（異なるスケール）")
    
    def run_ssd_inference(model, image_path, threshold=0.5):
        """
        SSDによる物体検出推論
        """
        from PIL import Image
        from torchvision import transforms
    
        # 画像の読み込みと前処理
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)
    
        # 推論
        model.eval()
        with torch.no_grad():
            predictions = model(img_tensor)
    
        # 結果の抽出
        pred = predictions[0]
        keep = pred['scores'] > threshold
    
        boxes = pred['boxes'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
    
        print(f"\n検出された物体数: {len(boxes)}")
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            print(f"  {i+1}. Label: {label}, Score: {score:.3f}")
    
        return boxes, labels, scores
    
    # 使用例
    # boxes, labels, scores = run_ssd_inference(ssd_model, 'test_image.jpg', threshold=0.6)
    

* * *

## 5.4 評価指標

### Precision と Recall

物体検出の評価には、情報検索と同様の指標が使われます。

$$ \text{Precision} = \frac{TP}{TP + FP} \quad \text{(検出の正確性)} $$

$$ \text{Recall} = \frac{TP}{TP + FN} \quad \text{(検出の網羅性)} $$

  * **TP (True Positive)** : 正しく検出された物体（IoU ≥ 閾値）
  * **FP (False Positive)** : 誤検出（IoU < 閾値 または 背景を物体と誤認）
  * **FN (False Negative)** : 検出漏れ（存在する物体を見逃した）

### NMS (Non-Maximum Suppression)

**NMS** は、重複する検出結果を除去し、1つの物体に対して1つのBoxのみを残すアルゴリズムです。
    
    
    ```mermaid
    graph LR
        A[検出Boxesスコア順にソート] --> B[最高スコアBox選択]
        B --> C[重複Box除去IoU > threshold]
        C --> D{残りBoxあり?}
        D -->|Yes| B
        D -->|No| E[最終検出結果]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style E fill:#e8f5e9
    ```
    
    
    def non_max_suppression(boxes, scores, iou_threshold=0.5):
        """
        Non-Maximum Suppression (NMS)の実装
    
        Args:
            boxes: Bounding Box座標 [[x_min, y_min, x_max, y_max], ...] (numpy array)
            scores: 信頼度スコア [0.9, 0.8, ...]
            iou_threshold: IoU閾値（これ以上重複するBoxは除去）
    
        Returns:
            keep_indices: 保持するBoxのインデックス
        """
        import numpy as np
    
        # スコアの降順でソート
        sorted_indices = np.argsort(scores)[::-1]
    
        keep_indices = []
    
        while len(sorted_indices) > 0:
            # 最高スコアのBoxを保持
            current = sorted_indices[0]
            keep_indices.append(current)
    
            if len(sorted_indices) == 1:
                break
    
            # 残りのBoxとのIoUを計算
            current_box = boxes[current]
            remaining_boxes = boxes[sorted_indices[1:]]
    
            ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])
    
            # IoU閾値以下のBoxのみ残す
            keep_mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][keep_mask]
    
        return np.array(keep_indices)
    
    # 使用例
    boxes = np.array([
        [50, 50, 150, 150],
        [55, 55, 155, 155],   # 最初のBoxと重複大
        [200, 200, 300, 300],
        [205, 205, 305, 305]  # 3番目のBoxと重複大
    ])
    scores = np.array([0.9, 0.85, 0.95, 0.88])
    
    keep_indices = non_max_suppression(boxes, scores, iou_threshold=0.5)
    print(f"元のBox数: {len(boxes)}")
    print(f"NMS後のBox数: {len(keep_indices)}")
    print(f"保持されたインデックス: {keep_indices}")
    print(f"保持されたBoxes:\n{boxes[keep_indices]}")
    
    # PyTorchの公式NMS実装（より高速）
    from torchvision.ops import nms
    
    def nms_torch(boxes, scores, iou_threshold=0.5):
        """
        PyTorch版NMS（C++実装で高速）
    
        Args:
            boxes: Tensor of shape [N, 4]
            scores: Tensor of shape [N]
            iou_threshold: IoU閾値
    
        Returns:
            keep: 保持するBoxのインデックス（Tensor）
        """
        keep = nms(boxes, scores, iou_threshold)
        return keep
    
    # 使用例
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    keep_torch = nms_torch(boxes_tensor, scores_tensor, iou_threshold=0.5)
    print(f"\nPyTorch NMS結果: {keep_torch}")
    

### mAP (mean Average Precision)

**mAP** は、物体検出の標準的な評価指標で、全クラスの平均精度を表します。

#### 計算手順

  1. **各クラスごとに** Precision-Recall曲線を描画
  2. 曲線の下側面積 **AP (Average Precision)** を計算
  3. 全クラスのAPの平均を取り **mAP** を算出

$$ \text{AP} = \int_0^1 P(r) \, dr $$

$$ \text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c $$
    
    
    def calculate_precision_recall(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5):
        """
        Precision-Recall曲線のための値を計算
    
        Args:
            pred_boxes: 予測Boxes [N, 4]
            pred_scores: 予測スコア [N]
            true_boxes: 正解Boxes [M, 4]
            iou_threshold: IoU閾値
    
        Returns:
            precisions, recalls: Precision-Recall値のリスト
        """
        import numpy as np
    
        # スコアの降順でソート
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
    
        num_true = len(true_boxes)
        matched_true = np.zeros(num_true, dtype=bool)
    
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
    
        for i, pred_box in enumerate(pred_boxes):
            # 正解Boxとの最大IoUを計算
            if len(true_boxes) == 0:
                fp[i] = 1
                continue
    
            ious = np.array([calculate_iou(pred_box, true_box) for true_box in true_boxes])
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
    
            # IoU閾値を超え、まだマッチしていない正解BoxならTP
            if max_iou >= iou_threshold and not matched_true[max_iou_idx]:
                tp[i] = 1
                matched_true[max_iou_idx] = True
            else:
                fp[i] = 1
    
        # 累積和
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
    
        # Precision と Recall
        recalls = tp_cumsum / num_true if num_true > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
        return precisions, recalls
    
    def calculate_ap(precisions, recalls):
        """
        Average Precision (AP)を計算（11点補間法）
    
        Args:
            precisions: Precision値のリスト
            recalls: Recall値のリスト
    
        Returns:
            ap: Average Precision
        """
        import numpy as np
    
        # 11点補間
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            # Recall ≥ t におけるPrecisionの最大値
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
    
        return ap
    
    # 使用例
    pred_boxes = np.array([
        [50, 50, 150, 150],
        [55, 55, 155, 155],
        [200, 200, 300, 300]
    ])
    pred_scores = np.array([0.9, 0.7, 0.85])
    true_boxes = np.array([
        [52, 52, 152, 152],
        [205, 205, 305, 305]
    ])
    
    precisions, recalls = calculate_precision_recall(
        pred_boxes, pred_scores, true_boxes, iou_threshold=0.5
    )
    
    ap = calculate_ap(precisions, recalls)
    print(f"Average Precision: {ap:.4f}")
    
    # Precision-Recall曲線の可視化
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, marker='o', linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.fill_between(recalls, precisions, alpha=0.2)
    plt.text(0.5, 0.5, f'AP = {ap:.4f}', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.show()
    
    def calculate_map(all_precisions, all_recalls, num_classes):
        """
        mean Average Precision (mAP)を計算
    
        Args:
            all_precisions: 各クラスのPrecisionリスト [[p1, p2, ...], ...]
            all_recalls: 各クラスのRecallリスト [[r1, r2, ...], ...]
            num_classes: クラス数
    
        Returns:
            mAP: mean Average Precision
        """
        aps = []
    
        for i in range(num_classes):
            ap = calculate_ap(all_precisions[i], all_recalls[i])
            aps.append(ap)
            print(f"Class {i}: AP = {ap:.4f}")
    
        mAP = np.mean(aps)
        print(f"\nmAP: {mAP:.4f}")
    
        return mAP
    

> **COCO mAP** : COCOデータセットでは、複数のIoU閾値（0.5, 0.55, ..., 0.95）でAPを計算し、その平均を取る厳しい評価を行います。

* * *

## 5.5 PyTorchでの物体検出

### torchvision.models.detectionの活用

PyTorchのtorchvisionには、事前学習済みの物体検出モデルが豊富に用意されています。
    
    
    import torch
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        fasterrcnn_mobilenet_v3_large_fpn,
        retinanet_resnet50_fpn,
        ssd300_vgg16
    )
    
    def compare_detection_models():
        """
        各種物体検出モデルの比較
        """
        models_info = {
            'Faster R-CNN (ResNet-50)': {
                'model': fasterrcnn_resnet50_fpn,
                'type': 'Two-Stage',
                'speed': '遅',
                'accuracy': '高'
            },
            'Faster R-CNN (MobileNetV3)': {
                'model': fasterrcnn_mobilenet_v3_large_fpn,
                'type': 'Two-Stage',
                'speed': '中',
                'accuracy': '中'
            },
            'RetinaNet (ResNet-50)': {
                'model': retinanet_resnet50_fpn,
                'type': 'One-Stage',
                'speed': '中',
                'accuracy': '高'
            },
            'SSD300 (VGG16)': {
                'model': ssd300_vgg16,
                'type': 'One-Stage',
                'speed': '速',
                'accuracy': '中'
            }
        }
    
        print("物体検出モデル比較:")
        print("-" * 80)
        for name, info in models_info.items():
            print(f"{name:35s} | Type: {info['type']:10s} | "
                  f"Speed: {info['speed']:3s} | Accuracy: {info['accuracy']:3s}")
        print("-" * 80)
    
    compare_detection_models()
    
    # カスタムデータセットでのFine-tuning
    from torch.utils.data import Dataset, DataLoader
    import json
    
    class CustomDetectionDataset(Dataset):
        """
        カスタム物体検出データセット（COCO形式）
        """
    
        def __init__(self, image_dir, annotation_file, transforms=None):
            """
            Args:
                image_dir: 画像ディレクトリパス
                annotation_file: COCOフォーマットのアノテーションファイル
                transforms: データ拡張
            """
            self.image_dir = image_dir
            self.transforms = transforms
    
            # アノテーションの読み込み
            with open(annotation_file, 'r') as f:
                self.coco_data = json.load(f)
    
            self.images = self.coco_data['images']
            self.annotations = self.coco_data['annotations']
    
            # 画像IDごとにアノテーションをグループ化
            self.image_to_annotations = {}
            for ann in self.annotations:
                image_id = ann['image_id']
                if image_id not in self.image_to_annotations:
                    self.image_to_annotations[image_id] = []
                self.image_to_annotations[image_id].append(ann)
    
        def __len__(self):
            return len(self.images)
    
        def __getitem__(self, idx):
            # 画像情報
            img_info = self.images[idx]
            image_id = img_info['id']
            img_path = f"{self.image_dir}/{img_info['file_name']}"
    
            # 画像の読み込み
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
    
            # アノテーションの取得
            anns = self.image_to_annotations.get(image_id, [])
    
            boxes = []
            labels = []
    
            for ann in anns:
                # COCO形式: [x, y, width, height] → [x_min, y_min, x_max, y_max]
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
    
            # Tensorに変換
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
    
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([image_id])
            }
    
            # データ拡張
            if self.transforms:
                img = self.transforms(img)
    
            return img, target
    
    # データセットの使用例
    # dataset = CustomDetectionDataset(
    #     image_dir='data/images',
    #     annotation_file='data/annotations.json',
    #     transforms=torchvision.transforms.ToTensor()
    # )
    #
    # data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    

### 訓練ループの実装
    
    
    def train_detection_model(model, data_loader, optimizer, device, epoch):
        """
        物体検出モデルの訓練（1エポック）
    
        Args:
            model: 物体検出モデル
            data_loader: データローダー
            optimizer: オプティマイザー
            device: 実行デバイス
            epoch: 現在のエポック数
        """
        model.train()
    
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(data_loader):
            # データをデバイスに転送
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
            # 順伝播（torchvisionのモデルは訓練時にlossを返す）
            loss_dict = model(images, targets)
    
            # 全損失の合計
            losses = sum(loss for loss in loss_dict.values())
    
            # 逆伝播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
    
            total_loss += losses.item()
    
            # 進捗表示
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}, '
                      f'Loss: {losses.item():.4f}')
                print(f'  Details: {", ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])}')
    
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}\n')
    
        return avg_loss
    
    def evaluate_detection_model(model, data_loader, device):
        """
        物体検出モデルの評価
    
        Args:
            model: 物体検出モデル
            data_loader: データローダー
            device: 実行デバイス
    
        Returns:
            metrics: 評価指標の辞書
        """
        model.eval()
    
        all_predictions = []
        all_targets = []
    
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
    
                # 推論
                predictions = model(images)
    
                all_predictions.extend([{k: v.cpu() for k, v in p.items()} for p in predictions])
                all_targets.extend([{k: v.cpu() for k, v in t.items()} for t in targets])
    
        # 評価指標の計算（簡易版）
        print("評価結果:")
        print(f"  総サンプル数: {len(all_predictions)}")
    
        # 平均検出数
        avg_detections = sum(len(p['boxes']) for p in all_predictions) / len(all_predictions)
        print(f"  平均検出数: {avg_detections:.2f}")
    
        return {'avg_detections': avg_detections}
    
    # 訓練の実行例
    def full_training_pipeline(num_epochs=10):
        """
        完全な訓練パイプライン
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # モデルの作成
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        model.to(device)
    
        # オプティマイザー
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
        # Learning Rate Scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
        # 訓練ループ
        for epoch in range(1, num_epochs + 1):
            # 訓練
            # train_loss = train_detection_model(model, train_loader, optimizer, device, epoch)
    
            # 評価
            # metrics = evaluate_detection_model(model, val_loader, device)
    
            # Learning Rate更新
            lr_scheduler.step()
    
            # モデル保存
            # torch.save(model.state_dict(), f'detection_model_epoch_{epoch}.pth')
    
            print(f"Epoch {epoch} completed.\n")
    
    # 使用例
    # full_training_pipeline(num_epochs=10)
    

* * *

## 5.6 実践：COCO形式データでの検出

### COCOデータセットの概要

**COCO (Common Objects in Context)** は、物体検出の標準ベンチマークデータセットです。

項目 | 詳細  
---|---  
**画像数** | 訓練: 118K枚、検証: 5K枚、テスト: 41K枚  
**クラス数** | 80クラス（person, car, dog, etc.）  
**アノテーション** | Bounding Box、Segmentation、Keypoints  
**評価指標** | mAP @ IoU=[0.50:0.05:0.95]  
  
### 完全な物体検出パイプライン
    
    
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    class ObjectDetectionPipeline:
        """
        物体検出の完全なパイプライン
        """
    
        def __init__(self, num_classes, pretrained=True, device=None):
            """
            Args:
                num_classes: 検出クラス数（背景を含む）
                pretrained: 事前学習済み重みを使用
                device: 実行デバイス
            """
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.num_classes = num_classes
    
            # モデルの構築
            self.model = self._build_model(pretrained)
            self.model.to(self.device)
    
            print(f"物体検出パイプライン初期化完了")
            print(f"  デバイス: {self.device}")
            print(f"  クラス数: {num_classes}")
    
        def _build_model(self, pretrained):
            """モデルの構築"""
            model = fasterrcnn_resnet50_fpn(pretrained=pretrained)
    
            # 最終層を置き換え
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
    
            return model
    
        def predict(self, image_path, conf_threshold=0.5, nms_threshold=0.5):
            """
            画像から物体を検出
    
            Args:
                image_path: 入力画像パス
                conf_threshold: 信頼度スコアの閾値
                nms_threshold: NMSのIoU閾値
    
            Returns:
                detections: 検出結果の辞書
            """
            # 画像の読み込み
            img = Image.open(image_path).convert('RGB')
            img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(self.device)
    
            # 推論
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(img_tensor)
    
            # 後処理
            pred = predictions[0]
    
            # NMS（torchvisionのモデルは内部でNMSを実行するが、追加で適用可能）
            keep = torchvision.ops.nms(pred['boxes'], pred['scores'], nms_threshold)
    
            # 閾値フィルタリング
            keep = keep[pred['scores'][keep] > conf_threshold]
    
            detections = {
                'boxes': pred['boxes'][keep].cpu().numpy(),
                'labels': pred['labels'][keep].cpu().numpy(),
                'scores': pred['scores'][keep].cpu().numpy()
            }
    
            return detections, img
    
        def visualize(self, image, detections, class_names, save_path=None):
            """
            検出結果の可視化
    
            Args:
                image: PIL Image
                detections: predict()の返り値
                class_names: クラス名のリスト
                save_path: 保存先パス（Noneなら表示のみ）
            """
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(image)
    
            colors = plt.cm.hsv(np.linspace(0, 1, len(class_names))).tolist()
    
            for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
    
                color = colors[label % len(colors)]
    
                # Bounding Box
                rect = patches.Rectangle(
                    (x_min, y_min), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
    
                # ラベル
                label_text = f'{class_names[label]}: {score:.2f}'
                ax.text(
                    x_min, y_min - 5,
                    label_text,
                    bbox=dict(facecolor=color, alpha=0.7),
                    fontsize=10, color='white', weight='bold'
                )
    
            ax.axis('off')
            plt.tight_layout()
    
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"結果を保存: {save_path}")
            else:
                plt.show()
    
        def batch_predict(self, image_paths, conf_threshold=0.5):
            """
            バッチ推論
    
            Args:
                image_paths: 画像パスのリスト
                conf_threshold: 信頼度閾値
    
            Returns:
                all_detections: 各画像の検出結果リスト
            """
            all_detections = []
    
            for img_path in image_paths:
                detections, img = self.predict(img_path, conf_threshold)
                all_detections.append({
                    'path': img_path,
                    'detections': detections,
                    'image': img
                })
    
            return all_detections
    
        def evaluate_coco(self, data_loader, coco_gt):
            """
            COCO形式での評価
    
            Args:
                data_loader: データローダー
                coco_gt: COCO Ground Truth アノテーション
    
            Returns:
                metrics: 評価指標
            """
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
    
            self.model.eval()
            coco_results = []
    
            with torch.no_grad():
                for images, targets in data_loader:
                    images = [img.to(self.device) for img in images]
                    predictions = self.model(images)
    
                    # COCO形式に変換
                    for target, pred in zip(targets, predictions):
                        image_id = target['image_id'].item()
    
                        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                            x_min, y_min, x_max, y_max = box.tolist()
    
                            coco_results.append({
                                'image_id': image_id,
                                'category_id': label.item(),
                                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                                'score': score.item()
                            })
    
            # COCO評価
            coco_dt = coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
    
            metrics = {
                'mAP': coco_eval.stats[0],
                'mAP_50': coco_eval.stats[1],
                'mAP_75': coco_eval.stats[2]
            }
    
            return metrics
    
    # 使用例
    if __name__ == '__main__':
        # COCO クラス名（簡略版）
        COCO_CLASSES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag'
            # ... 全91クラス
        ]
    
        # パイプラインの初期化
        pipeline = ObjectDetectionPipeline(num_classes=91, pretrained=True)
    
        # 単一画像の推論
        # detections, img = pipeline.predict('test_image.jpg', conf_threshold=0.7)
        # pipeline.visualize(img, detections, COCO_CLASSES, save_path='result.jpg')
    
        # バッチ推論
        # image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        # results = pipeline.batch_predict(image_list, conf_threshold=0.6)
    
        print("物体検出パイプライン準備完了")
    

* * *

## 本章のまとめ

### 学んだこと

  1. **物体検出の基礎**

     * Classification、Detection、Segmentationの違い
     * Bounding BoxとIoUの計算方法
     * 物体検出の課題と評価指標
  2. **Two-Stage Detectors**

     * R-CNN、Fast R-CNN、Faster R-CNNの進化
     * Region Proposal Networkの仕組み
     * 精度重視のアプローチ
  3. **One-Stage Detectors**

     * YOLO、SSDの設計思想
     * 速度と精度のトレードオフ
     * リアルタイム検出の実現
  4. **評価指標**

     * NMS（Non-Maximum Suppression）の実装
     * Precision-Recall曲線とAP
     * mAP（mean Average Precision）の計算
  5. **実装スキル**

     * PyTorch torchvisionでの物体検出
     * 訓練と評価のパイプライン構築
     * COCO形式データの扱い方

### モデル選択ガイド

要件 | 推奨モデル | 理由  
---|---|---  
**最高精度** | Faster R-CNN (ResNet-101) | Two-stageで精密な検出  
**リアルタイム** | YOLOv5s / YOLOv8 | 140+ FPS、軽量  
**バランス型** | YOLOv5m / RetinaNet | 速度と精度の両立  
**エッジデバイス** | MobileNet SSD | 低計算量、省メモリ  
**小物体検出** | Faster R-CNN + FPN | Multi-scale特徴抽出  
  
* * *

## 演習問題

### 問題1（難易度：medium）

IoU計算関数をNumPyで実装し、以下のテストケースで検証してください：

  * Box1: [0, 0, 100, 100], Box2: [50, 50, 150, 150] → IoU ≈ 0.143
  * Box1: [0, 0, 100, 100], Box2: [0, 0, 100, 100] → IoU = 1.0
  * Box1: [0, 0, 50, 50], Box2: [60, 60, 100, 100] → IoU = 0.0

解答例
    
    
    import numpy as np
    
    def calculate_iou_numpy(box1, box2):
        """NumPyによるIoU計算"""
        # 交差領域
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])
    
        inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
    
        # 各Boxの面積
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
        # IoU
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
    
        return iou
    
    # テスト
    test_cases = [
        ([0, 0, 100, 100], [50, 50, 150, 150], 0.143),
        ([0, 0, 100, 100], [0, 0, 100, 100], 1.0),
        ([0, 0, 50, 50], [60, 60, 100, 100], 0.0)
    ]
    
    for box1, box2, expected in test_cases:
        iou = calculate_iou_numpy(box1, box2)
        print(f"Box1: {box1}, Box2: {box2}")
        print(f"  計算IoU: {iou:.4f}, 期待値: {expected:.4f}, 一致: {abs(iou - expected) < 0.001}")
    

### 問題2（難易度：hard）

NMS（Non-Maximum Suppression）アルゴリズムをゼロから実装し、以下のテストデータで動作確認してください：
    
    
    boxes = [[50, 50, 150, 150], [55, 55, 155, 155], [200, 200, 300, 300], [205, 205, 305, 305]]
    scores = [0.9, 0.85, 0.95, 0.88]
    iou_threshold = 0.5
    

期待される出力: インデックス [2, 0]（スコア順）が保持される

ヒント

  * スコアの降順でソート
  * 最高スコアのBoxを保持し、重複するBoxを除去
  * 繰り返し処理で全Boxを処理

### 問題3（難易度：medium）

Faster R-CNNを使って、カスタム画像で物体検出を実行し、結果を可視化してください。検出された物体のクラス名とスコアを表示してください。

解答例
    
    
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from PIL import Image
    import torchvision.transforms as T
    
    # モデルのロード
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    # 画像の読み込み
    img = Image.open('your_image.jpg').convert('RGB')
    img_tensor = T.ToTensor()(img).unsqueeze(0)
    
    # 推論
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # 結果の表示
    pred = predictions[0]
    for i, (box, label, score) in enumerate(zip(pred['boxes'], pred['labels'], pred['scores'])):
        if score > 0.5:
            print(f"検出 {i+1}: クラス={COCO_CLASSES[label]}, スコア={score:.3f}, Box={box.tolist()}")
    
    # 可視化
    # visualize_bounding_boxes関数を使用
    

### 問題4（難易度：hard）

YOLOv5を使って、動画ファイル（またはWebカメラ）からリアルタイムで物体検出を行うスクリプトを作成してください。検出結果をフレームごとに表示し、FPSも計測してください。

ヒント

  * OpenCVで動画を読み込む（cv2.VideoCapture）
  * 各フレームでYOLOv5推論を実行
  * time.time()でFPSを計測
  * 結果をcv2.imshow()で表示

* * *

## 参考文献

  1. Girshick, R., et al. (2014). "Rich feature hierarchies for accurate object detection and semantic segmentation." _CVPR_.
  2. Girshick, R. (2015). "Fast R-CNN." _ICCV_.
  3. Ren, S., et al. (2016). "Faster R-CNN: Towards real-time object detection with region proposal networks." _TPAMI_.
  4. Redmon, J., et al. (2016). "You only look once: Unified, real-time object detection." _CVPR_.
  5. Liu, W., et al. (2016). "SSD: Single shot multibox detector." _ECCV_.
  6. Lin, T.-Y., et al. (2014). "Microsoft COCO: Common objects in context." _ECCV_.
  7. Lin, T.-Y., et al. (2017). "Focal loss for dense object detection." _ICCV_. (RetinaNet)
  8. Jocher, G., et al. (2022). "YOLOv5: State-of-the-art object detection." _Ultralytics_.
