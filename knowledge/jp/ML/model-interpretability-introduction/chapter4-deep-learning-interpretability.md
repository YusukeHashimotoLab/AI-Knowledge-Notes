---
title: "第4章:ディープラーニングモデルの解釈"
chapter_title: "第4章:ディープラーニングモデルの解釈"
subtitle: CNNとTransformerの可視化手法 - Saliency Maps、Grad-CAM、Attention可視化
reading_time: 30-35分
difficulty: 中級〜上級
code_examples: 10
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます:

  * ✅ 勾配ベースの可視化手法（Saliency Maps、Gradient × Input、SmoothGrad）を理解する
  * ✅ Grad-CAMでCNNの注目領域を可視化できる
  * ✅ Integrated Gradientsで属性を計算できる
  * ✅ Transformerのアテンション機構を可視化できる
  * ✅ PyTorchとCaptumでディープラーニングモデルを解釈できる
  * ✅ 画像分類とテキスト分類のモデルをデバッグできる

* * *

## 4.1 Saliency Mapsと勾配ベース手法

### 概要

**Saliency Maps（顕著性マップ）** は、ニューラルネットワークの予測に対する入力の各ピクセルの重要度を可視化する手法です。

> 「入力画像のどの部分がモデルの予測に最も影響を与えているかを勾配から計算する」

### 勾配ベース手法の分類
    
    
    ```mermaid
    graph TD
        A[勾配ベース可視化] --> B[Vanilla Gradients]
        A --> C[Gradient × Input]
        A --> D[SmoothGrad]
        A --> E[Integrated Gradients]
    
        B --> B1[最もシンプル∂y/∂x]
        C --> C1[勾配と入力の積より鮮明]
        D --> D1[ノイズを加えて平均ノイズ除去]
        E --> E1[経路積分理論的保証]
    
        style A fill:#e3f2fd
        style B fill:#fff3e0
        style C fill:#f3e5f5
        style D fill:#e8f5e9
        style E fill:#ffe0b2
    ```

### Vanilla Gradients

出力 $y_c$ のクラス $c$ に対する入力 $x$ の勾配を計算します。

$$ S_c(x) = \frac{\partial y_c}{\partial x} $$

#### PyTorchによる実装
    
    
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 事前学習済みモデルのロード
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    
    # 画像の前処理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    def vanilla_gradients(image_path, model, target_class=None):
        """
        Vanilla Gradientsでsaliency mapを生成
    
        Args:
            image_path: 画像ファイルパス
            model: PyTorchモデル
            target_class: ターゲットクラス（Noneの場合は最も確率の高いクラス）
    
        Returns:
            saliency: saliency map
            pred_class: 予測クラス
        """
        # 画像の読み込みと前処理
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        img_tensor.requires_grad = True
    
        # フォワードパス
        output = model(img_tensor)
    
        # ターゲットクラスの決定
        if target_class is None:
            target_class = output.argmax(dim=1).item()
    
        # 勾配の計算
        model.zero_grad()
        output[0, target_class].backward()
    
        # Saliency mapの生成（勾配の絶対値の最大値）
        saliency = img_tensor.grad.data.abs().max(dim=1)[0]
        saliency = saliency.squeeze().cpu().numpy()
    
        return saliency, target_class
    
    # 使用例
    saliency, pred = vanilla_gradients('cat.jpg', model)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 元画像
    img = Image.open('cat.jpg')
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Saliency map
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title(f'Vanilla Gradients (Class: {pred})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### Gradient × Input

勾配と入力値の要素積を取ることで、より解釈しやすい可視化を得ます。

$$ S_c(x) = x \odot \frac{\partial y_c}{\partial x} $$
    
    
    def gradient_input(image_path, model, target_class=None):
        """
        Gradient × Inputでsaliency mapを生成
        """
        # 画像の読み込み
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        img_tensor.requires_grad = True
    
        # フォワードパス
        output = model(img_tensor)
    
        if target_class is None:
            target_class = output.argmax(dim=1).item()
    
        # 勾配の計算
        model.zero_grad()
        output[0, target_class].backward()
    
        # Gradient × Input
        saliency = (img_tensor.grad.data * img_tensor.data).abs().max(dim=1)[0]
        saliency = saliency.squeeze().cpu().numpy()
    
        return saliency, target_class
    
    # 比較
    saliency_vanilla, _ = vanilla_gradients('cat.jpg', model)
    saliency_gi, pred = gradient_input('cat.jpg', model)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img = Image.open('cat.jpg')
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(saliency_vanilla, cmap='hot')
    axes[1].set_title('Vanilla Gradients')
    axes[1].axis('off')
    
    axes[2].imshow(saliency_gi, cmap='hot')
    axes[2].set_title('Gradient × Input')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### SmoothGrad

ノイズを加えた複数のサンプルの勾配を平均化することで、ノイズを除去します。

$$ \hat{S}_c(x) = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial y_c}{\partial (x + \mathcal{N}(0, \sigma^2))} $$
    
    
    def smooth_grad(image_path, model, target_class=None,
                    n_samples=50, noise_level=0.15):
        """
        SmoothGradでsaliency mapを生成
    
        Args:
            n_samples: ノイズサンプル数
            noise_level: ノイズの標準偏差
        """
        # 画像の読み込み
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
    
        # ターゲットクラスの決定
        with torch.no_grad():
            output = model(img_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
    
        # ノイズを加えたサンプルの勾配を計算
        gradients = []
        for _ in range(n_samples):
            # ノイズの追加
            noise = torch.randn_like(img_tensor) * noise_level
            noisy_img = img_tensor + noise
            noisy_img.requires_grad = True
    
            # 勾配の計算
            output = model(noisy_img)
            model.zero_grad()
            output[0, target_class].backward()
    
            gradients.append(noisy_img.grad.data)
    
        # 平均化
        avg_gradient = torch.stack(gradients).mean(dim=0)
        saliency = avg_gradient.abs().max(dim=1)[0]
        saliency = saliency.squeeze().cpu().numpy()
    
        return saliency, target_class
    
    # 使用例
    saliency_smooth, pred = smooth_grad('cat.jpg', model, n_samples=50)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(Image.open('cat.jpg'))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(saliency_vanilla, cmap='hot')
    axes[1].set_title('Vanilla Gradients')
    axes[1].axis('off')
    
    axes[2].imshow(saliency_smooth, cmap='hot')
    axes[2].set_title('SmoothGrad')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    

* * *

## 4.2 Grad-CAM

### 概要

**Grad-CAM (Gradient-weighted Class Activation Mapping)** は、CNNの最終畳み込み層を利用して、クラス特有の注目領域を可視化します。

> 「畳み込み層の特徴マップを勾配で重み付けすることで、クラス判別に重要な領域を特定する」

### アルゴリズム
    
    
    ```mermaid
    graph LR
        A[入力画像] --> B[CNN]
        B --> C[最終畳み込み層特徴マップ A^k]
        C --> D[グローバル平均プーリング]
        D --> E[全結合層]
        E --> F[クラススコア y^c]
    
        F --> G[勾配計算∂y^c/∂A^k]
        G --> H[グローバル平均プーリングα_k^c]
        H --> I[重み付き和L = ReLU Σ α_k^c A^k]
        I --> J[Grad-CAM]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style J fill:#e8f5e9
    ```

1\. 最終畳み込み層の特徴マップ $A^k$ を取得

2\. クラス $c$ のスコア $y^c$ に対する $A^k$ の勾配を計算

$$ \alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k} $$

3\. 重み付き和とReLUを適用

$$ L_{Grad-CAM}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right) $$

### 実装例
    
    
    class GradCAM:
        """
        Grad-CAMの実装
        """
        def __init__(self, model, target_layer):
            """
            Args:
                model: PyTorchモデル
                target_layer: 可視化対象の層（最終畳み込み層）
            """
            self.model = model
            self.target_layer = target_layer
            self.gradients = None
            self.activations = None
    
            # フックの登録
            target_layer.register_forward_hook(self.save_activation)
            target_layer.register_backward_hook(self.save_gradient)
    
        def save_activation(self, module, input, output):
            """フォワードパスで活性化を保存"""
            self.activations = output.detach()
    
        def save_gradient(self, module, grad_input, grad_output):
            """バックワードパスで勾配を保存"""
            self.gradients = grad_output[0].detach()
    
        def generate_cam(self, image_tensor, target_class=None):
            """
            Grad-CAMを生成
    
            Args:
                image_tensor: 入力画像テンソル
                target_class: ターゲットクラス
    
            Returns:
                cam: Grad-CAM
                pred_class: 予測クラス
            """
            # フォワードパス
            output = self.model(image_tensor)
    
            if target_class is None:
                target_class = output.argmax(dim=1).item()
    
            # バックワードパス
            self.model.zero_grad()
            output[0, target_class].backward()
    
            # 重みの計算（グローバル平均プーリング）
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
    
            # 重み付き和
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
    
            # ReLU
            cam = torch.relu(cam)
    
            # 正規化
            cam = cam - cam.min()
            cam = cam / cam.max()
    
            # リサイズ
            cam = torch.nn.functional.interpolate(
                cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False
            )
    
            return cam.squeeze().cpu().numpy(), target_class
    
    # ResNet50でGrad-CAMを使用
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    
    # 最終畳み込み層を指定
    target_layer = model.layer4[-1].conv3
    
    # Grad-CAMインスタンスの作成
    gradcam = GradCAM(model, target_layer)
    
    # 画像の読み込み
    img = Image.open('cat.jpg').convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor.requires_grad = True
    
    # Grad-CAMの生成
    cam, pred_class = gradcam.generate_cam(img_tensor)
    
    # 可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 元画像
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Grad-CAM
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f'Grad-CAM (Class: {pred_class})')
    axes[1].axis('off')
    
    # オーバーレイ
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### Grad-CAM++

Grad-CAMの改良版で、複数のオブジェクトや小さいオブジェクトに対してより正確な可視化を提供します。

$$ \alpha_k^c = \sum_i \sum_j \left( \frac{\partial^2 y^c}{(\partial A_{ij}^k)^2} \cdot \text{ReLU}\left(\frac{\partial y^c}{\partial A_{ij}^k}\right) \right) $$
    
    
    class GradCAMPlusPlus(GradCAM):
        """
        Grad-CAM++の実装
        """
        def generate_cam(self, image_tensor, target_class=None):
            """Grad-CAM++を生成"""
            # フォワードパス
            output = self.model(image_tensor)
    
            if target_class is None:
                target_class = output.argmax(dim=1).item()
    
            # 1次勾配と2次勾配の計算
            self.model.zero_grad()
            output[0, target_class].backward(retain_graph=True)
    
            grad_1 = self.gradients.clone()
    
            # 2次勾配
            self.model.zero_grad()
            grad_1.backward(torch.ones_like(grad_1), retain_graph=True)
            grad_2 = self.gradients.clone()
    
            # 3次勾配
            self.model.zero_grad()
            grad_2.backward(torch.ones_like(grad_2))
            grad_3 = self.gradients.clone()
    
            # 重みの計算
            alpha_num = grad_2
            alpha_denom = 2.0 * grad_2 + (grad_3 * self.activations).sum(dim=(2, 3), keepdim=True)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    
            alpha = alpha_num / alpha_denom
            weights = (alpha * torch.relu(grad_1)).sum(dim=(2, 3), keepdim=True)
    
            # CAMの計算
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
    
            # 正規化
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
    
            # リサイズ
            cam = torch.nn.functional.interpolate(
                cam, size=image_tensor.shape[2:], mode='bilinear', align_corners=False
            )
    
            return cam.squeeze().cpu().numpy(), target_class
    

* * *

## 4.3 Integrated Gradients

### 概要

**Integrated Gradients** は、ベースライン（例:黒画像）から入力画像までの経路に沿って勾配を積分することで、各特徴の寄与度を計算します。

> 「経路積分により、属性の合計がモデルの出力差と一致する理論的保証を持つ」

### 数式

ベースライン $x'$ から入力 $x$ への経路を $\gamma(\alpha) = x' + \alpha \cdot (x - x')$ とすると:

$$ \text{IntegratedGrad}_i(x) = (x_i - x'_i) \int_{\alpha=0}^{1} \frac{\partial F(\gamma(\alpha))}{\partial x_i} d\alpha $$

実装では、積分をリーマン和で近似します:

$$ \text{IntegratedGrad}_i(x) \approx (x_i - x'_i) \sum_{k=1}^{m} \frac{\partial F(x' + \frac{k}{m}(x - x'))}{\partial x_i} \cdot \frac{1}{m} $$

### Captumライブラリによる実装
    
    
    from captum.attr import IntegratedGradients, visualization as viz
    import torch.nn.functional as F
    
    # モデルとデータの準備
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    
    # 画像の読み込み
    img = Image.open('cat.jpg').convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Integrated Gradientsインスタンスの作成
    ig = IntegratedGradients(model)
    
    # ターゲットクラスの取得
    output = model(img_tensor)
    pred_class = output.argmax(dim=1).item()
    
    # ベースラインの設定（黒画像）
    baseline = torch.zeros_like(img_tensor)
    
    # Integrated Gradientsの計算
    attributions = ig.attribute(img_tensor, baseline, target=pred_class, n_steps=50)
    
    # 可視化
    def visualize_attributions(img, attributions, pred_class):
        """
        Integrated Gradientsの可視化
        """
        # テンソルをnumpy配列に変換
        img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    
        # 正規化を元に戻す
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
    
        # 属性の処理
        attr = attributions.squeeze().cpu().permute(1, 2, 0).numpy()
    
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
        # 元画像
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    
        # 属性（ヒートマップ）
        attr_sum = np.abs(attr).sum(axis=2)
        im = axes[1].imshow(attr_sum, cmap='hot')
        axes[1].set_title(f'Integrated Gradients (Class: {pred_class})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
    
        # オーバーレイ
        axes[2].imshow(img_np)
        axes[2].imshow(attr_sum, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
        plt.tight_layout()
        plt.show()
    
    visualize_attributions(img, attributions, pred_class)
    

### 異なるベースラインの影響
    
    
    # 異なるベースラインでの比較
    baselines = {
        'Black': torch.zeros_like(img_tensor),
        'White': torch.ones_like(img_tensor),
        'Random': torch.randn_like(img_tensor),
        'Blur': None  # ガウシアンブラー画像
    }
    
    # ガウシアンブラーベースライン
    from torchvision.transforms import GaussianBlur
    blur_transform = GaussianBlur(kernel_size=51, sigma=50)
    img_blur = blur_transform(img)
    baselines['Blur'] = transform(img_blur).unsqueeze(0).to(device)
    
    # 各ベースラインで計算
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 元画像
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx, (name, baseline) in enumerate(baselines.items(), 1):
        # Integrated Gradientsの計算
        attr = ig.attribute(img_tensor, baseline, target=pred_class, n_steps=50)
    
        # 可視化
        attr_sum = attr.squeeze().cpu().abs().sum(dim=0).numpy()
        im = axes[idx].imshow(attr_sum, cmap='hot')
        axes[idx].set_title(f'Baseline: {name}')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx])
    
    # 最後の空欄を非表示
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()
    

### 手法の比較

手法 | 計算量 | 理論的保証 | ノイズ | 用途  
---|---|---|---|---  
Vanilla Gradients | 低 | なし | 多い | クイック分析  
SmoothGrad | 中 | なし | 少ない | ノイズ除去  
Grad-CAM | 低 | なし | 少ない | CNN可視化  
Integrated Gradients | 高 | あり | 少ない | 精密な属性  
  
* * *

## 4.4 Attention可視化

### 概要

**Attention機構** は、Transformerモデルの中核であり、入力の異なる部分間の関係性を学習します。アテンション重みを可視化することで、モデルが何に注目しているかを理解できます。

### Self-Attentionの数式

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

  * $Q$: Query（クエリ）
  * $K$: Key（キー）
  * $V$: Value（バリュー）
  * $d_k$: キーの次元

### BERTのアテンション可視化
    
    
    from transformers import BertTokenizer, BertModel
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # BERTモデルとトークナイザーのロード
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()
    
    def visualize_attention(text, layer=0, head=0):
        """
        BERTのアテンション重みを可視化
    
        Args:
            text: 入力テキスト
            layer: 可視化する層（0-11）
            head: 可視化するヘッド（0-11）
        """
        # トークン化
        inputs = tokenizer(text, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
        # フォワードパス
        with torch.no_grad():
            outputs = model(**inputs)
    
        # アテンション重みの取得
        # attentions: (layers, batch, heads, seq_len, seq_len)
        attention = outputs.attentions[layer][0, head].cpu().numpy()
    
        # 可視化
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens,
                    cmap='viridis', ax=ax, cbar_kws={'label': 'Attention Weight'})
        ax.set_title(f'BERT Attention (Layer {layer}, Head {head})')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    # 使用例
    text = "The cat sat on the mat"
    visualize_attention(text, layer=0, head=0)
    

### Multi-Head Attentionの可視化
    
    
    def visualize_multi_head_attention(text, layer=0):
        """
        複数のアテンションヘッドを同時に可視化
    
        Args:
            text: 入力テキスト
            layer: 可視化する層
        """
        # トークン化
        inputs = tokenizer(text, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
        # フォワードパス
        with torch.no_grad():
            outputs = model(**inputs)
    
        # 12個のヘッドを可視化
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
    
        for head in range(12):
            attention = outputs.attentions[layer][0, head].cpu().numpy()
    
            sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens,
                       cmap='viridis', ax=axes[head], cbar=False)
            axes[head].set_title(f'Head {head}')
            axes[head].set_xlabel('')
            axes[head].set_ylabel('')
    
            if head % 4 != 0:
                axes[head].set_yticklabels([])
            if head < 8:
                axes[head].set_xticklabels([])
            else:
                axes[head].set_xticklabels(tokens, rotation=45, ha='right')
    
        plt.suptitle(f'BERT Multi-Head Attention (Layer {layer})', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # 使用例
    visualize_multi_head_attention("The quick brown fox jumps over the lazy dog", layer=5)
    

### BertVizによる対話的可視化
    
    
    # BertVizのインストール: pip install bertviz
    
    from bertviz import head_view, model_view
    from transformers import AutoTokenizer, AutoModel
    
    # モデルとトークナイザーのロード
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    # テキスト
    text = "The cat sat on the mat because it was tired"
    
    # トークン化
    inputs = tokenizer(text, return_tensors='pt')
    
    # フォワードパス
    with torch.no_grad():
        outputs = model(**inputs)
    
    # トークンの取得
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Head View（各ヘッドのアテンション）
    head_view(outputs.attentions, tokens)
    
    # Model View（全層のアテンション）
    model_view(outputs.attentions, tokens)
    

### Vision Transformerのアテンション可視化
    
    
    from transformers import ViTModel, ViTFeatureExtractor
    from PIL import Image
    import requests
    
    # Vision Transformerのロード
    model_name = 'google/vit-base-patch16-224'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    vit_model = ViTModel.from_pretrained(model_name, output_attentions=True)
    vit_model.eval()
    
    def visualize_vit_attention(image_path, layer=-1, head=0):
        """
        Vision Transformerのアテンションを可視化
    
        Args:
            image_path: 画像パス
            layer: 層インデックス（-1で最終層）
            head: ヘッドインデックス
        """
        # 画像の読み込みと前処理
        image = Image.open(image_path).convert('RGB')
        inputs = feature_extractor(images=image, return_tensors='pt')
    
        # フォワードパス
        with torch.no_grad():
            outputs = vit_model(**inputs)
    
        # アテンション重みの取得
        attention = outputs.attentions[layer][0, head].cpu().numpy()
    
        # CLSトークンのアテンションを取得（最初のトークン）
        cls_attention = attention[0, 1:]  # CLSトークンから画像パッチへ
    
        # 14x14グリッドにリシェイプ（ViT-Base-Patch16-224の場合）
        num_patches = int(cls_attention.shape[0] ** 0.5)
        cls_attention = cls_attention.reshape(num_patches, num_patches)
    
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
        # 元画像
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    
        # アテンションマップ
        im = axes[1].imshow(cls_attention, cmap='hot')
        axes[1].set_title(f'CLS Attention (Layer {layer}, Head {head})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
    
        # オーバーレイ
        from scipy.ndimage import zoom
        attention_resized = zoom(cls_attention, 224/num_patches, order=1)
        axes[2].imshow(image)
        axes[2].imshow(attention_resized, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
        plt.tight_layout()
        plt.show()
    
    # 使用例
    visualize_vit_attention('cat.jpg', layer=-1, head=0)
    

* * *

## 4.5 エンドツーエンド実践例

### 画像分類モデルの解釈

実際のユースケースで複数の可視化手法を組み合わせて使用します。
    
    
    class ImageClassifierInterpreter:
        """
        画像分類モデルの包括的解釈ツール
        """
        def __init__(self, model, device='cuda'):
            self.model = model.to(device)
            self.device = device
            self.model.eval()
    
            # Grad-CAMの準備
            if hasattr(model, 'layer4'):  # ResNet系
                target_layer = model.layer4[-1].conv3
            else:
                target_layer = list(model.children())[-2]
    
            self.gradcam = GradCAM(model, target_layer)
    
            # Integrated Gradientsの準備
            self.ig = IntegratedGradients(model)
    
        def interpret(self, image_path, methods=['gradcam', 'ig', 'smoothgrad']):
            """
            複数の手法でモデルを解釈
    
            Args:
                image_path: 画像パス
                methods: 使用する手法のリスト
    
            Returns:
                results: 解釈結果の辞書
            """
            # 画像の読み込み
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            img_tensor.requires_grad = True
    
            # 予測
            with torch.no_grad():
                output = self.model(img_tensor)
                probs = F.softmax(output, dim=1)
                top5_probs, top5_idx = probs.topk(5)
    
            results = {
                'image': img,
                'predictions': {
                    'classes': top5_idx[0].cpu().numpy(),
                    'probabilities': top5_probs[0].cpu().numpy()
                }
            }
    
            # Grad-CAM
            if 'gradcam' in methods:
                cam, _ = self.gradcam.generate_cam(img_tensor, target_class=top5_idx[0, 0].item())
                results['gradcam'] = cam
    
            # Integrated Gradients
            if 'ig' in methods:
                baseline = torch.zeros_like(img_tensor)
                attr = self.ig.attribute(img_tensor, baseline, target=top5_idx[0, 0].item())
                results['integrated_gradients'] = attr.squeeze().cpu().abs().sum(dim=0).numpy()
    
            # SmoothGrad
            if 'smoothgrad' in methods:
                saliency, _ = smooth_grad(image_path, self.model, target_class=top5_idx[0, 0].item())
                results['smoothgrad'] = saliency
    
            return results
    
        def visualize(self, results):
            """解釈結果を可視化"""
            n_methods = len([k for k in results.keys() if k not in ['image', 'predictions']])
    
            fig, axes = plt.subplots(1, n_methods + 1, figsize=(5 * (n_methods + 1), 5))
    
            # 元画像と予測
            axes[0].imshow(results['image'])
            pred_text = f"Top predictions:\n"
            for idx, (cls, prob) in enumerate(zip(results['predictions']['classes'][:3],
                                                  results['predictions']['probabilities'][:3])):
                pred_text += f"{idx+1}. Class {cls}: {prob:.2%}\n"
            axes[0].set_title(pred_text, fontsize=10)
            axes[0].axis('off')
    
            # 各手法の結果
            idx = 1
            if 'gradcam' in results:
                axes[idx].imshow(results['image'])
                axes[idx].imshow(results['gradcam'], cmap='jet', alpha=0.5)
                axes[idx].set_title('Grad-CAM')
                axes[idx].axis('off')
                idx += 1
    
            if 'integrated_gradients' in results:
                im = axes[idx].imshow(results['integrated_gradients'], cmap='hot')
                axes[idx].set_title('Integrated Gradients')
                axes[idx].axis('off')
                plt.colorbar(im, ax=axes[idx], fraction=0.046)
                idx += 1
    
            if 'smoothgrad' in results:
                axes[idx].imshow(results['smoothgrad'], cmap='hot')
                axes[idx].set_title('SmoothGrad')
                axes[idx].axis('off')
                idx += 1
    
            plt.tight_layout()
            plt.show()
    
    # 使用例
    model = models.resnet50(pretrained=True)
    interpreter = ImageClassifierInterpreter(model)
    
    # 解釈の実行
    results = interpreter.interpret('cat.jpg', methods=['gradcam', 'ig', 'smoothgrad'])
    interpreter.visualize(results)
    

### テキスト分類モデルの解釈
    
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from captum.attr import LayerIntegratedGradients
    
    class TextClassifierInterpreter:
        """
        テキスト分類モデルの解釈ツール
        """
        def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
    
            # Layer Integrated Gradientsの準備
            self.lig = LayerIntegratedGradients(self.forward_func,
                                               self.model.distilbert.embeddings)
    
        def forward_func(self, inputs):
            """モデルのフォワード関数"""
            return self.model(inputs_embeds=inputs).logits
    
        def interpret(self, text, target_class=None):
            """
            テキストの解釈
    
            Args:
                text: 入力テキスト
                target_class: ターゲットクラス（Noneで予測クラス）
    
            Returns:
                attributions: 各トークンの重要度
                tokens: トークンリスト
                prediction: 予測結果
            """
            # トークン化
            inputs = self.tokenizer(text, return_tensors='pt')
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
            # 予測
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                pred_class = outputs.logits.argmax(dim=1).item()
                pred_prob = probs[0, pred_class].item()
    
            if target_class is None:
                target_class = pred_class
    
            # Integrated Gradientsの計算
            input_embeds = self.model.distilbert.embeddings(inputs['input_ids'])
            baseline = torch.zeros_like(input_embeds)
    
            attributions = self.lig.attribute(
                input_embeds,
                baseline,
                target=target_class,
                n_steps=50
            )
    
            # トークンごとの属性にまとめる
            attributions_sum = attributions.sum(dim=-1).squeeze(0)
            attributions_sum = attributions_sum / torch.norm(attributions_sum)
            attributions_sum = attributions_sum.cpu().detach().numpy()
    
            return {
                'tokens': tokens,
                'attributions': attributions_sum,
                'prediction': {
                    'class': pred_class,
                    'probability': pred_prob,
                    'label': self.model.config.id2label[pred_class]
                }
            }
    
        def visualize(self, text, target_class=None):
            """解釈結果を可視化"""
            results = self.interpret(text, target_class)
    
            tokens = results['tokens']
            attributions = results['attributions']
    
            # 正規化（可視化用）
            attr_min, attr_max = attributions.min(), attributions.max()
            attributions_norm = (attributions - attr_min) / (attr_max - attr_min + 1e-8)
    
            # カラーマップ
            import matplotlib.cm as cm
            colors = cm.RdYlGn(attributions_norm)
    
            # テキスト表示
            fig, ax = plt.subplots(figsize=(15, 3))
            ax.axis('off')
    
            # 予測結果
            pred_text = f"Prediction: {results['prediction']['label']} ({results['prediction']['probability']:.2%})"
            ax.text(0.5, 0.9, pred_text, ha='center', va='top', fontsize=14, fontweight='bold',
                    transform=ax.transAxes)
    
            # トークンと重要度
            x_pos = 0.05
            for token, attr, color in zip(tokens, attributions_norm, colors):
                if token in ['[CLS]', '[SEP]', '[PAD]']:
                    continue
    
                # 背景色
                bbox_props = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='none')
                ax.text(x_pos, 0.5, token, ha='left', va='center', fontsize=12,
                       bbox=bbox_props, transform=ax.transAxes)
    
                # 重要度スコア
                ax.text(x_pos, 0.2, f'{attr:.3f}', ha='left', va='center', fontsize=8,
                       transform=ax.transAxes)
    
                x_pos += len(token) * 0.015 + 0.02
    
                if x_pos > 0.95:
                    break
    
            # カラーバー
            sm = plt.cm.ScalarMappable(cmap=cm.RdYlGn, norm=plt.Normalize(vmin=attr_min, vmax=attr_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
            cbar.set_label('Attribution Score', fontsize=10)
    
            plt.tight_layout()
            plt.show()
    
    # 使用例
    text_interpreter = TextClassifierInterpreter()
    
    # ポジティブな感情
    text_interpreter.visualize("This movie is absolutely fantastic and amazing!")
    
    # ネガティブな感情
    text_interpreter.visualize("This is the worst film I have ever seen.")
    

### モデルデバッグの実践
    
    
    def debug_model_prediction(model, image_path, true_label, expected_label):
        """
        誤分類のデバッグ
    
        Args:
            model: 分類モデル
            image_path: 画像パス
            true_label: 正解ラベル
            expected_label: 期待されるラベル
        """
        interpreter = ImageClassifierInterpreter(model)
    
        # 解釈
        results = interpreter.interpret(image_path, methods=['gradcam', 'ig'])
    
        pred_class = results['predictions']['classes'][0]
        pred_prob = results['predictions']['probabilities'][0]
    
        print(f"=== モデルデバッグレポート ===")
        print(f"正解ラベル: {true_label}")
        print(f"予測ラベル: {pred_class} (確率: {pred_prob:.2%})")
        print(f"期待ラベル: {expected_label}")
    
        if pred_class != expected_label:
            print(f"\n❌ 誤分類を検出しました")
            print(f"\nTop-5予測:")
            for idx, (cls, prob) in enumerate(zip(results['predictions']['classes'],
                                                  results['predictions']['probabilities'])):
                marker = "✓" if cls == true_label else " "
                print(f"  {marker} {idx+1}. クラス {cls}: {prob:.2%}")
    
            # 可視化
            interpreter.visualize(results)
    
            # 解釈
            print("\n=== 解釈 ===")
            print("Grad-CAMを確認してください:")
            print("- モデルは画像のどの領域に注目していますか?")
            print("- 注目領域は正解ラベルに対して妥当ですか?")
            print("- 背景やノイズに注目していませんか?")
        else:
            print(f"\n✓ 正しく分類されました")
            interpreter.visualize(results)
    
    # 使用例
    model = models.resnet50(pretrained=True)
    debug_model_prediction(model, 'dog.jpg', true_label=254, expected_label=254)
    

* * *

## 4.6 本章のまとめ

### 学んだこと

  1. **勾配ベース手法**

     * Vanilla Gradients: シンプルな勾配可視化
     * Gradient × Input: より鮮明な可視化
     * SmoothGrad: ノイズ除去
  2. **Grad-CAM**

     * CNNの注目領域を可視化
     * 最終畳み込み層の活用
     * Grad-CAM++による改良
  3. **Integrated Gradients**

     * 経路積分による属性計算
     * 理論的保証を持つ手法
     * ベースライン選択の重要性
  4. **Attention可視化**

     * TransformerのSelf-Attention
     * Multi-Head Attentionの解釈
     * BertVizによる対話的可視化
  5. **実践的応用**

     * 画像分類の解釈
     * テキスト分類の解釈
     * モデルデバッグの手法

### 次の章へ

第5章では、**モデル解釈の実務応用** を学びます:

  * モデル監査とバイアス検出
  * 規制対応（GDPR、AI Act）
  * ステークホルダーへの説明
  * 継続的モニタリング

* * *

## 演習問題

### 問題1（難易度: easy）

Vanilla GradientsとGrad-CAMの主な違いを3つ挙げてください。

解答例

**解答** :

  1. **使用する情報** : Vanilla Gradientsは入力に対する勾配のみ、Grad-CAMは最終畳み込み層の特徴マップと勾配を使用
  2. **解像度** : Vanilla Gradientsは入力画像と同じ解像度、Grad-CAMは低解像度から補間
  3. **ノイズ** : Vanilla Gradientsはノイズが多い、Grad-CAMは滑らかで解釈しやすい

### 問題2（難易度: medium）

SmoothGradを実装し、ノイズサンプル数（n_samples）が結果に与える影響を調べてください。

解答例
    
    
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 異なるサンプル数での比較
    n_samples_list = [10, 25, 50, 100]
    
    fig, axes = plt.subplots(1, len(n_samples_list) + 1, figsize=(20, 4))
    
    # 元画像
    img = Image.open('cat.jpg')
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for idx, n_samples in enumerate(n_samples_list, 1):
        saliency, _ = smooth_grad('cat.jpg', model, n_samples=n_samples)
    
        axes[idx].imshow(saliency, cmap='hot')
        axes[idx].set_title(f'n_samples={n_samples}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    

**観察** :

  * n_samplesが小さい（10）: ノイズが残る
  * n_samplesが適度（50）: 滑らかで鮮明
  * n_samplesが大きい（100）: 計算コストが高いが、さらに滑らか
  * **推奨** : 実務では50前後が良いバランス

### 問題3（難易度: medium）

Integrated Gradientsで異なるベースライン（黒、白、ブラー）を使用した場合、結果がどのように変わるか説明してください。

解答例

**解答** :

ベースライン | 特徴 | 適用場面  
---|---|---  
黒画像 | 全ピクセルが0  
最も一般的 | 通常の画像分類  
白画像 | 全ピクセルが1  
黒背景の画像で有効 | 医療画像など  
ブラー画像 | 構造は保持、詳細は失う  
より現実的 | テクスチャが重要な場合  
ランダムノイズ | 無秩序な画像  
参照として | 比較検証  
  
**ベースライン選択のガイドライン** :

  * 一般的には黒画像が推奨
  * ドメイン知識に基づいて選択
  * 複数のベースラインで結果を比較
  * 結果が一貫しているか確認

### 問題4（難易度: hard）

BERTのアテンション重みを可視化し、"The cat sat on the mat"という文で、各単語が他のどの単語に注目しているかを分析してください。

解答例
    
    
    from transformers import BertTokenizer, BertModel
    import torch
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    # モデルとトークナイザーのロード
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    model.eval()
    
    text = "The cat sat on the mat"
    
    # トークン化
    inputs = tokenizer(text, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # フォワードパス
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 全層の平均アテンション
    all_attentions = torch.stack([att.squeeze(0) for att in outputs.attentions])
    avg_attention = all_attentions.mean(dim=[0, 1]).cpu().numpy()  # 層とヘッドで平均
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(avg_attention, xticklabels=tokens, yticklabels=tokens,
                cmap='viridis', ax=ax, cbar_kws={'label': 'Average Attention'})
    ax.set_title('BERT Average Attention Across All Layers and Heads')
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # 分析
    print("=== アテンション分析 ===\n")
    
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
    
        # 各トークンが最も注目している単語（自分自身を除く）
        attention_weights = avg_attention[i].copy()
        attention_weights[i] = -1  # 自分自身を除外
        max_idx = np.argmax(attention_weights)
    
        print(f"'{token}' が最も注目: '{tokens[max_idx]}' (重み: {attention_weights[max_idx]:.3f})")
    
    print("\n観察:")
    print("- 'cat'は'sat'に注目（主語と動詞の関係）")
    print("- 'sat'は'cat'と'on'に注目（動詞が主語と前置詞に）")
    print("- 'mat'は'the'と'on'に注目（名詞が冠詞と前置詞に）")
    

**期待される観察結果** :

  * **構文的関係** : 主語と動詞、修飾語と被修飾語が相互に注目
  * **局所性** : 近隣のトークンへの注目が強い
  * **文法的パターン** : 前置詞が後続の名詞に注目

### 問題5（難易度: hard）

誤分類された画像に対してGrad-CAMとIntegrated Gradientsを適用し、誤分類の原因を分析するコードを書いてください。

解答例
    
    
    import torch
    import torchvision.models as models
    from captum.attr import IntegratedGradients
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    class MisclassificationAnalyzer:
        """
        誤分類分析ツール
        """
        def __init__(self, model, device='cuda'):
            self.model = model.to(device)
            self.device = device
            self.model.eval()
    
            # Grad-CAMの準備
            self.gradcam = GradCAM(model, model.layer4[-1].conv3)
    
            # Integrated Gradientsの準備
            self.ig = IntegratedGradients(model)
    
        def analyze_misclassification(self, image_path, true_label, imagenet_labels):
            """
            誤分類の詳細分析
    
            Args:
                image_path: 画像パス
                true_label: 正解ラベル
                imagenet_labels: ImageNetラベル辞書
            """
            # 画像の読み込み
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            img_tensor.requires_grad = True
    
            # 予測
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            pred_label = output.argmax(dim=1).item()
    
            print(f"{'='*60}")
            print(f"誤分類分析レポート")
            print(f"{'='*60}")
            print(f"\n正解: {imagenet_labels[true_label]}")
            print(f"予測: {imagenet_labels[pred_label]} (確率: {probs[0, pred_label]:.2%})")
    
            # Top-5予測
            top5_probs, top5_idx = probs.topk(5)
            print(f"\nTop-5予測:")
            for i, (idx, prob) in enumerate(zip(top5_idx[0], top5_probs[0])):
                marker = "✓" if idx == true_label else " "
                print(f"  {marker} {i+1}. {imagenet_labels[idx.item()]}: {prob:.2%}")
    
            # 正解クラスのスコア
            true_prob = probs[0, true_label]
            true_rank = (probs[0] > true_prob).sum().item() + 1
            print(f"\n正解クラスのランク: {true_rank}位 (確率: {true_prob:.2%})")
    
            # Grad-CAM（予測クラス）
            cam_pred, _ = self.gradcam.generate_cam(img_tensor, target_class=pred_label)
    
            # Grad-CAM（正解クラス）
            cam_true, _ = self.gradcam.generate_cam(img_tensor, target_class=true_label)
    
            # Integrated Gradients
            baseline = torch.zeros_like(img_tensor)
            attr_pred = self.ig.attribute(img_tensor, baseline, target=pred_label, n_steps=50)
            attr_true = self.ig.attribute(img_tensor, baseline, target=true_label, n_steps=50)
    
            # 可視化
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
            # 元画像
            axes[0, 0].imshow(img)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
    
            # 予測クラスのGrad-CAM
            axes[0, 1].imshow(img)
            axes[0, 1].imshow(cam_pred, cmap='jet', alpha=0.5)
            axes[0, 1].set_title(f'Grad-CAM: Predicted\n{imagenet_labels[pred_label]}')
            axes[0, 1].axis('off')
    
            # 正解クラスのGrad-CAM
            axes[0, 2].imshow(img)
            axes[0, 2].imshow(cam_true, cmap='jet', alpha=0.5)
            axes[0, 2].set_title(f'Grad-CAM: True\n{imagenet_labels[true_label]}')
            axes[0, 2].axis('off')
    
            # スペーサー
            axes[1, 0].axis('off')
    
            # 予測クラスのIG
            attr_pred_sum = attr_pred.squeeze().cpu().abs().sum(dim=0).numpy()
            axes[1, 1].imshow(attr_pred_sum, cmap='hot')
            axes[1, 1].set_title(f'IG: Predicted\n{imagenet_labels[pred_label]}')
            axes[1, 1].axis('off')
    
            # 正解クラスのIG
            attr_true_sum = attr_true.squeeze().cpu().abs().sum(dim=0).numpy()
            axes[1, 2].imshow(attr_true_sum, cmap='hot')
            axes[1, 2].set_title(f'IG: True\n{imagenet_labels[true_label]}')
            axes[1, 2].axis('off')
    
            plt.tight_layout()
            plt.show()
    
            # 診断
            print(f"\n{'='*60}")
            print(f"診断")
            print(f"{'='*60}")
            print("1. Grad-CAMを比較:")
            print("   - 予測クラスと正解クラスで注目領域が異なりますか?")
            print("   - 予測クラスは背景やノイズに注目していますか?")
            print("\n2. Integrated Gradientsを確認:")
            print("   - 正解クラスに必要な特徴が画像に存在しますか?")
            print("   - 予測クラスの誤った特徴が強く現れていますか?")
            print("\n3. 考えられる原因:")
            if true_prob < 0.01:
                print("   - モデルが正解クラスをほとんど考慮していない")
                print("   - データセットに類似例が不足している可能性")
            elif true_rank <= 5:
                print("   - 正解クラスは上位にある（境界ケース）")
                print("   - クラス間の類似性が高い可能性")
            else:
                print("   - 画像の品質や前処理に問題がある可能性")
                print("   - オブジェクトの一部が隠れている可能性")
    
    # ImageNetラベルの読み込み（簡略版）
    imagenet_labels = {
        254: 'Pug',
        281: 'Tabby Cat',
        # ... 他のラベル
    }
    
    # 使用例
    model = models.resnet50(pretrained=True)
    analyzer = MisclassificationAnalyzer(model)
    
    # 誤分類された画像を分析
    analyzer.analyze_misclassification('pug.jpg', true_label=254, imagenet_labels=imagenet_labels)
    

**出力例** :
    
    
    ============================================================
    誤分類分析レポート
    ============================================================
    
    正解: Pug
    予測: Tabby Cat (確率: 45.23%)
    
    Top-5予測:
       1. Tabby Cat: 45.23%
       2. Egyptian Cat: 23.45%
      ✓ 3. Pug: 12.34%
       4. Bulldog: 8.90%
       5. Chihuahua: 5.67%
    
    正解クラスのランク: 3位 (確率: 12.34%)
    
    ============================================================
    診断
    ============================================================
    1. Grad-CAMを比較:
       - 予測クラスと正解クラスで注目領域が異なりますか?
       - 予測クラスは背景やノイズに注目していますか?
    
    2. Integrated Gradientsを確認:
       - 正解クラスに必要な特徴が画像に存在しますか?
       - 予測クラスの誤った特徴が強く現れていますか?
    
    3. 考えられる原因:
       - 正解クラスは上位にある（境界ケース）
       - クラス間の類似性が高い可能性
    

* * *

## 参考文献

  1. Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). _Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps_. arXiv:1312.6034.
  2. Selvaraju, R. R., et al. (2017). _Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization_. ICCV 2017.
  3. Sundararajan, M., Taly, A., & Yan, Q. (2017). _Axiomatic Attribution for Deep Networks_. ICML 2017.
  4. Smilkov, D., et al. (2017). _SmoothGrad: removing noise by adding noise_. arXiv:1706.03825.
  5. Chattopadhay, A., et al. (2018). _Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks_. WACV 2018.
  6. Vaswani, A., et al. (2017). _Attention is All You Need_. NeurIPS 2017.
  7. Vig, J. (2019). _A Multiscale Visualization of Attention in the Transformer Model_. ACL 2019.
  8. Natekar, P., & Sharma, M. (2020). _Captum: A unified and generic model interpretability library for PyTorch_. arXiv:2009.07896.
