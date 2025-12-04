---
title: 第5章：生成モデルの応用 (Applications of Generative Models)
chapter_title: 第5章：生成モデルの応用 (Applications of Generative Models)
subtitle: Text-to-Image生成からアバター作成システムまでの実践的応用
reading_time: 25-30分
difficulty: 中級〜上級
code_examples: 7
exercises: 5
---

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ Stable Diffusionを使ったText-to-Image生成を実装できる
  * ✅ プロンプトエンジニアリングの技術を理解し、効果的な画像を生成できる
  * ✅ Image-to-Image変換（Style transfer、Super-resolution）を実装できる
  * ✅ Conditional GAN（cGAN）による条件付き生成の仕組みを理解できる
  * ✅ Audio生成技術（WaveGAN）の基礎を理解できる
  * ✅ 実用的なアバター生成システムを構築できる
  * ✅ 生成AIの倫理的課題と責任ある利用方法を理解できる

* * *

## 5.1 Text-to-Image生成

### Stable Diffusionの概要

**Stable Diffusion** は、テキストプロンプトから高品質な画像を生成する拡散モデルです。2022年にStability AIによって公開され、オープンソースで利用可能な最も強力なText-to-Image生成モデルの一つです。
    
    
    ```mermaid
    graph LR
        A[テキストプロンプト] --> B[CLIP Text Encoder]
        B --> C[Text Embedding77×768]
        C --> D[U-Net Denoiser]
        E[ランダムノイズLatent Space] --> D
        D --> F[Denoising Steps20-50回]
        F --> G[VAE Decoder]
        G --> H[生成画像512×512 or 1024×1024]
    
        style A fill:#e3f2fd
        style H fill:#c8e6c9
        style D fill:#fff9c4
    ```

#### Stable Diffusionのアーキテクチャ構成要素

コンポーネント | 役割 | 技術詳細  
---|---|---  
**Text Encoder** | テキストを埋め込みベクトルに変換 | CLIP ViT-L/14（OpenAI）  
**VAE Encoder/Decoder** | 画像とLatent空間の変換 | 圧縮率8×、512×512→64×64  
**U-Net Denoiser** | ノイズ除去と画像生成 | Cross-attention機構でテキスト条件付け  
**Scheduler** | ノイズスケジュール管理 | DDPM, DDIM, Euler, DPM-Solver++  
  
### Stable Diffusionの実装
    
    
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from PIL import Image
    import matplotlib.pyplot as plt
    
    class StableDiffusionGenerator:
        """
        Stable Diffusionを使ったText-to-Image生成クラス
    
        Features:
        - 複数のスケジューラサポート（DDPM, DDIM, Euler, DPM-Solver++）
        - Negative promptサポート
        - CFG（Classifier-Free Guidance）制御
        - シード固定による再現性
        """
    
        def __init__(self, model_id="stabilityai/stable-diffusion-2-1", device="cuda"):
            """
            Args:
                model_id: HuggingFaceのモデルID
                device: 使用デバイス（cuda or cpu）
            """
            self.device = device if torch.cuda.is_available() else "cpu"
    
            # パイプラインの初期化
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None  # 実運用では適切なフィルタリングを実装
            )
    
            # より高速なDPM-Solver++スケジューラを使用
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
    
            self.pipe = self.pipe.to(self.device)
    
            # メモリ最適化（GPU使用時）
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
    
        def generate(
            self,
            prompt,
            negative_prompt="",
            num_inference_steps=25,
            guidance_scale=7.5,
            width=512,
            height=512,
            seed=None,
            num_images=1
        ):
            """
            テキストプロンプトから画像を生成
    
            Args:
                prompt: 生成したい画像の説明文
                negative_prompt: 避けたい要素の説明文
                num_inference_steps: ノイズ除去のステップ数（20-50推奨）
                guidance_scale: CFGスケール（7-15推奨、高いほどプロンプトに忠実）
                width, height: 生成画像のサイズ（8の倍数）
                seed: 再現性のためのシード値
                num_images: 生成する画像の枚数
    
            Returns:
                生成された画像のリスト
            """
            # シード設定
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
    
            # 画像生成
            with torch.autocast(self.device):
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    num_images_per_prompt=num_images
                )
    
            return output.images
    
        def generate_grid(self, prompts, **kwargs):
            """
            複数のプロンプトから画像グリッドを生成
    
            Args:
                prompts: プロンプトのリスト
                **kwargs: generateメソッドへの追加引数
    
            Returns:
                グリッド画像
            """
            images = []
            for prompt in prompts:
                img = self.generate(prompt, num_images=1, **kwargs)[0]
                images.append(img)
    
            # グリッド作成
            n = len(images)
            cols = int(n ** 0.5)
            rows = (n + cols - 1) // cols
    
            w, h = images[0].size
            grid = Image.new('RGB', (w * cols, h * rows))
    
            for idx, img in enumerate(images):
                grid.paste(img, ((idx % cols) * w, (idx // cols) * h))
    
            return grid
    
    # 使用例
    if __name__ == "__main__":
        # ジェネレーターの初期化
        sd = StableDiffusionGenerator()
    
        # 基本的な生成
        prompt = "A beautiful sunset over mountains, oil painting style, highly detailed"
        negative_prompt = "blurry, low quality, distorted"
    
        images = sd.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=42,
            num_images=2
        )
    
        # 画像表示
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for idx, img in enumerate(images):
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f'Image {idx + 1}')
        plt.tight_layout()
        plt.show()
    
        # 複数プロンプトでグリッド生成
        prompts = [
            "A cat astronaut in space, digital art",
            "A futuristic city at night, cyberpunk style",
            "A magical forest with glowing mushrooms",
            "A steampunk robot playing violin"
        ]
    
        grid = sd.generate_grid(
            prompts,
            negative_prompt="ugly, blurry, low quality",
            num_inference_steps=25,
            guidance_scale=7.5,
            seed=42
        )
    
        plt.figure(figsize=(10, 10))
        plt.imshow(grid)
        plt.axis('off')
        plt.title('Generated Image Grid')
        plt.show()
    

### プロンプトエンジニアリング技術

**プロンプトエンジニアリング** は、望む画像を生成するためにテキストプロンプトを最適化する技術です。効果的なプロンプトの構成要素を理解することが重要です。

#### 効果的なプロンプトの構造

$$ \text{Prompt} = \text{Subject} + \text{Style} + \text{Quality} + \text{Details} + \text{Modifiers} $$

要素 | 説明 | 例  
---|---|---  
**Subject** | メインの被写体 | "a majestic lion", "a futuristic building"  
**Style** | 芸術スタイル | "oil painting", "anime style", "photorealistic"  
**Quality** | 品質修飾子 | "highly detailed", "8k resolution", "masterpiece"  
**Details** | 具体的な詳細 | "golden hour lighting", "dramatic shadows"  
**Modifiers** | 追加の調整 | "trending on artstation", "by Greg Rutkowski"  
      
    
    class PromptEngineer:
        """
        効果的なプロンプトを構築するためのヘルパークラス
        """
    
        # プロンプトテンプレート
        STYLE_KEYWORDS = {
            'photorealistic': 'photorealistic, photo, realistic, high quality photograph',
            'digital_art': 'digital art, digital painting, artstation',
            'oil_painting': 'oil painting, traditional art, canvas',
            'anime': 'anime style, manga, japanese animation',
            'cyberpunk': 'cyberpunk style, neon lights, futuristic',
            '3d_render': '3d render, octane render, unreal engine, blender'
        }
    
        QUALITY_KEYWORDS = [
            'highly detailed',
            '8k resolution',
            'masterpiece',
            'best quality',
            'sharp focus',
            'professional'
        ]
    
        NEGATIVE_KEYWORDS = [
            'blurry',
            'low quality',
            'bad anatomy',
            'distorted',
            'ugly',
            'duplicate',
            'watermark'
        ]
    
        @staticmethod
        def build_prompt(
            subject,
            style='photorealistic',
            quality_level='high',
            additional_details=None,
            artist=None
        ):
            """
            構造化されたプロンプトを構築
    
            Args:
                subject: メインの被写体
                style: スタイルキーワード
                quality_level: 品質レベル（'high', 'medium', 'low'）
                additional_details: 追加の詳細（リストまたは文字列）
                artist: アーティスト名（オプション）
    
            Returns:
                構築されたプロンプト
            """
            components = [subject]
    
            # スタイル追加
            if style in PromptEngineer.STYLE_KEYWORDS:
                components.append(PromptEngineer.STYLE_KEYWORDS[style])
            else:
                components.append(style)
    
            # 品質キーワード追加
            if quality_level == 'high':
                components.extend(PromptEngineer.QUALITY_KEYWORDS[:4])
            elif quality_level == 'medium':
                components.extend(PromptEngineer.QUALITY_KEYWORDS[:2])
    
            # 追加詳細
            if additional_details:
                if isinstance(additional_details, list):
                    components.extend(additional_details)
                else:
                    components.append(additional_details)
    
            # アーティスト名
            if artist:
                components.append(f"by {artist}")
    
            return ", ".join(components)
    
        @staticmethod
        def build_negative_prompt(custom_negatives=None):
            """
            Negative promptを構築
    
            Args:
                custom_negatives: カスタムネガティブキーワード
    
            Returns:
                Negative prompt文字列
            """
            negatives = PromptEngineer.NEGATIVE_KEYWORDS.copy()
            if custom_negatives:
                negatives.extend(custom_negatives)
            return ", ".join(negatives)
    
        @staticmethod
        def optimize_for_faces(base_prompt):
            """
            顔の生成に特化したプロンプト最適化
            """
            face_keywords = [
                'detailed face',
                'perfect eyes',
                'symmetrical face',
                'professional portrait',
                'sharp facial features'
            ]
            return f"{base_prompt}, {', '.join(face_keywords)}"
    
        @staticmethod
        def optimize_for_landscapes(base_prompt):
            """
            風景画に特化したプロンプト最適化
            """
            landscape_keywords = [
                'wide angle',
                'epic vista',
                'atmospheric',
                'dramatic lighting',
                'depth of field'
            ]
            return f"{base_prompt}, {', '.join(landscape_keywords)}"
    
    # 使用例
    if __name__ == "__main__":
        pe = PromptEngineer()
    
        # ポートレート生成
        portrait_prompt = pe.build_prompt(
            subject="a young woman with flowing red hair",
            style="digital_art",
            quality_level="high",
            additional_details=["golden hour lighting", "soft shadows"],
            artist="Ilya Kuvshinov"
        )
        portrait_prompt = pe.optimize_for_faces(portrait_prompt)
    
        # 風景画生成
        landscape_prompt = pe.build_prompt(
            subject="a serene mountain lake surrounded by pine trees",
            style="oil_painting",
            quality_level="high",
            additional_details=["misty morning", "reflections on water"]
        )
        landscape_prompt = pe.optimize_for_landscapes(landscape_prompt)
    
        # Negative prompt
        negative = pe.build_negative_prompt(["deformed", "disfigured"])
    
        print("Portrait Prompt:")
        print(portrait_prompt)
        print("\nLandscape Prompt:")
        print(landscape_prompt)
        print("\nNegative Prompt:")
        print(negative)
    

> **プロンプトのベストプラクティス** :
> 
>   * **具体的に記述** : 「美しい風景」より「雪山に囲まれた澄んだ湖、夕暮れの光」
>   * **品質キーワード活用** : "highly detailed", "8k", "masterpiece"など
>   * **Negative prompt活用** : 避けたい要素を明示的に指定
>   * **重み付け使用** : (keyword:1.5)で強調、(keyword:0.8)で弱める
>   * **アーティスト参照** : 特定のスタイルを得るため著名アーティスト名を使用
> 

### CFG（Classifier-Free Guidance）の理論

CFGは、条件付き生成において品質を向上させる技術です。条件付きモデルと無条件モデルの予測を組み合わせます。

$$ \epsilon_\theta(z_t, c, t) = \epsilon_\theta(z_t, \emptyset, t) + s \cdot (\epsilon_\theta(z_t, c, t) - \epsilon_\theta(z_t, \emptyset, t)) $$

ここで：

  * $\epsilon_\theta(z_t, c, t)$: 条件$c$（テキスト）付きノイズ予測
  * $\epsilon_\theta(z_t, \emptyset, t)$: 無条件ノイズ予測
  * $s$: ガイダンススケール（通常7〜15）
  * $z_t$: 時刻$t$のLatent変数

    
    
    ```mermaid
    graph TB
        A[Input: Noisy Latent z_t] --> B[Conditional Pathwith Text c]
        A --> C[Unconditional Pathno text]
    
        B --> D[ε_cond]
        C --> E[ε_uncond]
    
        D --> F[Guidance Calculation]
        E --> F
    
        F --> G[ε_guided = ε_uncond + s × Δε]
        G --> H[Denoising Step]
    
        I[Guidance Scale s] --> F
    
        style A fill:#e3f2fd
        style H fill:#c8e6c9
        style I fill:#fff9c4
    ```

* * *

## 5.2 Image-to-Image変換

### Style Transfer（スタイル転送）

**Style Transfer** は、ある画像のスタイル（色彩、筆触、質感）を別の画像のコンテンツに適用する技術です。Stable Diffusionでは、初期ノイズの代わりに既存画像を使用します。
    
    
    from diffusers import StableDiffusionImg2ImgPipeline
    from PIL import Image
    import torch
    
    class StyleTransferGenerator:
        """
        Stable Diffusionを使ったImage-to-Image変換クラス
    
        Features:
        - Style transfer
        - Image variation生成
        - Strength制御による変換度合い調整
        """
    
        def __init__(self, model_id="stabilityai/stable-diffusion-2-1", device="cuda"):
            self.device = device if torch.cuda.is_available() else "cpu"
    
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            )
    
            self.pipe = self.pipe.to(self.device)
    
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
    
        def transfer_style(
            self,
            input_image,
            style_prompt,
            strength=0.75,
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=None
        ):
            """
            画像にスタイルを適用
    
            Args:
                input_image: 入力画像（PIL ImageまたはパスString）
                style_prompt: 適用したいスタイルの説明
                strength: 変換の強度（0.0-1.0、高いほど大きく変化）
                guidance_scale: CFGスケール
                num_inference_steps: ステップ数
                seed: ランダムシード
    
            Returns:
                スタイル変換後の画像
            """
            # 画像読み込み
            if isinstance(input_image, str):
                input_image = Image.open(input_image).convert('RGB')
    
            # リサイズ（8の倍数に）
            w, h = input_image.size
            w = (w // 8) * 8
            h = (h // 8) * 8
            input_image = input_image.resize((w, h))
    
            # シード設定
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
    
            # スタイル転送
            with torch.autocast(self.device):
                output = self.pipe(
                    prompt=style_prompt,
                    image=input_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                )
    
            return output.images[0]
    
        def create_variations(
            self,
            input_image,
            prompt,
            num_variations=4,
            strength=0.5,
            **kwargs
        ):
            """
            入力画像のバリエーションを生成
    
            Args:
                input_image: 入力画像
                prompt: 変換の方向性を示すプロンプト
                num_variations: 生成するバリエーション数
                strength: 変換強度
    
            Returns:
                バリエーション画像のリスト
            """
            variations = []
            for i in range(num_variations):
                seed = kwargs.get('seed', None)
                if seed is not None:
                    seed = seed + i
    
                var_img = self.transfer_style(
                    input_image,
                    prompt,
                    strength=strength,
                    seed=seed,
                    **{k: v for k, v in kwargs.items() if k != 'seed'}
                )
                variations.append(var_img)
    
            return variations
    
    # 使用例
    if __name__ == "__main__":
        st = StyleTransferGenerator()
    
        # スタイル転送の例
        input_image = "path/to/photo.jpg"
    
        style_prompts = [
            "oil painting in the style of Van Gogh, swirling brushstrokes",
            "anime style, Studio Ghibli aesthetic, vibrant colors",
            "cyberpunk style, neon lights, futuristic",
            "watercolor painting, soft colors, artistic"
        ]
    
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    
        # 元画像表示
        original = Image.open(input_image)
        axes[0].imshow(original)
        axes[0].set_title('Original')
        axes[0].axis('off')
    
        # 各スタイルで転送
        for idx, style_prompt in enumerate(style_prompts):
            styled_img = st.transfer_style(
                input_image,
                style_prompt,
                strength=0.75,
                num_inference_steps=50,
                seed=42
            )
    
            axes[idx + 1].imshow(styled_img)
            axes[idx + 1].set_title(style_prompt[:30] + '...')
            axes[idx + 1].axis('off')
    
        # 最後のセルは非表示
        axes[-1].axis('off')
    
        plt.tight_layout()
        plt.show()
    

### Super-Resolution（超解像）

**Super-Resolution** は、低解像度画像から高解像度画像を生成する技術です。拡散モデルベースのアプローチが最先端の性能を示しています。
    
    
    import torch
    import torch.nn as nn
    from diffusers import StableDiffusionUpscalePipeline
    from PIL import Image
    
    class SuperResolutionModel:
        """
        Stable Diffusion Upscalerを使った超解像クラス
    
        Features:
        - 4倍アップスケーリング
        - ノイズ除去と詳細補完
        - プロンプトによる品質制御
        """
    
        def __init__(self, model_id="stabilityai/stable-diffusion-x4-upscaler", device="cuda"):
            self.device = device if torch.cuda.is_available() else "cpu"
    
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
    
            self.pipe = self.pipe.to(self.device)
    
            if self.device == "cuda":
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()
    
        def upscale(
            self,
            input_image,
            prompt="high quality, detailed",
            num_inference_steps=50,
            guidance_scale=7.5,
            noise_level=20,
            seed=None
        ):
            """
            画像を4倍にアップスケール
    
            Args:
                input_image: 低解像度入力画像
                prompt: 品質向上のプロンプト
                num_inference_steps: ステップ数
                guidance_scale: CFGスケール
                noise_level: ノイズレベル（0-100、高いほどより多くの詳細を生成）
                seed: ランダムシード
    
            Returns:
                アップスケールされた画像
            """
            # 画像読み込み
            if isinstance(input_image, str):
                input_image = Image.open(input_image).convert('RGB')
    
            # シード設定
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
    
            # アップスケーリング
            with torch.autocast(self.device):
                upscaled = self.pipe(
                    prompt=prompt,
                    image=input_image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    noise_level=noise_level,
                    generator=generator
                ).images[0]
    
            return upscaled
    
        def progressive_upscale(self, input_image, target_size, **kwargs):
            """
            段階的アップスケーリング（非常に大きなサイズ向け）
    
            Args:
                input_image: 入力画像
                target_size: 目標サイズ (width, height)
    
            Returns:
                アップスケールされた画像
            """
            if isinstance(input_image, str):
                input_image = Image.open(input_image).convert('RGB')
    
            current_img = input_image
            current_size = current_img.size
    
            while current_size[0] < target_size[0] or current_size[1] < target_size[1]:
                # 4倍アップスケール
                current_img = self.upscale(current_img, **kwargs)
                current_size = current_img.size
    
                print(f"Upscaled to: {current_size}")
    
                # 目標サイズを超えたら終了
                if current_size[0] >= target_size[0] and current_size[1] >= target_size[1]:
                    break
    
            # 最終的に目標サイズにリサイズ
            if current_size != target_size:
                current_img = current_img.resize(target_size, Image.LANCZOS)
    
            return current_img
    
    # 使用例
    if __name__ == "__main__":
        sr = SuperResolutionModel()
    
        # 低解像度画像をアップスケール
        low_res_image = "path/to/low_res.jpg"
    
        # 異なるノイズレベルで比較
        noise_levels = [10, 20, 40, 60]
    
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
    
        # 元画像
        original = Image.open(low_res_image)
        axes[0].imshow(original)
        axes[0].set_title(f'Original ({original.size[0]}x{original.size[1]})')
        axes[0].axis('off')
    
        # 各ノイズレベルでアップスケール
        for idx, noise_level in enumerate(noise_levels):
            upscaled = sr.upscale(
                low_res_image,
                prompt="high quality, sharp, detailed, professional photograph",
                noise_level=noise_level,
                seed=42
            )
    
            axes[idx + 1].imshow(upscaled)
            axes[idx + 1].set_title(f'Noise Level {noise_level}\n({upscaled.size[0]}x{upscaled.size[1]})')
            axes[idx + 1].axis('off')
    
        axes[-1].axis('off')
    
        plt.tight_layout()
        plt.show()
    

* * *

## 5.3 条件付き生成 (Conditional Generation)

### Conditional GAN (cGAN)

**Conditional GAN** は、クラスラベルや属性情報などの条件に基づいて画像を生成するGANの拡張です。GeneratorとDiscriminatorの両方が条件情報を受け取ります。
    
    
    ```mermaid
    graph TB
        A[Random Noise z] --> G[Generator G]
        B[Condition cClass Label] --> G
        G --> C[Fake Image x̃]
    
        D[Real Image x] --> Disc[Discriminator D]
        C --> Disc
        B --> Disc
    
        Disc --> E[Real/Fake + Class]
    
        style A fill:#e3f2fd
        style B fill:#fff9c4
        style C fill:#ffccbc
        style D fill:#c8e6c9
        style E fill:#f8bbd0
    ```

#### cGANの目的関数

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x|c)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z|c)|c))] $$

ここで：

  * $c$: 条件情報（クラスラベル、テキスト、画像など）
  * $D(x|c)$: 条件$c$を考慮した識別
  * $G(z|c)$: 条件$c$に基づく生成

    
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class ConditionalGenerator(nn.Module):
        """
        Conditional GANのGenerator
    
        条件情報（クラスラベル）を埋め込みとして結合し、
        条件付き画像生成を実現
        """
    
        def __init__(self, latent_dim=100, num_classes=10, img_size=32, channels=3):
            super(ConditionalGenerator, self).__init__()
    
            self.latent_dim = latent_dim
            self.num_classes = num_classes
            self.img_size = img_size
    
            # クラス埋め込み
            self.label_emb = nn.Embedding(num_classes, latent_dim)
    
            # Generator本体
            self.init_size = img_size // 4
            self.fc = nn.Linear(latent_dim * 2, 128 * self.init_size ** 2)
    
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
    
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
    
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
    
                nn.Conv2d(64, channels, 3, stride=1, padding=1),
                nn.Tanh()
            )
    
        def forward(self, noise, labels):
            """
            Args:
                noise: ランダムノイズ (batch_size, latent_dim)
                labels: クラスラベル (batch_size,)
    
            Returns:
                生成画像 (batch_size, channels, img_size, img_size)
            """
            # クラス埋め込みを取得
            label_embedding = self.label_emb(labels)  # (batch_size, latent_dim)
    
            # ノイズとラベル埋め込みを結合
            gen_input = torch.cat([noise, label_embedding], dim=1)  # (batch_size, latent_dim*2)
    
            # 全結合層
            out = self.fc(gen_input)
            out = out.view(out.size(0), 128, self.init_size, self.init_size)
    
            # 畳み込み層
            img = self.conv_blocks(out)
    
            return img
    
    
    class ConditionalDiscriminator(nn.Module):
        """
        Conditional GANのDiscriminator
    
        画像とクラスラベルを入力として受け取り、
        Real/Fakeの識別を行う
        """
    
        def __init__(self, num_classes=10, img_size=32, channels=3):
            super(ConditionalDiscriminator, self).__init__()
    
            self.num_classes = num_classes
            self.img_size = img_size
    
            # クラス埋め込み（画像サイズに展開）
            self.label_emb = nn.Embedding(num_classes, img_size * img_size)
    
            # Discriminator本体
            self.conv_blocks = nn.Sequential(
                nn.Conv2d(channels + 1, 64, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
    
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
    
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            )
    
            # 出力層のサイズを計算
            ds_size = img_size // 2 ** 3
            self.adv_layer = nn.Sequential(
                nn.Linear(256 * ds_size ** 2, 1),
                nn.Sigmoid()
            )
    
        def forward(self, img, labels):
            """
            Args:
                img: 入力画像 (batch_size, channels, img_size, img_size)
                labels: クラスラベル (batch_size,)
    
            Returns:
                識別スコア (batch_size, 1)
            """
            # クラス埋め込みを画像サイズに変形
            label_embedding = self.label_emb(labels)  # (batch_size, img_size*img_size)
            label_embedding = label_embedding.view(-1, 1, self.img_size, self.img_size)
    
            # 画像とラベル埋め込みを結合
            d_in = torch.cat([img, label_embedding], dim=1)  # (batch_size, channels+1, H, W)
    
            # 畳み込み層
            out = self.conv_blocks(d_in)
            out = out.view(out.size(0), -1)
    
            # 識別
            validity = self.adv_layer(out)
    
            return validity
    
    
    class ConditionalGANTrainer:
        """
        Conditional GANの訓練クラス
        """
    
        def __init__(self, latent_dim=100, num_classes=10, img_size=32, channels=3, device='cuda'):
            self.device = device
            self.latent_dim = latent_dim
            self.num_classes = num_classes
    
            # モデル初期化
            self.generator = ConditionalGenerator(
                latent_dim, num_classes, img_size, channels
            ).to(device)
    
            self.discriminator = ConditionalDiscriminator(
                num_classes, img_size, channels
            ).to(device)
    
            # 最適化器
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
            # 損失関数
            self.criterion = nn.BCELoss()
    
        def train_step(self, real_imgs, labels):
            """
            1訓練ステップ
    
            Args:
                real_imgs: 実画像バッチ
                labels: 対応するクラスラベル
    
            Returns:
                d_loss, g_loss: DiscriminatorとGeneratorの損失
            """
            batch_size = real_imgs.size(0)
    
            # ラベル
            real_label = torch.ones(batch_size, 1, device=self.device)
            fake_label = torch.zeros(batch_size, 1, device=self.device)
    
            # ---------------------
            # Discriminatorの訓練
            # ---------------------
            self.optimizer_D.zero_grad()
    
            # Real画像の識別
            real_validity = self.discriminator(real_imgs, labels)
            d_real_loss = self.criterion(real_validity, real_label)
    
            # Fake画像の生成と識別
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            gen_labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
            fake_imgs = self.generator(z, gen_labels)
            fake_validity = self.discriminator(fake_imgs.detach(), gen_labels)
            d_fake_loss = self.criterion(fake_validity, fake_label)
    
            # Discriminator損失
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            self.optimizer_D.step()
    
            # -----------------
            # Generatorの訓練
            # -----------------
            self.optimizer_G.zero_grad()
    
            # Generatorの損失（Discriminatorを騙す）
            gen_validity = self.discriminator(fake_imgs, gen_labels)
            g_loss = self.criterion(gen_validity, real_label)
    
            g_loss.backward()
            self.optimizer_G.step()
    
            return d_loss.item(), g_loss.item()
    
        def generate_samples(self, num_samples=10, class_id=None):
            """
            サンプル画像を生成
    
            Args:
                num_samples: 生成数
                class_id: 特定クラスを生成（Noneの場合はランダム）
    
            Returns:
                生成画像とラベル
            """
            self.generator.eval()
    
            with torch.no_grad():
                z = torch.randn(num_samples, self.latent_dim, device=self.device)
    
                if class_id is not None:
                    labels = torch.full((num_samples,), class_id, dtype=torch.long, device=self.device)
                else:
                    labels = torch.randint(0, self.num_classes, (num_samples,), device=self.device)
    
                gen_imgs = self.generator(z, labels)
    
            self.generator.train()
            return gen_imgs, labels
    
    # 使用例
    if __name__ == "__main__":
        # MNIST風の設定
        trainer = ConditionalGANTrainer(
            latent_dim=100,
            num_classes=10,
            img_size=28,
            channels=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
        # 各クラスのサンプルを生成
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.flatten()
    
        for class_id in range(10):
            gen_imgs, _ = trainer.generate_samples(num_samples=1, class_id=class_id)
            img = gen_imgs[0].cpu().squeeze().numpy()
    
            axes[class_id].imshow(img, cmap='gray')
            axes[class_id].set_title(f'Class {class_id}')
            axes[class_id].axis('off')
    
        plt.tight_layout()
        plt.show()
    

* * *

## 5.4 Audio生成

### WaveGAN概要

**WaveGAN** は、生の音声波形を直接生成するGANです。画像生成GANのアーキテクチャを1次元畳み込みに適応させています。
    
    
    ```mermaid
    graph LR
        A[Random Noise100-dim] --> B[FC Layer16×256]
        B --> C[Reshape256×16]
        C --> D[Transposed Conv1D×5Upsample]
        D --> E[Output16384 samples1 second @ 16kHz]
    
        F[Real Audio] --> G[Conv1D×5Downsample]
        E --> G
        G --> H[FC Layer] --> I[Real/Fake]
    
        style A fill:#e3f2fd
        style E fill:#c8e6c9
        style I fill:#f8bbd0
    ```

#### WaveGANの特徴

特徴 | 画像GAN | WaveGAN  
---|---|---  
**畳み込み** | 2D Conv | 1D Conv（時間軸）  
**サンプル長** | 64×64ピクセル | 16384サンプル（1秒@16kHz）  
**アップサンプリング** | 2倍ずつ | 4倍、8倍、16倍など  
**正規化** | Batch Norm | Phase Shuffle（位相シャッフル）  
  
> **Phase Shuffle** : WaveGANの重要な技術で、訓練中にランダムに位相をシフトさせることで、Discriminatorが特定の位相に過学習するのを防ぎます。これによりアーティファクトの少ない自然な音声が生成されます。

* * *

## 5.5 実践プロジェクト: アバター生成システム

### アバター生成の要件定義

実用的なアバター生成システムには以下の機能が必要です：

  * **多様性** : 異なる顔立ち、髪型、服装のバリエーション
  * **一貫性** : 同一人物の異なるポーズや表情
  * **制御性** : ユーザー指定の属性（髪色、目の色など）
  * **品質** : 高解像度で自然な見た目

    
    
    ```mermaid
    graph TB
        A[User Input] --> B[Text Prompt Builder]
        A --> C[Attribute SelectorHair, Eyes, Style]
    
        B --> D[Stable Diffusion Pipeline]
        C --> D
    
        D --> E[Initial Generation512×512]
        E --> F[Face Detection &Alignment]
        F --> G[Super Resolution2048×2048]
    
        G --> H{Quality Check}
        H -->|Pass| I[Final Avatar]
        H -->|Fail| D
    
        style A fill:#e3f2fd
        style I fill:#c8e6c9
        style H fill:#fff9c4
    ```
    
    
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    class AvatarGenerationSystem:
        """
        包括的なアバター生成システム
    
        Features:
        - カスタマイズ可能な属性（髪、目、スタイルなど）
        - 一貫性のある複数ポーズ生成
        - 自動品質チェック
        - 超解像による高品質化
        """
    
        # 属性オプション
        HAIR_STYLES = ['long flowing', 'short', 'curly', 'straight', 'wavy', 'braided']
        HAIR_COLORS = ['blonde', 'brunette', 'black', 'red', 'white', 'blue', 'pink']
        EYE_COLORS = ['blue', 'green', 'brown', 'hazel', 'gray', 'amber']
        STYLES = ['anime', 'realistic', 'cartoon', 'semi-realistic', 'fantasy']
        EXPRESSIONS = ['smiling', 'serious', 'happy', 'calm', 'mysterious']
        BACKGROUNDS = ['simple background', 'gradient background', 'nature background', 'abstract background']
    
        def __init__(self, device="cuda"):
            self.device = device if torch.cuda.is_available() else "cpu"
    
            # Text-to-Imageパイプライン
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            ).to(self.device)
    
            # Upscalerパイプライン
            self.upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
    
            if self.device == "cuda":
                self.sd_pipe.enable_attention_slicing()
                self.upscale_pipe.enable_attention_slicing()
    
        def build_avatar_prompt(
            self,
            gender='female',
            hair_style=None,
            hair_color=None,
            eye_color=None,
            style=None,
            expression=None,
            background=None,
            additional_features=None
        ):
            """
            アバタープロンプトを構築
    
            Args:
                gender: 'male' or 'female'
                hair_style, hair_color, eye_color: 髪と目の属性
                style: アートスタイル
                expression: 表情
                background: 背景
                additional_features: 追加の特徴（リスト）
    
            Returns:
                構築されたプロンプト
            """
            # ランダム選択（指定がない場合）
            hair_style = hair_style or random.choice(self.HAIR_STYLES)
            hair_color = hair_color or random.choice(self.HAIR_COLORS)
            eye_color = eye_color or random.choice(self.EYE_COLORS)
            style = style or random.choice(self.STYLES)
            expression = expression or random.choice(self.EXPRESSIONS)
            background = background or random.choice(self.BACKGROUNDS)
    
            # プロンプト構築
            components = [
                f"portrait of a {gender}",
                f"{hair_color} {hair_style} hair",
                f"{eye_color} eyes",
                expression,
                background,
                f"{style} style",
                "highly detailed face",
                "professional digital art",
                "8k quality",
                "perfect anatomy",
                "beautiful lighting"
            ]
    
            # 追加特徴
            if additional_features:
                components.extend(additional_features)
    
            prompt = ", ".join(components)
    
            # Negative prompt
            negative_prompt = ", ".join([
                "blurry", "low quality", "distorted", "deformed",
                "bad anatomy", "disfigured", "ugly", "duplicate",
                "extra limbs", "mutation", "watermark"
            ])
    
            return prompt, negative_prompt
    
        def generate_avatar(
            self,
            prompt=None,
            negative_prompt=None,
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=None,
            upscale=False,
            **prompt_kwargs
        ):
            """
            アバターを生成
    
            Args:
                prompt: カスタムプロンプト（Noneの場合は自動構築）
                negative_prompt: Negative prompt
                num_inference_steps: ステップ数
                guidance_scale: CFGスケール
                seed: ランダムシード
                upscale: 超解像を適用するか
                **prompt_kwargs: build_avatar_promptへの引数
    
            Returns:
                生成されたアバター画像
            """
            # プロンプト構築
            if prompt is None:
                prompt, auto_negative = self.build_avatar_prompt(**prompt_kwargs)
                negative_prompt = negative_prompt or auto_negative
    
            # シード設定
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
    
            # 初期生成
            with torch.autocast(self.device):
                output = self.sd_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=512,
                    height=512
                )
    
            avatar = output.images[0]
    
            # 超解像（オプション）
            if upscale:
                with torch.autocast(self.device):
                    avatar = self.upscale_pipe(
                        prompt=prompt,
                        image=avatar,
                        num_inference_steps=50,
                        guidance_scale=7.5,
                        noise_level=20
                    ).images[0]
    
            return avatar, prompt
    
        def generate_avatar_set(
            self,
            num_avatars=4,
            consistent_style=True,
            upscale=False,
            seed=None,
            **base_kwargs
        ):
            """
            一貫性のあるアバターセットを生成
    
            Args:
                num_avatars: 生成数
                consistent_style: スタイルを統一するか
                upscale: 超解像を適用
                seed: ベースシード
                **base_kwargs: 共通の属性
    
            Returns:
                アバター画像のリスト、プロンプトのリスト
            """
            avatars = []
            prompts = []
    
            # 一貫性のための固定パラメータ
            if consistent_style:
                style = base_kwargs.get('style') or random.choice(self.STYLES)
                base_kwargs['style'] = style
    
            for i in range(num_avatars):
                current_seed = seed + i if seed is not None else None
    
                # 各アバター生成（表情を変える）
                avatar, prompt = self.generate_avatar(
                    expression=random.choice(self.EXPRESSIONS),
                    seed=current_seed,
                    upscale=upscale,
                    **base_kwargs
                )
    
                avatars.append(avatar)
                prompts.append(prompt)
    
            return avatars, prompts
    
        def create_avatar_sheet(self, avatars, prompts, grid_size=(2, 2)):
            """
            アバターシートを作成（複数アバターをグリッド表示）
    
            Args:
                avatars: アバター画像のリスト
                prompts: 対応するプロンプトのリスト
                grid_size: グリッドサイズ (rows, cols)
    
            Returns:
                グリッド画像
            """
            rows, cols = grid_size
            w, h = avatars[0].size
    
            # テキスト用のスペースを追加
            text_height = 100
            grid = Image.new('RGB', (w * cols, h * rows + text_height * rows), 'white')
            draw = ImageDraw.Draw(grid)
    
            for idx, (avatar, prompt) in enumerate(zip(avatars[:rows*cols], prompts[:rows*cols])):
                row = idx // cols
                col = idx % cols
    
                # アバター貼り付け
                grid.paste(avatar, (col * w, row * (h + text_height)))
    
                # プロンプトテキスト（簡略版）
                short_prompt = prompt[:60] + '...' if len(prompt) > 60 else prompt
                text_y = row * (h + text_height) + h + 10
                draw.text((col * w + 10, text_y), short_prompt, fill='black')
    
            return grid
    
    # 使用例
    if __name__ == "__main__":
        avatar_system = AvatarGenerationSystem()
    
        # 単一アバター生成
        avatar, prompt = avatar_system.generate_avatar(
            gender='female',
            hair_color='pink',
            eye_color='blue',
            style='anime',
            expression='smiling',
            seed=42,
            upscale=False
        )
    
        print(f"Generated avatar with prompt: {prompt}")
        avatar.show()
    
        # アバターセット生成（一貫性あり）
        avatars, prompts = avatar_system.generate_avatar_set(
            num_avatars=4,
            gender='male',
            style='realistic',
            hair_color='black',
            consistent_style=True,
            seed=100
        )
    
        # アバターシート作成
        sheet = avatar_system.create_avatar_sheet(avatars, prompts, grid_size=(2, 2))
        sheet.show()
    
        # ランダムバリエーション
        random_avatars = []
        for i in range(6):
            avatar, _ = avatar_system.generate_avatar(seed=i*10)
            random_avatars.append(avatar)
    
        random_sheet = avatar_system.create_avatar_sheet(
            random_avatars,
            ['Random Avatar'] * 6,
            grid_size=(2, 3)
        )
        random_sheet.show()
    

### アートワーク作成システム
    
    
    class ArtworkCreationSystem:
        """
        芸術作品生成システム
    
        Features:
        - 様々なアートスタイル（油絵、水彩、デジタルアートなど）
        - 構図制御
        - 色彩パレット指定
        - アーティストスタイル模倣
        """
    
        ART_STYLES = {
            'oil_painting': 'oil painting on canvas, thick brush strokes, impasto technique',
            'watercolor': 'watercolor painting, soft colors, transparent layers, paper texture',
            'digital_art': 'digital art, digital painting, trending on artstation, highly detailed',
            'impressionism': 'impressionist style, loose brushwork, emphasis on light, outdoor scene',
            'surrealism': 'surrealist art, dreamlike, bizarre imagery, subconscious inspiration',
            'abstract': 'abstract art, non-representational, geometric shapes, bold colors',
            'minimalist': 'minimalist art, simple composition, limited color palette, negative space',
            'cyberpunk': 'cyberpunk art, neon colors, futuristic, high tech low life aesthetic'
        }
    
        COMPOSITIONS = {
            'rule_of_thirds': 'rule of thirds composition, balanced',
            'symmetrical': 'symmetrical composition, centered, mirror-like',
            'diagonal': 'diagonal composition, dynamic, movement',
            'golden_ratio': 'golden ratio composition, harmonious proportions',
            'minimalist': 'minimalist composition, lots of negative space'
        }
    
        COLOR_PALETTES = {
            'warm': 'warm color palette, reds, oranges, yellows',
            'cool': 'cool color palette, blues, greens, purples',
            'monochromatic': 'monochromatic color scheme, shades of single color',
            'complementary': 'complementary colors, high contrast',
            'pastel': 'pastel colors, soft, muted tones',
            'vibrant': 'vibrant colors, saturated, bold'
        }
    
        def __init__(self, device="cuda"):
            self.device = device if torch.cuda.is_available() else "cpu"
    
            self.sd_pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None
            ).to(self.device)
    
            if self.device == "cuda":
                self.sd_pipe.enable_attention_slicing()
    
        def create_artwork(
            self,
            subject,
            art_style='digital_art',
            composition='rule_of_thirds',
            color_palette='vibrant',
            artist_reference=None,
            mood=None,
            additional_details=None,
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=None,
            size=(768, 768)
        ):
            """
            芸術作品を生成
    
            Args:
                subject: 主題（例: "a mountain landscape", "a cat"）
                art_style: アートスタイル
                composition: 構図
                color_palette: 色彩パレット
                artist_reference: 参照アーティスト名
                mood: 雰囲気（例: "melancholic", "joyful"）
                additional_details: 追加詳細
                num_inference_steps: ステップ数
                guidance_scale: CFGスケール
                seed: ランダムシード
                size: 画像サイズ
    
            Returns:
                生成されたアートワーク、使用したプロンプト
            """
            # プロンプト構築
            components = [subject]
    
            # スタイル
            if art_style in self.ART_STYLES:
                components.append(self.ART_STYLES[art_style])
            else:
                components.append(art_style)
    
            # 構図
            if composition in self.COMPOSITIONS:
                components.append(self.COMPOSITIONS[composition])
    
            # 色彩パレット
            if color_palette in self.COLOR_PALETTES:
                components.append(self.COLOR_PALETTES[color_palette])
    
            # 雰囲気
            if mood:
                components.append(f"{mood} mood")
    
            # アーティスト参照
            if artist_reference:
                components.append(f"in the style of {artist_reference}")
    
            # 品質キーワード
            components.extend([
                "masterpiece",
                "highly detailed",
                "professional",
                "award winning"
            ])
    
            # 追加詳細
            if additional_details:
                if isinstance(additional_details, list):
                    components.extend(additional_details)
                else:
                    components.append(additional_details)
    
            prompt = ", ".join(components)
    
            # Negative prompt
            negative_prompt = "low quality, blurry, distorted, ugly, bad art, amateur"
    
            # 生成
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
    
            with torch.autocast(self.device):
                output = self.sd_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    width=size[0],
                    height=size[1]
                )
    
            return output.images[0], prompt
    
        def create_series(
            self,
            base_subject,
            num_variations=4,
            vary_parameter='color_palette',
            **base_kwargs
        ):
            """
            テーマに沿った作品シリーズを生成
    
            Args:
                base_subject: 基本主題
                num_variations: バリエーション数
                vary_parameter: 変化させるパラメータ
                **base_kwargs: 固定パラメータ
    
            Returns:
                作品リスト、プロンプトリスト
            """
            artworks = []
            prompts = []
    
            # 変化させる値のリスト
            if vary_parameter == 'color_palette':
                variations = list(self.COLOR_PALETTES.keys())
            elif vary_parameter == 'art_style':
                variations = list(self.ART_STYLES.keys())
            elif vary_parameter == 'composition':
                variations = list(self.COMPOSITIONS.keys())
            else:
                variations = [None] * num_variations
    
            for i, variation in enumerate(variations[:num_variations]):
                kwargs = base_kwargs.copy()
                if variation:
                    kwargs[vary_parameter] = variation
    
                seed = base_kwargs.get('seed')
                if seed is not None:
                    kwargs['seed'] = seed + i
    
                artwork, prompt = self.create_artwork(base_subject, **kwargs)
                artworks.append(artwork)
                prompts.append(prompt)
    
            return artworks, prompts
    
    # 使用例
    if __name__ == "__main__":
        art_system = ArtworkCreationSystem()
    
        # 単一作品生成
        artwork, prompt = art_system.create_artwork(
            subject="a serene zen garden with cherry blossoms",
            art_style="watercolor",
            composition="rule_of_thirds",
            color_palette="pastel",
            mood="peaceful",
            seed=42
        )
    
        print(f"Artwork prompt: {prompt}")
        artwork.show()
    
        # 作品シリーズ（色彩パレット変化）
        artworks, prompts = art_system.create_series(
            base_subject="a mystical forest",
            num_variations=4,
            vary_parameter='color_palette',
            art_style='digital_art',
            composition='diagonal',
            seed=100
        )
    
        # グリッド表示
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
    
        for idx, (artwork, prompt) in enumerate(zip(artworks, prompts)):
            axes[idx].imshow(artwork)
            axes[idx].axis('off')
            # 色彩パレット名を抽出
            palette = prompt.split('color palette')[0].split(',')[-1].strip()
            axes[idx].set_title(palette.capitalize())
    
        plt.tight_layout()
        plt.show()
    

* * *

## 5.6 倫理的考慮事項

### 生成AIの倫理的課題

生成モデルの応用には重要な倫理的課題が伴います。責任ある開発と利用のために以下の点を考慮する必要があります。

課題 | 説明 | 対策例  
---|---|---  
**ディープフェイク** | 本物と見分けがつかない偽画像・動画 | 電子透かし、出所証明、検出技術  
**著作権侵害** | 訓練データの権利、生成物の帰属 | ライセンス確認、適切なクレジット表記  
**バイアスと公平性** | 訓練データのバイアスが生成物に反映 | 多様なデータセット、バイアス検出  
**悪用リスク** | 有害コンテンツ、詐欺、ハラスメント | セーフティフィルター、利用規約  
**プライバシー** | 訓練データの個人情報漏洩 | データ匿名化、差分プライバシー  
  
#### 責任ある利用のためのベストプラクティス
    
    
    ```mermaid
    graph TB
        A[生成AIの開発・利用] --> B[透明性]
        A --> C[説明責任]
        A --> D[公平性]
        A --> E[プライバシー保護]
        A --> F[安全性]
    
        B --> B1[モデルの限界を明示]
        B --> B2[生成物である旨を表示]
    
        C --> C1[利用規約の整備]
        C --> C2[監査可能性の確保]
    
        D --> D1[バイアステスト実施]
        D --> D2[多様な表現の保証]
    
        E --> E1[データ保護措置]
        E --> E2[同意取得プロセス]
    
        F --> F1[有害コンテンツフィルター]
        F --> F2[誤用防止機能]
    
        style A fill:#e3f2fd
        style B fill:#c8e6c9
        style C fill:#fff9c4
        style D fill:#ffccbc
        style E fill:#f8bbd0
        style F fill:#b2dfdb
    ```

> **実装時の推奨事項** :
> 
>   1. **Safety Checker実装** : 有害・不適切コンテンツの検出と除外
>   2. **透かし埋め込み** : AI生成物であることを示す不可視マーカー
>   3. **利用ログ記録** : 悪用時の追跡可能性確保
>   4. **ユーザー教育** : 適切な利用方法とリスクの周知
>   5. **継続的監視** : モデルの振る舞いとバイアスのモニタリング
> 

### 法的・規制的側面

  * **著作権法** : AI生成物の著作権帰属は複雑で、各国で対応が異なる
  * **肖像権・パブリシティ権** : 実在人物に似た画像の生成と利用には注意が必要
  * **EU AI Act** : 高リスクAIシステムへの規制、透明性要件
  * **プラットフォームポリシー** : 各プラットフォームの利用規約遵守

* * *

## 演習問題

**演習1: プロンプトエンジニアリング**

**課題** : 以下のシナリオに対して効果的なプロンプトを設計してください：

  1. 中世ヨーロッパの城の風景画（油絵スタイル）
  2. 未来都市のネオン街（サイバーパンクスタイル）
  3. 静かな日本庭園（水彩画スタイル）

**要件** :

  * Subject、Style、Quality、Details、Modifiersを含める
  * 適切なNegative promptを設計
  * CFGスケールとステップ数を推奨

**ヒント** : PromptEngineerクラスを参考に、各要素を明確に分離して構築しましょう。

**演習2: Style Transfer実装**

**課題** : StyleTransferGeneratorクラスを拡張し、以下の機能を追加してください：

  1. **複数スタイル比較** : 1つの画像に複数のスタイルを適用し比較表示
  2. **強度グラデーション** : strength値を段階的に変化させた結果を表示
  3. **スタイル合成** : 2つのスタイルプロンプトを組み合わせ

**期待される出力** : グリッド画像で各バリエーションを可視化

**演習3: Conditional GAN拡張**

**課題** : ConditionalGANクラスを拡張し、複数属性の条件付け（Multi-Label Conditional GAN）を実装してください。

**仕様** :

  * 2つ以上の属性を同時に条件付け（例: クラス + 色）
  * 各属性に対する埋め込み層を実装
  * 属性を組み合わせて生成できるようにする

**評価基準** : 指定した複数属性を持つ画像が生成されるか

**演習4: アバターシステム改善**

**課題** : AvatarGenerationSystemに以下の機能を追加してください：

  1. **アバター編集機能** : 生成後に属性を部分的に変更
  2. **一貫性スコア** : 複数生成されたアバターの一貫性を評価
  3. **バッチ処理** : 大量のアバターを効率的に生成
  4. **カスタムスタイル学習** : ユーザー提供画像からスタイルを学習

**実装のポイント** : Image-to-Image変換を活用し、既存アバターを基に修正する方法を検討しましょう。

**演習5: 倫理的セーフガード実装**

**課題** : 生成システムに倫理的セーフガードを実装してください：

**実装項目** :

  1. **コンテンツフィルター** : 不適切なプロンプトを検出・拒否
  2. **透かし埋め込み** : AI生成を示すマーカーを画像に追加
  3. **生成ログ** : プロンプトと生成物を記録
  4. **バイアス検出** : 特定属性の過剰/過少表現を検出

**テストケース** :

  * 不適切なプロンプトが適切に拒否されるか
  * 透かしが画像に含まれているか（目視 or プログラム的検出）
  * 多様性指標が一定基準を満たすか

* * *

## まとめ

この章では、生成モデルの実践的応用について学習しました：

  * **Text-to-Image生成** : Stable Diffusionを使った高品質画像生成とプロンプトエンジニアリング技術
  * **Image-to-Image変換** : Style transferとSuper-resolutionによる画像変換
  * **条件付き生成** : Conditional GANによる制御可能な生成
  * **Audio生成** : WaveGANによる音声波形生成の基礎
  * **実践プロジェクト** : アバター生成システムとアートワーク作成の包括的実装
  * **倫理的考慮** : 責任あるAI開発のための課題と対策

生成AIは強力な技術ですが、その利用には責任が伴います。技術的スキルと倫理的配慮の両方を持って、社会に貢献する応用を開発していきましょう。
