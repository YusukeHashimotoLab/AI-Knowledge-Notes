---
title: 第2章：DFT計算の自動化（VASP, Quantum ESPRESSO）
chapter_title: 第2章：DFT計算の自動化（VASP, Quantum ESPRESSO）
subtitle: 
reading_time: 20-25分
difficulty: 中級〜上級
code_examples: 6
exercises: 0
---

# 第2章：DFT計算の自動化（VASP, Quantum ESPRESSO）

ASEやpymatgenでの入力生成から実行管理まで、ミスを減らす自動化の基本を学びます。

**💡 補足:** 入力生成はテンプレート化し、差分だけ埋める設計に。失敗時の再実行手順も用意します。

## 学習目標

この章を読むことで、以下を習得できます：

  * ✅ ASEを使ってDFT計算を自動実行できる
  * ✅ VASPとQuantum ESPRESSOの入力ファイルを自動生成できる
  * ✅ pymatgenのInputSetで標準化された設定を使える
  * ✅ エラーを検出して自動リスタートできる
  * ✅ 構造最適化の収束判定を自動化できる

* * *

## 2.1 ASE（Atomic Simulation Environment）基礎

### ASEとは

**ASE（Atomic Simulation Environment）** は、原子スケールシミュレーションのためのPythonライブラリです。

**特徴** : \- ✅ 多様な計算コード対応（VASP、QE、LAMMPS等） \- ✅ 構造生成・操作の統一インターフェース \- ✅ 計算結果の解析ツール \- ✅ MITライセンス（オープンソース）

### インストール
    
    
    # conda環境を推奨
    conda create -n ht_computing python=3.10
    conda activate ht_computing
    
    # ASEインストール
    pip install ase
    
    # 追加パッケージ
    pip install numpy scipy matplotlib
    

### 基本的な構造生成
    
    
    from ase import Atoms
    from ase.build import bulk, molecule
    import numpy as np
    
    # 例1: バルク結晶の生成
    si = bulk('Si', 'diamond', a=5.43)
    print(f"Si結晶: {len(si)}原子")
    print(f"格子定数: {si.cell.cellpar()}")
    
    # 例2: 分子の生成
    h2o = molecule('H2O')
    print(f"H2O分子: {len(h2o)}原子")
    
    # 例3: カスタム構造
    # LiCoO2（層状構造）
    a = 2.82
    c = 14.05
    positions = [
        [0, 0, 0],       # Li
        [0, 0, 0.5*c],   # Co
        [1/3, 2/3, 0.25*c],  # O
        [2/3, 1/3, 0.75*c],  # O
    ]
    atoms = Atoms('LiCoO2',
                  positions=positions,
                  cell=[a, a, c, 90, 90, 120],
                  pbc=True)
    
    print(f"LiCoO2: {atoms.get_chemical_formula()}")
    

**出力** :
    
    
    Si結晶: 2原子
    格子定数: [5.43 5.43 5.43 90.  90.  90. ]
    H2O分子: 3原子
    LiCoO2: LiCoO2
    

### 構造の可視化
    
    
    from ase.visualize import view
    
    # 3D可視化（GUIが起動）
    view(si)
    
    # ファイル保存
    from ase.io import write
    
    # CIF形式で保存
    write('si_structure.cif', si)
    
    # POSCAR形式（VASP用）
    write('POSCAR', si, format='vasp')
    
    # XYZ形式
    write('structure.xyz', si)
    

* * *

## 2.2 VASP自動化

### VASPインターフェース設定
    
    
    from ase.calculators.vasp import Vasp
    
    # VASP計算器の設定
    calc = Vasp(
        # 基本設定
        xc='PBE',           # 交換相関汎関数
        encut=520,          # エネルギーカットオフ（eV）
    
        # k-point設定
        kpts=(8, 8, 8),     # Monkhorst-Pack格子
        gamma=True,         # Γ点中心
    
        # 電子構造
        ismear=0,           # Gaussianスミアリング
        sigma=0.05,         # スミアリング幅（eV）
    
        # 収束判定
        ediff=1e-5,         # エネルギー収束（eV）
    
        # 並列化
        ncore=4,            # 並列効率向上
    
        # 出力制御
        lwave=False,        # WAVECARを保存しない
        lcharg=False,       # CHGCARを保存しない
    )
    

### 構造最適化の自動化
    
    
    from ase.optimize import BFGS
    from ase.calculators.vasp import Vasp
    from ase.io import read, write
    import os
    
    def relax_structure(atoms, output_dir='relaxation'):
        """
        構造最適化を自動実行
    
        Parameters:
        -----------
        atoms : ase.Atoms
            最適化する構造
        output_dir : str
            出力ディレクトリ
    
        Returns:
        --------
        relaxed_atoms : ase.Atoms
            最適化後の構造
        """
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
    
        # VASP計算器設定
        calc = Vasp(
            xc='PBE',
            encut=520,
            kpts=(8, 8, 8),
            ediff=1e-5,
            ibrion=2,       # 構造最適化アルゴリズム
            nsw=100,        # 最大イオンステップ数
            isif=3,         # セル形状も最適化
            ediffg=-0.01,   # 力の収束判定（eV/Å）
        )
    
        atoms.calc = calc
    
        # 最適化実行
        print(f"構造最適化を開始: {atoms.get_chemical_formula()}")
    
        try:
            # BFGSアルゴリズムで最適化
            optimizer = BFGS(atoms, trajectory='optimization.traj')
            optimizer.run(fmax=0.01)  # 最大力が0.01 eV/Å以下
    
            print("構造最適化が完了しました")
            print(f"最終エネルギー: {atoms.get_potential_energy():.3f} eV")
    
            # 最適化後の構造を保存
            write('CONTCAR', atoms, format='vasp')
    
            return atoms
    
        except Exception as e:
            print(f"エラーが発生しました: {e}")
            return None
    
        finally:
            os.chdir('..')
    
    # 使用例
    si = bulk('Si', 'diamond', a=5.43)
    relaxed_si = relax_structure(si, output_dir='si_relaxation')
    

### INCARファイル自動生成
    
    
    def generate_incar(calculation_type='relax'):
        """
        計算タイプに応じたINCAR設定を生成
    
        Parameters:
        -----------
        calculation_type : str
            'relax', 'static', 'band', 'dos'のいずれか
    
        Returns:
        --------
        incar_dict : dict
            INCAR設定の辞書
        """
        # 共通設定
        base_settings = {
            'SYSTEM': 'Automated calculation',
            'PREC': 'Accurate',
            'ENCUT': 520,
            'EDIFF': 1e-5,
            'ISMEAR': 0,
            'SIGMA': 0.05,
            'LREAL': False,
            'LWAVE': False,
            'LCHARG': False,
        }
    
        # 計算タイプ別設定
        if calculation_type == 'relax':
            specific = {
                'IBRION': 2,      # CG法
                'NSW': 100,       # 最大イオンステップ
                'ISIF': 3,        # セル形状最適化
                'EDIFFG': -0.01,  # 力の収束
            }
    
        elif calculation_type == 'static':
            specific = {
                'IBRION': -1,     # イオン緩和なし
                'NSW': 0,
                'LCHARG': True,   # 電荷密度を保存
            }
    
        elif calculation_type == 'band':
            specific = {
                'IBRION': -1,
                'NSW': 0,
                'ICHARG': 11,     # 電荷密度を読み込み
                'LORBIT': 11,     # DOSとバンド
            }
    
        elif calculation_type == 'dos':
            specific = {
                'IBRION': -1,
                'NSW': 0,
                'ICHARG': 11,
                'LORBIT': 11,
                'NEDOS': 2001,    # DOS解像度
            }
    
        else:
            raise ValueError(f"Unknown calculation type: {calculation_type}")
    
        # 設定を結合
        incar_dict = {**base_settings, **specific}
    
        return incar_dict
    
    # 使用例
    relax_incar = generate_incar('relax')
    print("構造最適化用INCAR:")
    for key, value in relax_incar.items():
        print(f"{key} = {value}")
    

### K-point自動設定
    
    
    from ase.dft.kpoints import monkhorst_pack
    import numpy as np
    
    def auto_kpoints(atoms, kpt_density=1000):
        """
        セルサイズに応じてk-point密度を自動設定
    
        Parameters:
        -----------
        atoms : ase.Atoms
            構造
        kpt_density : float
            k-point密度（1/Å³あたりのk-point数）
    
        Returns:
        --------
        kpts : tuple
            k-point格子（nx, ny, nz）
        """
        # セルの逆格子ベクトル長さ
        cell = atoms.get_cell()
        reciprocal_cell = cell.reciprocal()
        lengths = np.linalg.norm(reciprocal_cell, axis=1)
    
        # k-point数を計算
        kpts = []
        for length in lengths:
            # k-point密度から必要な分割数を計算
            n = max(1, int(np.ceil(kpt_density / length)))
            kpts.append(n)
    
        return tuple(kpts)
    
    # 使用例
    si = bulk('Si', 'diamond', a=5.43)
    kpts = auto_kpoints(si, kpt_density=1000)
    print(f"Si用k-point格子: {kpts}")  # 例: (12, 12, 12)
    
    # 大きなセル
    supercell = si * (2, 2, 2)
    kpts_super = auto_kpoints(supercell, kpt_density=1000)
    print(f"スーパーセル用k-point: {kpts_super}")  # 例: (6, 6, 6)
    

* * *

## 2.3 Quantum ESPRESSO自動化

### QE計算器設定
    
    
    from ase.calculators.espresso import Espresso
    
    # Quantum ESPRESSO計算器
    calc_qe = Espresso(
        # 実行ファイルパス
        command='pw.x -in PREFIX.pwi > PREFIX.pwo',
    
        # 擬ポテンシャル
        pseudopotentials={
            'Si': 'Si.pbe-n-kjpaw_psl.1.0.0.UPF',
            'O': 'O.pbe-n-kjpaw_psl.1.0.0.UPF',
        },
        pseudo_dir='/path/to/pseudopotentials',
    
        # 入力パラメータ
        input_data={
            'control': {
                'calculation': 'relax',
                'restart_mode': 'from_scratch',
                'prefix': 'pwscf',
                'outdir': './tmp',
                'pseudo_dir': '/path/to/pseudopotentials',
            },
            'system': {
                'ecutwfc': 60,      # 波動関数カットオフ（Ry）
                'ecutrho': 480,     # 電荷密度カットオフ
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.01,
            },
            'electrons': {
                'conv_thr': 1e-8,   # 収束判定
                'mixing_beta': 0.7,
            },
            'ions': {
                'ion_dynamics': 'bfgs',
            },
        },
    
        # k-point
        kpts=(8, 8, 8),
        koffset=(0, 0, 0),
    )
    

### QE入力ファイルテンプレート
    
    
    def generate_qe_input(atoms, calculation='relax'):
        """
        Quantum ESPRESSO入力ファイルを生成
    
        Parameters:
        -----------
        atoms : ase.Atoms
            計算する構造
        calculation : str
            'relax', 'scf', 'nscf', 'bands'
    
        Returns:
        --------
        input_str : str
            入力ファイル内容
        """
        from ase.io.espresso import write_espresso_in
    
        # 計算タイプ別設定
        if calculation == 'relax':
            input_data = {
                'control': {
                    'calculation': 'relax',
                    'restart_mode': 'from_scratch',
                },
                'system': {
                    'ecutwfc': 60,
                    'ecutrho': 480,
                },
                'electrons': {
                    'conv_thr': 1e-8,
                },
                'ions': {
                    'ion_dynamics': 'bfgs',
                },
            }
    
        elif calculation == 'scf':
            input_data = {
                'control': {
                    'calculation': 'scf',
                },
                'system': {
                    'ecutwfc': 60,
                    'ecutrho': 480,
                },
                'electrons': {
                    'conv_thr': 1e-8,
                },
            }
    
        elif calculation == 'bands':
            input_data = {
                'control': {
                    'calculation': 'bands',
                },
                'system': {
                    'ecutwfc': 60,
                    'ecutrho': 480,
                    'nbnd': 20,  # バンド数
                },
                'electrons': {
                    'conv_thr': 1e-8,
                },
            }
    
        # 入力ファイル書き込み
        write_espresso_in(
            'pw.in',
            atoms,
            input_data=input_data,
            pseudopotentials={el: f"{el}.pbe.UPF" for el in set(atoms.get_chemical_symbols())},
            kpts=(8, 8, 8),
        )
    
        with open('pw.in', 'r') as f:
            input_str = f.read()
    
        return input_str
    
    # 使用例
    si = bulk('Si', 'diamond', a=5.43)
    qe_input = generate_qe_input(si, calculation='relax')
    print(qe_input)
    

* * *

## 2.4 pymatgenによる高度な自動化

### InputSetの活用

pymatgenは、Materials Projectで使用されている標準化された計算設定を提供します。
    
    
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
    
    # 構造を読み込み（CIFファイルから）
    structure = Structure.from_file("LiCoO2.cif")
    
    # Materials Project標準の構造最適化設定
    relax_set = MPRelaxSet(structure)
    
    # INCAR, KPOINTS, POTCARを自動生成
    relax_set.write_input("relax_calculation")
    
    # ディレクトリ内容:
    # relax_calculation/
    #   ├── INCAR
    #   ├── POSCAR
    #   ├── KPOINTS
    #   └── POTCAR
    
    # 静的計算設定（構造最適化後）
    static_set = MPStaticSet(structure)
    static_set.write_input("static_calculation")
    

### カスタムInputSet
    
    
    from pymatgen.io.vasp.sets import DictSet
    
    # カスタム設定
    custom_incar = {
        "ENCUT": 600,      # デフォルトより高精度
        "EDIFF": 1e-6,
        "ISMEAR": -5,      # Tetrahedron法
        "LORBIT": 11,
        "LWAVE": False,
        "LCHARG": True,
    }
    
    # カスタムInputSet作成
    custom_set = DictSet(
        structure,
        config_dict={
            "INCAR": custom_incar,
            "KPOINTS": {"reciprocal_density": 200},  # k-point密度
        }
    )
    
    custom_set.write_input("custom_calculation")
    

### エラー検出とリスタート
    
    
    from pymatgen.io.vasp.outputs import Vasprun, Outcar
    from pymatgen.io.vasp.sets import MPRelaxSet
    import os
    
    def check_convergence(directory):
        """
        VASP計算の収束をチェック
    
        Returns:
        --------
        status : str
            'converged', 'not_converged', 'error'
        """
        try:
            # vasprun.xmlを読み込み
            vasprun = Vasprun(os.path.join(directory, "vasprun.xml"))
    
            if vasprun.converged:
                return 'converged'
            else:
                return 'not_converged'
    
        except Exception as e:
            print(f"エラー: {e}")
            return 'error'
    
    def auto_restart(directory, max_attempts=3):
        """
        収束しない計算を自動リスタート
    
        Parameters:
        -----------
        directory : str
            計算ディレクトリ
        max_attempts : int
            最大リトライ回数
        """
        for attempt in range(max_attempts):
            status = check_convergence(directory)
    
            if status == 'converged':
                print("計算が収束しました")
                return True
    
            elif status == 'not_converged':
                print(f"未収束。リトライ {attempt+1}/{max_attempts}")
    
                # 設定を緩和
                # 例: エネルギーカットオフを下げる、スミアリングを増やす
                modify_incar(directory, {'ENCUT': 450, 'SIGMA': 0.1})
    
                # リスタート
                restart_calculation(directory)
    
            elif status == 'error':
                print("致命的エラー。スキップします")
                return False
    
        print("最大リトライ回数に達しました")
        return False
    
    def modify_incar(directory, new_params):
        """INCARファイルを修正"""
        from pymatgen.io.vasp.inputs import Incar
    
        incar_file = os.path.join(directory, "INCAR")
        incar = Incar.from_file(incar_file)
    
        # パラメータを更新
        for key, value in new_params.items():
            incar[key] = value
    
        # 保存
        incar.write_file(incar_file)
        print(f"INCARを更新: {new_params}")
    
    def restart_calculation(directory):
        """計算を再実行"""
        import subprocess
    
        # CONTCARをPOSCARにコピー
        os.system(f"cp {directory}/CONTCAR {directory}/POSCAR")
    
        # VASP再実行
        os.chdir(directory)
        subprocess.run(["mpirun", "-np", "48", "vasp_std"])
        os.chdir("..")
    

* * *

## 2.5 バッチ処理

### 複数材料の自動計算
    
    
    import os
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import MPRelaxSet
    
    def batch_relax(structure_files, output_root='calculations'):
        """
        複数の構造を一括で構造最適化
    
        Parameters:
        -----------
        structure_files : list
            CIFファイルのリスト
        output_root : str
            出力ルートディレクトリ
        """
        os.makedirs(output_root, exist_ok=True)
    
        for cif_file in structure_files:
            # 構造を読み込み
            structure = Structure.from_file(cif_file)
            formula = structure.composition.reduced_formula
    
            print(f"処理中: {formula}")
    
            # 出力ディレクトリ
            calc_dir = os.path.join(output_root, formula)
    
            # InputSet作成
            relax_set = MPRelaxSet(structure)
            relax_set.write_input(calc_dir)
    
            # ジョブスクリプト作成
            create_job_script(calc_dir, formula)
    
            # ジョブ投入（SLURMの場合）
            os.chdir(calc_dir)
            os.system("sbatch job.sh")
            os.chdir("../..")
    
            print(f"  → ジョブ投入完了: {calc_dir}")
    
    def create_job_script(directory, jobname):
        """SLURMジョブスクリプト作成"""
        script = f"""#!/bin/bash
    #SBATCH --job-name={jobname}
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=48
    #SBATCH --time=24:00:00
    #SBATCH --partition=standard
    
    module load vasp/6.3.0
    
    mpirun -np 48 vasp_std
    """
    
        with open(os.path.join(directory, "job.sh"), 'w') as f:
            f.write(script)
    
    # 使用例
    cif_files = [
        "LiCoO2.cif",
        "LiNiO2.cif",
        "LiMnO2.cif",
        "LiFePO4.cif",
    ]
    
    batch_relax(cif_files, output_root='battery_materials')
    

* * *

## 2.6 演習問題

### 問題1（難易度: easy）

**問題** : ASEを使ってNaCl（岩塩構造）の結晶を生成し、POSCARファイルに保存してください。格子定数は5.64 Åです。

ヒント `ase.build.bulk`関数を使用。結晶構造は'rocksalt'を指定。  解答例
    
    
    from ase.build import bulk
    from ase.io import write
    
    # NaCl結晶生成
    nacl = bulk('NaCl', 'rocksalt', a=5.64)
    
    # POSCAR保存
    write('POSCAR_NaCl', nacl, format='vasp')
    
    print(f"NaCl: {nacl.get_chemical_formula()}")
    print(f"原子数: {len(nacl)}")
    print(f"セルパラメータ: {nacl.cell.cellpar()}")
    

**出力**: 
    
    
    NaCl: NaCl
    原子数: 2
    セルパラメータ: [5.64 5.64 5.64 90.  90.  90. ]
    

### 問題2（難易度: medium）

**問題** : pymatgenのMPRelaxSetを使って、Li₂O（Li2O.cif）の構造最適化用入力ファイルを生成してください。さらに、k-point密度を2000に変更してください。

ヒント `MPRelaxSet`のuser_incar_settingsとuser_kpoints_settingsを使用。  解答例
    
    
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import MPRelaxSet
    
    # 構造読み込み
    structure = Structure.from_file("Li2O.cif")
    
    # k-point密度をカスタマイズ
    relax_set = MPRelaxSet(
        structure,
        user_kpoints_settings={"reciprocal_density": 2000}
    )
    
    # 入力ファイル生成
    relax_set.write_input("li2o_relax")
    
    print("入力ファイルを生成しました: li2o_relax/")
    print(f"k-point設定:")
    with open("li2o_relax/KPOINTS", 'r') as f:
        print(f.read())
    

### 問題3（難易度: hard）

**問題** : 100個の酸化物（CIF形式）について、以下の条件で自動計算を設定してください：

  1. Materials Project標準設定で構造最適化
  2. k-point密度は1500
  3. 各材料ごとにディレクトリを作成
  4. SLURMジョブスクリプトを自動生成
  5. エラーハンドリング（読み込み失敗時はスキップ）

ヒント バッチ処理のコード例を参考に、エラーハンドリングを追加。  解答例
    
    
    import os
    from pymatgen.core import Structure
    from pymatgen.io.vasp.sets import MPRelaxSet
    from glob import glob
    
    def batch_process_oxides(cif_directory, output_root='oxide_calculations'):
        """
        100個の酸化物を一括処理
        """
        # CIFファイルリスト
        cif_files = glob(os.path.join(cif_directory, "*.cif"))
        print(f"CIFファイル数: {len(cif_files)}")
    
        os.makedirs(output_root, exist_ok=True)
    
        success_count = 0
        error_count = 0
    
        for cif_file in cif_files:
            try:
                # 構造読み込み
                structure = Structure.from_file(cif_file)
                formula = structure.composition.reduced_formula
    
                # 酸化物のみ処理
                if 'O' not in structure.composition.elements:
                    print(f"スキップ（酸化物ではない）: {formula}")
                    continue
    
                print(f"処理中 ({success_count+1}): {formula}")
    
                # 出力ディレクトリ
                calc_dir = os.path.join(output_root, formula)
    
                # InputSet作成（k-point密度1500）
                relax_set = MPRelaxSet(
                    structure,
                    user_kpoints_settings={"reciprocal_density": 1500}
                )
                relax_set.write_input(calc_dir)
    
                # ジョブスクリプト作成
                create_slurm_script(calc_dir, formula)
    
                success_count += 1
    
            except Exception as e:
                print(f"エラー: {cif_file} - {e}")
                error_count += 1
                continue
    
        print(f"\n完了:")
        print(f"  成功: {success_count}")
        print(f"  失敗: {error_count}")
    
    def create_slurm_script(directory, jobname):
        """SLURMジョブスクリプト"""
        script = f"""#!/bin/bash
    #SBATCH --job-name={jobname}
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=48
    #SBATCH --time=24:00:00
    #SBATCH --partition=standard
    #SBATCH --output=slurm-%j.out
    #SBATCH --error=slurm-%j.err
    
    # 環境設定
    module load intel/2021.2
    module load vasp/6.3.0
    
    # VASP実行
    mpirun -np 48 vasp_std
    
    # 収束チェック
    if grep -q "reached required accuracy" OUTCAR; then
        echo "収束成功"
        exit 0
    else
        echo "収束失敗"
        exit 1
    fi
    """
    
        with open(os.path.join(directory, "job.sh"), 'w') as f:
            f.write(script)
    
    # 実行
    batch_process_oxides("oxide_cifs", output_root="oxide_ht_calculations")
    

**出力例**: 
    
    
    CIFファイル数: 100
    処理中 (1): Li2O
    処理中 (2): Na2O
    スキップ（酸化物ではない）: LiCl
    処理中 (3): MgO
    ...
    完了:
      成功: 95
      失敗: 5
    

* * *

## 2.7 まとめ

この章では、ASEとpymatgenを使ったDFT計算の自動化を学びました。

**キーポイント** :

  1. **ASE** : 統一インターフェースで多様な計算コードを操作
  2. **VASP自動化** : INCAR、KPOINTS、POTCARの自動生成
  3. **QE自動化** : 入力ファイルテンプレート
  4. **pymatgen InputSet** : Materials Project標準設定
  5. **エラー処理** : 収束チェックと自動リスタート
  6. **バッチ処理** : 複数材料の一括計算

**次のステップ** :

第3章では、**ジョブスケジューリングと並列化** を学びます。SLURMスクリプトの作成、MPIによる大規模並列計算、1000材料規模の効率的な管理手法を習得します。

**[第3章: ジョブスケジューリングと並列化 →](<./chapter-3.html>)**

* * *

## 参考文献

  1. Larsen, A. H., et al. (2017). "The atomic simulation environment—a Python library for working with atoms." _Journal of Physics: Condensed Matter_ , 29(27), 273002.

  2. Ong, S. P., et al. (2013). "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis." _Computational Materials Science_ , 68, 314-319.

  3. Kresse, G., & Furthmüller, J. (1996). "Efficient iterative schemes for ab initio total-energy calculations using a plane-wave basis set." _Physical Review B_ , 54(16), 11169.

  4. Giannozzi, P., et al. (2009). "QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials." _Journal of Physics: Condensed Matter_ , 21(39), 395502.

* * *

**ライセンス** : CC BY 4.0 **作成日** : 2025-10-17 **作成者** : Dr. Yusuke Hashimoto, Tohoku University
