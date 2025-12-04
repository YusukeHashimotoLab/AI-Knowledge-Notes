---
title: 第1章：Python基礎
chapter_title: 第1章：Python基礎
---

**Pythonプログラミングの基本をマスターしよう**

## はじめに

機械学習を学ぶ第一歩は、**Pythonプログラミング** の基礎を固めることです。Pythonは機械学習で最も広く使われているプログラミング言語で、シンプルで読みやすい構文が特徴です。

この章では、以下の内容を学びます：

  * 変数とデータ型の基礎
  * 関数の定義と使い方
  * 制御構文（条件分岐・繰り返し）
  * リスト内包表記
  * ファイル入出力
  * エラー処理

## 1\. 変数とデータ型

### 1.1 基本的なデータ型

Pythonには主に以下のデータ型があります：

データ型 | 説明 | 例  
---|---|---  
`int` | 整数 | 42, -10, 0  
`float` | 浮動小数点数 | 3.14, -0.5, 2.0  
`str` | 文字列 | "Hello", 'Python'  
`bool` | 真偽値 | True, False  
`list` | リスト（配列） | [1, 2, 3], ["a", "b"]  
`dict` | 辞書（キーと値のペア） | {"name": "Alice", "age": 25}  
  
#### 例1：変数とデータ型
    
    
    # 整数
    age = 25
    print(f"年齢: {age}, 型: {type(age)}")  # 出力: 年齢: 25, 型: <class 'int'>
    
    # 浮動小数点数
    height = 170.5
    print(f"身長: {height}cm, 型: {type(height)}")  # 出力: 身長: 170.5cm, 型: <class 'float'>
    
    # 文字列
    name = "太郎"
    print(f"名前: {name}, 型: {type(name)}")  # 出力: 名前: 太郎, 型: <class 'str'>
    
    # 真偽値
    is_student = True
    print(f"学生: {is_student}, 型: {type(is_student)}")  # 出力: 学生: True, 型: <class 'bool'>
    
    # リスト
    scores = [85, 90, 78, 92]
    print(f"成績: {scores}, 型: {type(scores)}")  # 出力: 成績: [85, 90, 78, 92], 型: <class 'list'>
    
    # 辞書
    person = {"name": "太郎", "age": 25, "city": "東京"}
    print(f"人物: {person}, 型: {type(person)}")  # 出力: 人物: {'name': '太郎', 'age': 25, 'city': '東京'}, 型: <class 'dict'>
    

### 1.2 リストと辞書の操作

#### 例2：リストの基本操作
    
    
    # リストの作成
    numbers = [1, 2, 3, 4, 5]
    
    # 要素のアクセス（0から始まるインデックス）
    print(numbers[0])    # 出力: 1
    print(numbers[-1])   # 出力: 5（最後の要素）
    
    # スライシング
    print(numbers[1:4])  # 出力: [2, 3, 4]（インデックス1から3まで）
    print(numbers[:3])   # 出力: [1, 2, 3]（最初から3個）
    print(numbers[2:])   # 出力: [3, 4, 5]（インデックス2から最後まで）
    
    # 要素の追加
    numbers.append(6)
    print(numbers)       # 出力: [1, 2, 3, 4, 5, 6]
    
    # 要素の削除
    numbers.remove(3)
    print(numbers)       # 出力: [1, 2, 4, 5, 6]
    
    # リストの長さ
    print(len(numbers))  # 出力: 5
    

#### 例3：辞書の基本操作
    
    
    # 辞書の作成
    student = {
        "name": "花子",
        "age": 20,
        "major": "情報科学"
    }
    
    # 値のアクセス
    print(student["name"])        # 出力: 花子
    print(student.get("age"))     # 出力: 20
    
    # 値の追加・更新
    student["gpa"] = 3.8
    student["age"] = 21
    print(student)  # 出力: {'name': '花子', 'age': 21, 'major': '情報科学', 'gpa': 3.8}
    
    # キーの確認
    print("name" in student)      # 出力: True
    print("email" in student)     # 出力: False
    
    # すべてのキーと値
    print(student.keys())         # 出力: dict_keys(['name', 'age', 'major', 'gpa'])
    print(student.values())       # 出力: dict_values(['花子', 21, '情報科学', 3.8])
    

## 2\. 関数の定義と使い方

関数は、再利用可能なコードのまとまりです。`def`キーワードを使って定義します。

#### 例4：関数の基本
    
    
    # 引数なしの関数
    def greet():
        print("こんにちは！")
    
    greet()  # 出力: こんにちは！
    
    # 引数ありの関数
    def greet_person(name):
        print(f"こんにちは、{name}さん！")
    
    greet_person("太郎")  # 出力: こんにちは、太郎さん！
    
    # 戻り値のある関数
    def add(a, b):
        return a + b
    
    result = add(5, 3)
    print(result)  # 出力: 8
    
    # デフォルト引数
    def power(base, exponent=2):
        return base ** exponent
    
    print(power(3))      # 出力: 9（3の2乗）
    print(power(3, 3))   # 出力: 27（3の3乗）
    
    # 複数の戻り値
    def calculate(a, b):
        return a + b, a - b, a * b, a / b
    
    sum_val, diff, prod, quot = calculate(10, 2)
    print(f"和: {sum_val}, 差: {diff}, 積: {prod}, 商: {quot}")
    # 出力: 和: 12, 差: 8, 積: 20, 商: 5.0
    
    
    
    ```mermaid
    graph LR
        A[関数呼び出し] --> B[引数を渡す]
        B --> C[関数内で処理]
        C --> D[戻り値を返す]
        D --> E[呼び出し元で使用]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style E fill:#e8f5e9
    ```

## 3\. 制御構文

### 3.1 条件分岐（if文）

#### 例5：if文の基本
    
    
    # 基本的なif文
    score = 85
    
    if score >= 90:
        print("優秀です！")
    elif score >= 70:
        print("良好です。")
    elif score >= 60:
        print("合格です。")
    else:
        print("不合格です。")
    # 出力: 良好です。
    
    # 複数条件
    age = 25
    has_license = True
    
    if age >= 18 and has_license:
        print("運転できます。")
    else:
        print("運転できません。")
    # 出力: 運転できます。
    
    # 三項演算子
    temperature = 25
    weather = "暑い" if temperature > 30 else "涼しい"
    print(weather)  # 出力: 涼しい
    

### 3.2 繰り返し（for文・while文）

#### 例6：for文とwhile文
    
    
    # for文（リストの要素を反復）
    fruits = ["りんご", "バナナ", "オレンジ"]
    for fruit in fruits:
        print(f"好きな果物: {fruit}")
    # 出力:
    # 好きな果物: りんご
    # 好きな果物: バナナ
    # 好きな果物: オレンジ
    
    # range関数を使った繰り返し
    for i in range(5):
        print(i)
    # 出力: 0, 1, 2, 3, 4
    
    # while文
    count = 0
    while count < 5:
        print(f"カウント: {count}")
        count += 1
    # 出力: カウント: 0, カウント: 1, ..., カウント: 4
    
    # enumerate（インデックス付き反復）
    for index, fruit in enumerate(fruits):
        print(f"{index}: {fruit}")
    # 出力:
    # 0: りんご
    # 1: バナナ
    # 2: オレンジ
    
    # 辞書の反復
    student = {"name": "太郎", "age": 20, "major": "CS"}
    for key, value in student.items():
        print(f"{key}: {value}")
    # 出力:
    # name: 太郎
    # age: 20
    # major: CS
    
    
    
    ```mermaid
    graph TD
        A[開始] --> B{条件チェック}
        B -->|True| C[処理実行]
        C --> D[更新]
        D --> B
        B -->|False| E[終了]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style E fill:#e8f5e9
    ```

## 4\. リスト内包表記

リスト内包表記は、リストを簡潔に作成する強力な機能です。

#### 例7：リスト内包表記
    
    
    # 通常の方法
    squares = []
    for i in range(10):
        squares.append(i ** 2)
    print(squares)  # 出力: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    
    # リスト内包表記（1行で書ける）
    squares = [i ** 2 for i in range(10)]
    print(squares)  # 出力: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    
    # 条件付きリスト内包表記
    even_squares = [i ** 2 for i in range(10) if i % 2 == 0]
    print(even_squares)  # 出力: [0, 4, 16, 36, 64]
    
    # 文字列の変換
    words = ["hello", "world", "python"]
    upper_words = [word.upper() for word in words]
    print(upper_words)  # 出力: ['HELLO', 'WORLD', 'PYTHON']
    
    # ネストしたリスト内包表記
    matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
    print(matrix)
    # 出力: [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
    
    # 辞書内包表記
    numbers = [1, 2, 3, 4, 5]
    squares_dict = {n: n ** 2 for n in numbers}
    print(squares_dict)  # 出力: {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
    

## 5\. ファイル入出力

#### 例8：ファイルの読み書き
    
    
    # ファイルへの書き込み
    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write("こんにちは、Python！\n")
        f.write("機械学習を学びましょう。\n")
    
    # ファイルの読み込み（全体）
    with open("sample.txt", "r", encoding="utf-8") as f:
        content = f.read()
        print(content)
    
    # ファイルの読み込み（行ごと）
    with open("sample.txt", "r", encoding="utf-8") as f:
        for line in f:
            print(line.strip())  # strip()で改行を削除
    
    # ファイルへの追記
    with open("sample.txt", "a", encoding="utf-8") as f:
        f.write("新しい行を追加\n")
    
    # CSVファイルの読み書き
    import csv
    
    # CSVへの書き込み
    data = [
        ["名前", "年齢", "都市"],
        ["太郎", "25", "東京"],
        ["花子", "30", "大阪"]
    ]
    
    with open("data.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)
    
    # CSVの読み込み
    with open("data.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
    

## 6\. エラー処理（try-except）

#### 例9：例外処理
    
    
    # 基本的な例外処理
    try:
        result = 10 / 0  # ZeroDivisionError
    except ZeroDivisionError:
        print("ゼロで割ることはできません！")
    # 出力: ゼロで割ることはできません！
    
    # 複数の例外処理
    def safe_divide(a, b):
        try:
            return a / b
        except ZeroDivisionError:
            print("エラー: ゼロで割れません")
            return None
        except TypeError:
            print("エラー: 数値を入力してください")
            return None
    
    print(safe_divide(10, 2))     # 出力: 5.0
    print(safe_divide(10, 0))     # 出力: エラー: ゼロで割れません, None
    print(safe_divide(10, "a"))   # 出力: エラー: 数値を入力してください, None
    
    # else と finally
    try:
        number = int(input("数値を入力: "))
        result = 100 / number
    except ValueError:
        print("エラー: 数値を入力してください")
    except ZeroDivisionError:
        print("エラー: ゼロ以外を入力してください")
    else:
        print(f"結果: {result}")  # エラーがない場合のみ実行
    finally:
        print("処理完了")  # 常に実行される
    
    # カスタム例外
    class NegativeNumberError(Exception):
        pass
    
    def sqrt(x):
        if x < 0:
            raise NegativeNumberError("負の数の平方根は計算できません")
        return x ** 0.5
    
    try:
        print(sqrt(16))   # 出力: 4.0
        print(sqrt(-4))   # NegativeNumberError発生
    except NegativeNumberError as e:
        print(f"エラー: {e}")
    

## 7\. 実践例：簡単なデータ処理

#### 例10：学生の成績処理プログラム
    
    
    # 学生データ
    students = [
        {"name": "太郎", "scores": [85, 90, 78]},
        {"name": "花子", "scores": [92, 88, 95]},
        {"name": "次郎", "scores": [70, 75, 80]},
        {"name": "桃子", "scores": [88, 91, 87]}
    ]
    
    # 平均点を計算する関数
    def calculate_average(scores):
        return sum(scores) / len(scores)
    
    # 成績評価を返す関数
    def get_grade(average):
        if average >= 90:
            return "A"
        elif average >= 80:
            return "B"
        elif average >= 70:
            return "C"
        elif average >= 60:
            return "D"
        else:
            return "F"
    
    # 各学生の成績を処理
    results = []
    for student in students:
        avg = calculate_average(student["scores"])
        grade = get_grade(avg)
        results.append({
            "name": student["name"],
            "average": avg,
            "grade": grade
        })
    
    # 結果を表示
    print("=" * 50)
    print("成績レポート")
    print("=" * 50)
    for result in results:
        print(f"{result['name']}: 平均 {result['average']:.2f} (評価: {result['grade']})")
    
    # 最高得点の学生を見つける
    best_student = max(results, key=lambda x: x["average"])
    print("=" * 50)
    print(f"最優秀学生: {best_student['name']} ({best_student['average']:.2f})")
    
    # リスト内包表記で優秀な学生を抽出
    excellent_students = [r["name"] for r in results if r["average"] >= 85]
    print(f"優秀な学生 (平均85以上): {', '.join(excellent_students)}")
    
    
    
    ```mermaid
    graph TD
        A[データ入力] --> B[平均点計算]
        B --> C[成績評価]
        C --> D[結果の格納]
        D --> E[レポート出力]
        E --> F[統計情報表示]
    
        style A fill:#e3f2fd
        style C fill:#fff3e0
        style F fill:#e8f5e9
    ```

## まとめ

この章では、Pythonプログラミングの基礎を学びました：

  * ✅ **データ型** : int, float, str, bool, list, dict
  * ✅ **関数** : def, return, 引数, デフォルト引数
  * ✅ **制御構文** : if-elif-else, for, while
  * ✅ **リスト内包表記** : 簡潔なリスト作成
  * ✅ **ファイルI/O** : open, read, write, CSV処理
  * ✅ **例外処理** : try-except-finally

**次のステップ** : これらの基礎を使って、第2章でNumPyによる数値計算を学びます。

## 演習問題

演習1：FizzBuzz問題

**問題** : 1から100までの数字を出力するプログラムを書いてください。ただし、3の倍数のときは数字の代わりに「Fizz」、5の倍数のときは「Buzz」、3と5の両方の倍数のときは「FizzBuzz」と出力してください。
    
    
    # 解答例
    for i in range(1, 101):
        if i % 15 == 0:  # 3と5の倍数（15の倍数）
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)
    

演習2：素数判定関数

**問題** : 与えられた数が素数かどうかを判定する関数`is_prime(n)`を作成してください。
    
    
    # 解答例
    def is_prime(n):
        """素数判定関数"""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    # テスト
    test_numbers = [2, 3, 4, 17, 20, 29, 100]
    for num in test_numbers:
        result = "素数" if is_prime(num) else "素数でない"
        print(f"{num}: {result}")
    

演習3：リスト内包表記

**問題** : 1から100までの数字のうち、3の倍数または5の倍数（または両方）のリストをリスト内包表記で作成してください。
    
    
    # 解答例
    multiples = [i for i in range(1, 101) if i % 3 == 0 or i % 5 == 0]
    print(multiples)
    print(f"総数: {len(multiples)}個")
    print(f"合計: {sum(multiples)}")
    
    # 出力: [3, 5, 6, 9, 10, 12, 15, ..., 100]
    # 総数: 47個
    # 合計: 2418
    

演習4：辞書の集計

**問題** : 文字列から各文字の出現回数を辞書として返す関数`count_chars(text)`を作成してください。
    
    
    # 解答例
    def count_chars(text):
        """文字の出現回数を数える"""
        char_count = {}
        for char in text:
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
        return char_count
    
    # または辞書のget()を使った方法
    def count_chars_v2(text):
        char_count = {}
        for char in text:
            char_count[char] = char_count.get(char, 0) + 1
        return char_count
    
    # テスト
    text = "hello world"
    result = count_chars(text)
    print(result)
    # 出力: {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}
    

演習5：ファイル処理

**問題** : テキストファイルを読み込み、各行の単語数を数えて、結果を新しいファイルに出力するプログラムを作成してください。
    
    
    # 解答例
    def count_words_in_file(input_file, output_file):
        """ファイル内の各行の単語数を数える"""
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
    
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("行番号,単語数,内容\n")
                for i, line in enumerate(lines, 1):
                    word_count = len(line.split())
                    f.write(f"{i},{word_count},{line.strip()}\n")
    
            print(f"結果を {output_file} に保存しました。")
    
        except FileNotFoundError:
            print(f"エラー: {input_file} が見つかりません")
        except Exception as e:
            print(f"エラーが発生しました: {e}")
    
    # テスト用ファイルを作成
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write("Python is great\n")
        f.write("Machine learning is fun\n")
        f.write("Let's learn together\n")
    
    # 関数を実行
    count_words_in_file("test.txt", "result.txt")
    

[← シリーズトップ](<./index.html>) [第2章: NumPy基礎 →](<./chapter2-numpy-basics.html>)
