# LAD (Log Anomary Detection)

uv使います。


## 環境構築
```bash
# パッケージのインストール
uv sync
# 仮想環境の起動
source .venv/bin/activate
```

## 実行
基本的にbashを使用する。
実行環境やデータの保存場所が大幅に変更されている。

現状では網屋データセット(?)をLogBERTで学習させる以下のコードが動作:

### 訓練
```bash 
bash src/parallel_bash/parallel13_recovered.sh
```

### テスト
```bash
bash src/parallel_bash/test2_single.sh
```

他のデータや前処理なども行う場合は、上記スクリプトに対応させる形で記述する。

## メモ
pythonのバージョンは3.11にする必要がある

```bash
uv python install 3.11
# .python-versionを3.11にする
```

データセットはシンボリックリンクを作成しよう。
```bash
# リカバリー版で確認
ln -s /home/local/amiya/dataset/ ./dataset
# 過去の重みを使うなら スクリプトは絶対パスから相対パスにしよう
ln -s /home/local/amiya/ ./amiya
```
