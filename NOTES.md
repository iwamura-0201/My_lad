# NaN問題解決の記録

最終更新: 2025-12-05

## 問題の概要

トレーニング中にNaN Lossが発生し、学習が進まなかった。

### 観察された症状
- バッチ12付近からNaNが発生し始め、以降のバッチで継続
- `cls_output`（モデルの出力）にNaNが含まれていた
- 最初のバッチは正常だが、2バッチ目以降でNaNが発生

---

## 原因の分析

### デバッグで判明した事実
1. **Forward passの途中でNaN発生**: Transformer Block 0の出力でNaNを検出
2. **Softmax attentionまでは正常**: attention scoresの計算は問題なし
3. **バッチ1は正常、バッチ2で発生**: 重み更新後にNaNが発生

### 推定された原因
上記の観察から、**勾配爆発（Gradient Explosion）** が根本原因と推定：
- 1バッチ目の学習で勾配が大きくなりすぎる
- optimizer.step()で重みが大幅に更新される
- 更新後の重みで2バッチ目を処理するとNaNが発生

---

## 修正内容と理由

### 1. 重み初期化の追加 (`src/model/logbert/bert.py`)

**仮定した原因**: PyTorchのデフォルト初期化では、重みの初期値が大きすぎて勾配が不安定になる

**対処**: Xavier初期化を適用し、gain=0.1で重みを小さく保つ

```python
def _init_weights(self):
    for module in self.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.1)  # 小さなgainで初期化
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 小さな標準偏差
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
```

**なぜこれが効くのか**: 
- 重みが小さいと、出力の値域も小さくなる
- 勾配が爆発しにくくなり、学習が安定する

---

### 2. Gradient Clipping強化 (`src/train_val.py`)

**仮定した原因**: max_norm=10.0では勾配クリッピングが弱すぎて、大きな勾配が通過してしまう

**対処**: max_normを10.0から1.0に変更

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| Gradient Clipping | `max_norm=10.0` | `max_norm=1.0` |

**なぜこれが効くのか**:
- 勾配のL2ノルムが1.0を超えないように制限
- 異常に大きな勾配による重みの急激な変化を防止

---

### 3. NaNガード機能の追加 (`src/train_val.py`)

**仮定した原因**: NaNが発生したバッチで学習を続けると、NaNが伝播して全ての重みがNaNになる

**対処**: 出力、Loss、勾配のいずれかにNaNが検出された場合、そのバッチをスキップ

```python
# 出力にNaNがある場合はスキップ
if torch.isnan(output["cls_output"]).any():
    continue

# LossがNaNの場合はスキップ
if torch.isnan(sum_loss):
    continue

# 勾配にNaNがある場合はスキップ
if has_nan_grad:
    continue
```

**なぜこれが効くのか**:
- NaNを含むバッチで重みを更新しないことで、NaNの伝播を防止
- 他の正常なバッチで学習を継続できる

---

### 4. 学習率とWarmupの調整 (`src/conf/bert/test.yaml`)

**仮定した原因**: 学習率0.0001でも高すぎて、重みの更新量が大きすぎる

**対処**: 学習率を1/10に削減し、Warmup期間を延長

| 項目 | 変更前 | 変更後 | 理由 |
|------|--------|--------|------|
| 学習率 | 0.0001 | 0.00001 | 重み更新量を抑制 |
| Warmup step | 5 | 10 | ゆっくり学習率を上げる |
| Warmup初期値 | 1e-5 | 1e-6 | より小さな初期学習率から開始 |

**なぜこれが効くのか**:
- 小さな学習率 = 小さな重み更新 = 安定した学習
- Warmupで序盤の不安定な勾配による影響を軽減

---

### 5. HyperSphereLoss無効化 (`src/conf/bert/test.yaml`)

**仮定した原因**: HyperSphereLossの計算過程でNaNが発生する可能性がある（問題の切り分け）

**対処**: HyperSphereLossのbiasを0にして無効化

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| Mask Loss bias | 0.9 | 1.0 |
| HyperSphere Loss bias | 0.1 | 0.0（無効化）|

**なぜこれが効くのか**:
- 問題の切り分けとして、まずMask Lossのみで安定動作を確認
- HyperSphereLossは後から段階的に有効化できる

---

## 検証結果

```
epoch 1 || TRAIN_Loss:2.3029 ||VAL_Loss:1.1817
epoch 2 || TRAIN_Loss:1.1627 ||VAL_Loss:1.1067
epoch 3 || TRAIN_Loss:1.1215 ||VAL_Loss:1.0977
```

**NaN発生なし、Lossが正常に減少** ✅

---

## プロジェクト構成

### ルートディレクトリのファイル

| ファイル | 用途 |
|---------|------|
| `README.md` | プロジェクトの説明・使用方法 |
| `NOTES.md` | 本ファイル（変更履歴・技術メモ） |
| `recoed.md` | データ修正の履歴 |
| `recovered_readme.md` | 過去の実験ドキュメント（参考用） |
| `pyproject.toml` | Python依存関係の定義 |
| `uv.lock` | 依存関係のロックファイル |

### 主要ディレクトリ

| ディレクトリ | 用途 |
|-------------|------|
| `src/` | メインのソースコード |
| `src_sakurai/` | 櫻井くんの過去実装（参考用） |
| `data/` | トレーニング・テストデータ |

---

## 今後のTODO

- [ ] 本番トレーニング（Epochs: 200）
- [ ] HyperSphereLossを段階的に有効化（bias=0.01から）
- [ ] num_workersを6以下に設定（警告対応）
