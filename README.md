# LAD

log anomary detectionの略称かな？

開発はsingularityを使うよ。

## 環境構築
- singularityのコンテナに入るだけ
```bash
# コンテナ (読み込み専用)
singularity shell --bind /home/local/amiya:/amiya --nv lad.sif
# サンボドックス (書き込み可能)
singularity shell --bind /home/local/amiya:/amiya --nv lad
```

### (Optional) singularityのビルド
- 環境が大幅に変化した場合はビルドしておくと後々楽になる
    - 後輩への引継ぎ
    - どのような環境が自分でもわからなくなったとき
    - 同じ環境を再現できる自身が無いとき
- sudo権限が必要
    - 必要ならサーバー大臣を呼ぶこと
- @サーバー大臣
    - 以下のコマンドを実行してコンテナをビルドする
    - lad.sifは上書きされるので、気になるなら名前を変更するか、`lad2.sif` みたいな名前でビルドする 
```bash
# サンドボックス
sudo singularity build --sandbox lad lad.def
# コンテナ
sudo singularity build lad.sif lad.def
# サンドボックスからコンテナへ
sudo singularity build lad.sif lad
```


```
uv init
uv python install 3.11
.python-versionを3.11にしておこう
```

