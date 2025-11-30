## 修正履歴

### [修正No:001]
- **修正日時**: 2025/11/19 17:21
- **修正場所**: data/raw/ScenarioData/{projectname}/Evtx
- **修正理由**: 開発時における、security.evtxのcsvへのパースにかかる時間削減
- **修正内容**: Security.evtxに対してレコードサイズを1/100にしたSecurity_1of100.evtxを作成
- **補足**: 抽出作業によってxmlの階層は破壊されていないと思われる。（要確認）

### [修正No:002]
- **修正日時**: 2025/11/19 19:37
- **修正場所**: data/raw/ScenarioData/{projectname}/Evtx
- **修正理由**: 開発時における、security.evtxのcsvへのパースにかかる時間削減
- **修正内容**: Security.evtxに対して先頭100件を抽出したSecurity_top1pct.evtxを作成
- **補足**: 抽出作業によってxmlの階層は破壊されていないと思われる。（要確認）
