# Live2D Dataset Curation

這份資料集目前按「人形 Live2D 部件分割」整理；非人形插畫、深度圖、說明/規約頁、macOS sidecar 檔都不進訓練。

## 本次整理結果

- 掃描到 `28` 個 `.psd`。
- 忽略 `10` 個 PSD：`3` 個 macOS `._*.psd` sidecar、`4` 個 `*_depth.psd`、`2` 個說明/規約頁、`1` 個非人形插畫。
- 進入 COCO 的有效 PSD：`18` 個。
- 探索版 taxonomy 從原本 30 類擴展到 `42` 類；原始 `1-30` 類 ID 保持不變。
- 新增 `hair/eyes/eyebrows/cheek/hands/arms/feet/legs/shoes/tail/torso/ears`，用來承接沒有左右或前後語義的 layer。
- `帽子`、`油纸伞`、`油紙傘`、`傘/伞`、`武器` 已納入 `accessory`。
- 探索版輸出產生 `242` 個 COCO segmentation annotations。
- 探索版 `unmatched_layers.json` 從 `294` 行降到 `49` 行。
- 新增 24 類訓練版 canonical taxonomy，輸出 `217` 個 annotations，剩餘 `66` 行 unmatched；此版本刻意保守，會把非 canonical 或需看 preview 的 layer 留在 unmatched。
- RLE mask decode 檢查通過，沒有壞 mask。

## 主要輸出

- 人工排除/override 檔：`dataset/live2d_psd/layer_overrides.json`
- 訓練版 canonical taxonomy：`scripts/live2d/live2d_canonical_taxonomy.json`
- 訓練版 COCO 輸出：`dataset/live2d_parts_canonical_check`
- COCO 輸出：`dataset/live2d_parts_check`
- 被忽略 PSD 報告：`dataset/live2d_parts_check/ignored_psds.json`
- 未命中 layer 報告：`dataset/live2d_parts_check/unmatched_layers.json`
- 原始 PSD 總覽圖：`dataset/live2d_dataset_audit/psd_contact_sheet.jpg`
- COCO mask overlay 總覽圖：`dataset/live2d_dataset_audit/converted_overlay_sheet.jpg`

## 亂碼處理

- JSON 檔案使用 UTF-8 寫出，`ensure_ascii=False`。
- audit 圖片改用 `scripts/live2d/audit_dataset.py` 生成，會優先使用 Noto Sans / 微軟雅黑 / Meiryo 等中日文字體。
- PowerShell 若仍顯示亂碼，通常是終端 code page 問題；實際 JSON/Markdown 可用 UTF-8 編輯器正常開啟。

## 排除規則

- `__MACOSX/**/._*.psd`：不是 PSD 主檔，只是 macOS resource-fork sidecar。
- `*_depth.psd`：深度/灰階 companion，不是 RGB 部件分割資料。
- visible composite 是 readme、使用規約、黑底說明圖的 PSD：不作訓練圖。
- 純非人形主體插畫：不混入目前的人形 taxonomy。

## 已補 taxonomy

- RRAILab：`tail`、`legs`、`shoes`。
- Rikka：`身体`、`髪`、`目`、`眉`、`チーク`。
- Domino base：`hands`、`arms`、`feet`、`legs`、`torso`。
- ハスキー：`しっぽ`、`ふくらはぎ`、`太もも`、`靴`、`手`、`腕`。
- 第二輪明確 layer：`けもみみ -> ears`、`拖尾 -> tail`、`襟 -> upper clothes`、`鎖骨/体 -> torso`、`赤面 -> cheek`。
- 明確道具：`帽子`、`油纸伞`、`油紙傘`、`武器`、`铃铛`。

## 下一步建議

- 對剩下的 `49` 行 unmatched 做第三輪人工審核。
- 優先檢查 `图层组.../四肢`、頭部縮寫層、`ひらひら`、背景組裡的裝飾層；這些可能是可用部件或道具，但需要看 layer preview 才能安全分類。
- 不要把背景、說明文字、純裝飾亂碼 layer 強行塞進人體部件類別。
