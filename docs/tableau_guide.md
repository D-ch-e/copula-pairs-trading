# Tableau Public 建置指南

目標：用 Tableau Public 做一份給面試官看的 dashboard，先一眼看到結論，再往下探索細節。資料由 `python -m scripts.export_tableau` 產生在 `exports/tableau/`，欄位定義見 [tableau_data_dictionary.md](tableau_data_dictionary.md)。

> Tableau Public 上的 workbook 與資料一律公開。匯出檔已經只放 benchmark 的月報酬；策略報酬是你自己的計算結果。

## 0. 準備

1. 先跑完：`python -m scripts.run_experiments`，再 `python -m scripts.export_tableau`。終端機最後會印出「The dashboard should show」核對清單，做完後要逐項對上。
2. 下載安裝 [Tableau Public](https://www.tableau.com/products/public/download)（免費，Mac 可用），註冊帳號。

## 1. 連接資料

建立三個資料來源（Data Source）：

**A. Results**（主要）
1. Connect → To a File → Text file → 選 `runs.csv`。
2. 把 `strategy_daily.csv`、`monthly_returns.csv`、`trades.csv`、`subperiods.csv`、`significance.csv` 依序拖到畫布上，每個都和 `runs.csv` 建立 Relationship：`run_id = run_id`（不要用 Join）。
3. 檢查欄位型別：`date`、`month`、`signal_date`、`entry_date`、`exit_date` 為 Date；`is_main_spec`、`is_benchmark` 為 Boolean（若被讀成文字，點欄位圖示改型別）。
4. 同名欄位（如 `pair`、`spec`）在多個表都有，**篩選與顏色一律用 `runs.csv` 的版本**。

**B. Copula fits**：只連 `copula_fits.csv`。

**C. Tail calibration**：只連 `tail_calibration.csv`。

## 2. 參數

在 Results 資料來源建立三個參數（右鍵 → Create Parameter），之後在每個 dashboard 右上角 Show Parameter：

| 名稱 | 型別 | 允許值 | 預設 |
|---|---|---|---|
| p_Pair | String, List | GLD-SLV, GDX-GDXJ, XOM-CVX, KO-PEP | GLD-SLV |
| p_Cost | Integer, List | 0, 10, 25, 50, 100 | 50 |
| p_Exposure | Float, List | 2（顯示為 200%）, 1（顯示為 100%） | 2 |

## 3. 計算欄位

**Results**

```
// Show Run：選定配對的策略（選定成本與曝險）＋該配對的 benchmark
[pair] = [p_Pair] AND ([is_benchmark] OR ([cost_bps] = [p_Cost] AND [gross_exposure] = [p_Exposure]))
```

KPI（固定顯示主規格，不隨參數變動）：

```
// KPI Main Sharpe
{FIXED : MAX(IF [is_main_spec] THEN [sharpe] END)}
// KPI Main Return
{FIXED : MAX(IF [is_main_spec] THEN [ann_return] END)}
// KPI Main MDD
{FIXED : MAX(IF [is_main_spec] THEN [max_drawdown] END)}
// KPI Breakeven
{FIXED : MAX(IF [is_main_spec] THEN [breakeven_bps] END)}
// KPI Main Gross Sharpe（同規格、0 bps）
{FIXED : MAX(IF [pair] = "GLD-SLV" AND [spec] = "Copula t (main)" AND [cost_bps] = 0 AND [gross_exposure] = 2 THEN [sharpe] END)}
// KPI GLD Return / KPI GLD Sharpe / KPI GLD MDD
{FIXED : MAX(IF [spec] = "GLD" THEN [ann_return] END)}
{FIXED : MAX(IF [spec] = "GLD" THEN [sharpe] END)}
{FIXED : MAX(IF [spec] = "GLD" THEN [max_drawdown] END)}
```

**Copula fits**

```
// t Better OOS（每月 1 或 0，取平均即為比例）
IIF([t_oos_loglik] > [gaussian_oos_loglik], 1, 0)
// Fit On Bound
[t_nu_at_bound] = 1 OR [t_rho_at_bound] = 1
```

數字格式：報酬、MDD、VaR → Percentage 1 位小數；Sharpe → Number 2 位；breakeven → Number 0 位並加 " bps" 後綴。

## 4. 配色（與 notebook 圖一致）

在任一以 `spec` 上色的圖：Color → Edit Colors，逐項點色塊，用 Mac 色票的 Hex 輸入：

| spec | Hex |
|---|---|
| Copula t (main) | #2a78d6 |
| Copula Gaussian | #eb6834 |
| Copula t, paper exit | #1baf7a |
| z-score baseline | #eda100 |
| GLD | #3d3c39 |
| SPY | #8a8984 |
| 50/50 GLD+SLV | #b5b4ae |

設定一次後，同一資料來源的其他圖會沿用。背景用白色或 #fcfcfb；格線淡灰。

## 5. 工作表

每張都把 `Show Run`（Results）拖到 Filters 並選 True，除非另有說明。

| # | 名稱 | 設定 |
|---|---|---|
| 1 | Headline | Results。Marks = Text；把各 KPI 欄位拖到 Text，編輯文字成：「Net of 50 bps, the main copula strategy on GLD-SLV returned <KPI Main Return> a year (Sharpe <KPI Main Sharpe>), against <KPI GLD Return> for buy-and-hold GLD (Sharpe <KPI GLD Sharpe>).」不需要 Show Run 篩選。這樣結論句的數字來自資料，不是手打。 |
| 2 | KPI cards | Results。Measure Names 放 Columns、Measure Values 放 Text，只留 KPI Main Sharpe、KPI Main Return、KPI Main MDD、KPI Breakeven、KPI GLD Return；字放大。不需要 Show Run 篩選。 |
| 3 | Equity curve | Results。Columns：`month`（monthly_returns，Exact Date，Continuous）；Rows：AVG(`wealth`)（monthly_returns）；Color：`spec`（runs）；Marks = Line。Y 軸 Edit Axis → Logarithmic。標題「Growth of $1 NAV (monthly)」。 |
| 4 | Drawdown | 同上，Rows 換成 AVG(`drawdown`)，格式為百分比；Marks = Line 或 Area。標題註明「monthly; daily max drawdown in the table」。 |
| 5 | Metrics table | Results。Rows：`spec`；Columns：Measure Names；Text：Measure Values，只保留 ann_return、ann_vol、sharpe、sortino、max_drawdown、n_trades、win_rate、breakeven_bps，以及 significance 的 p_nw、sharpe_ci_low、sharpe_ci_high。 |
| 6 | Cost sensitivity | Results。Filters：`pair` = p_Pair（用計算欄位 `[pair] = [p_Pair]`）、`is_benchmark` = False、`gross_exposure` = 2（**不要**用 Show Run，因為要看所有成本）。Columns：`cost_bps`（Dimension → Continuous）；Rows：AVG(`ann_return`)；Color：`spec`；Marks = Line，再加 Circle。Analytics → Constant Line = 0。 |
| 7 | Copula parameters | Copula fits。Filter：`[pair] = [p_Pair]`。Columns：`month`（Continuous）；Rows：AVG(`t_rho`)、AVG(`t_nu`)、AVG(`t_tail_dep`) 三個疊放（各自一個軸，不是雙軸）。`t_nu` 軸改 Logarithmic；`t_tail_dep` 百分比。在 ρ 的那一列再加 AVG(`gaussian_rho`) 用 Measure Values 比較。 |
| 8 | t vs Gaussian by pair | Copula fits。Rows：`pair`；Columns：AVG(`t Better OOS`)，百分比；加 Constant Line 50%。這張取代「t beats Gaussian 57–69%」的 KPI，更直觀。 |
| 9 | Tail calibration | Tail calibration。Filters：`[pair] = [p_Pair]`、`version` 排除 Marginal exceedance。Columns：`version`、`measure`；Rows：SUM(`value`)；Color：`measure`；Marks = Bar。加 Constant Line 0.0025（independence）。 |
| 10 | Marginal drift | Tail calibration。Filters：`[pair] = [p_Pair]`、`version` = Marginal exceedance。Rows：`measure`；Text：SUM(`value`)，百分比。說明文字：「nominal 5%」。 |
| 11 | Stress periods | Results。Filter：`period` 排除 other。Columns：`period_label`（依 `period_order` 排序）；Rows：AVG(`total_return`)；Color：`spec`；Marks = Bar（side-by-side：Columns 再加 `spec`）。 |
| 12 | Holding period | Results。Filter：`spec`（runs）做成單選下拉。`holding_days` 右鍵 Create → Bins（size 1）；Columns：holding_days (bin)；Rows：CNT(`trade_no`)。 |
| 13 | Exit reasons | Results。同上篩選。Rows：`exit_reason`；Columns：CNT(`trade_no`) → Quick Table Calculation → Percent of Total。 |

## 6. Dashboard

Size：Fixed 1200 × 900。建立四個 dashboard，發佈時以分頁顯示：

1. **Overview**：上方標題「Copula pairs trading on gold and silver」＋ Headline（#1）＋ disclaimer（「Research project, not investment advice. Daily data 2007–2025.」）；第二列 KPI cards（#2）；左下 Equity curve（#3）疊 Drawdown（#4）；右下 Cost sensitivity（#6）；最下 Metrics table（#5）。右上 Show Parameter：p_Pair、p_Cost、p_Exposure。
2. **Copula dependence**：Copula parameters（#7）、t vs Gaussian by pair（#8）、Tail calibration（#9）、Marginal drift（#10）。加一段說明：「t-copula fits better by out-of-sample log-likelihood in all four pairs; once marginal drift is removed, its tail-count advantage is not consistent.」
3. **Stress periods & trades**：Stress periods（#11）、Holding period（#12）、Exit reasons（#13）。
4. **Method**：Text 物件放 README 的「Paper vs. this project」對照表與 Limitations，以及 GitHub 連結（repo 公開後）。

每個 dashboard 再用 Device Preview → Phone 建一個手機版面，元件改為上下排列。

## 7. 發佈

1. File → Save to Tableau Public As…，登入，命名「Copula Pairs Trading: Gold vs Silver」。
2. 在 Tableau Public 網頁的 viz 設定中：勾選 **Show Sheets as Tabs**；依喜好決定是否允許下載 workbook。
3. 複製網址，交給我加進 README 的 Dashboard 一節，也可放到履歷。

## 8. 發佈前核對

以 `export_tableau` 印出的核對清單為準（2026-10-26 的資料）：

- 主規格（GLD-SLV、Copula t、50 bps、200%）：淨 Sharpe −0.99、年化 −7.9%、最大回撤 78.5%、216 筆交易
- 買進持有 GLD：年化 10.2%、Sharpe 0.56
- 損益兩平成本：−17 bps
- t 勝出月份比例：57%–69%（四個配對）
- 子期間（GLD-SLV 主規格）：金融海嘯 −3.5%、2011 −18.4%、COVID −11.0%、2022 −5.5%

注意：Equity curve 與 Drawdown 用的是月資料，波動大的序列在圖上的最大回撤會比表格（日資料）小，屬正常。例如 GLD：月資料 42.9%、日資料 45.6%；主規格兩者都是 78.5%。

## 履歷描述（建完後可用）

*Built an interactive Tableau Public dashboard of a copula pairs-trading backtest (4 pairs × 4 strategies × 5 cost levels), with parameter-driven equity, drawdown, cost-sensitivity and copula-dependence views fed by a reproducible Python/SQLite export pipeline.*
