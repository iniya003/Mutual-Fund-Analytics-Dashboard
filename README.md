# Bluestock Fintech — Mutual Fund Analytics Capstone
## Day 6: Advanced Analytics + Risk Metrics

This package contains the Day 6 deliverables for the Bluestock MF Capstone Project.

## Folder structure
```
day6_advanced_analytics/
├── data/                          10 source CSVs (as provided)
├── notebooks/
│   └── 05_advanced_analytics.ipynb    Executed notebook — code + charts + markdown insights
├── scripts/
│   ├── advanced_analytics.py          Standalone script version (run: python advanced_analytics.py)
│   └── recommender.py                 Standalone fund recommendation engine
├── outputs/
│   ├── var_cvar_report.csv            Historical VaR(95%) + CVaR for all 40 schemes
│   ├── rolling_sharpe_chart.png       Rolling 90-day Sharpe — 5 key large-cap funds
│   ├── rolling_sharpe_values.csv      Underlying rolling Sharpe numbers
│   ├── cohort_analysis.csv            Investor cohort behaviour by first-transaction year
│   ├── sip_continuity.csv             At-risk SIP investor flags (gap > 35 days)
│   ├── recommendations.csv            Top 3 funds per risk appetite (Low/Moderate/High)
│   ├── sector_hhi.csv                 Sector concentration (HHI) per equity fund
│   └── sector_hhi_chart.png           Top 15 most concentrated funds, chart
└── README.md
```

## How to run
```bash
cd scripts
pip install pandas numpy matplotlib
python advanced_analytics.py     # runs everything, writes to ../outputs/
python recommender.py            # prints recommendations for Low/Moderate/High risk appetite
```

Or open `notebooks/05_advanced_analytics.ipynb` in Jupyter — it is already executed, so all
charts and tables render immediately without re-running.

## Key insights (see notebook for full detail)
1. Small-cap schemes carry the highest historical VaR — roughly double the daily downside
   risk of large-cap funds.
2. The 2024 investor cohort contributes the largest share of total amount invested,
   reflecting the industry-wide SIP growth surge.
3. A large share of long-tenure SIP investors (6+ installments) show a gap of 35+ days
   between SIPs and are flagged at-risk of discontinuation.
4. Sector concentration (HHI) varies widely — several large-cap/thematic funds are
   "highly concentrated" (HHI ≥ 2500), often due to heavy Banking/Financials weight.
5. Within each risk band, the top Sharpe-ranked fund is typically a Direct plan with a
   materially lower expense ratio than its Regular plan equivalent.

## Next steps for the full capstone
This is Day 6 of the 7-day capstone plan. Day 7 (Final Report + Presentation + GitHub
push) can now pull the CSVs and PNGs in `outputs/` directly into the final PDF report and
slide deck.
