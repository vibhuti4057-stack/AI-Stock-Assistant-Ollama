import chromadb
import pandas as pd

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="stock_knowledge")

summary = pd.read_csv("data/summary.csv", index_col=0)
risk = pd.read_csv("data/risk_metrics.csv", index_col=0)
backtest = pd.read_csv("data/backtest_df.csv", index_col=0)

documents = []
ids = []
doc_id = 0

for company in summary.index:
    text = (f"{company}: average close price ${summary.loc[company,'avg_close']}, "
            f"volatility {summary.loc[company,'volatility_std_pct']}%, "
            f"total growth {summary.loc[company,'total_growth_pct']}%, "
            f"Sharpe-like ratio {summary.loc[company,'sharpe_like_ratio']}.")
    documents.append(text)
    ids.append(f"summary_{doc_id}")
    doc_id += 1

for company in risk.index:
    text = (f"{company}: CAGR {risk.loc[company,'CAGR_%']}%, "
            f"maximum drawdown {risk.loc[company,'Max_Drawdown_%']}%.")
    documents.append(text)
    ids.append(f"risk_{doc_id}")
    doc_id += 1

for company in backtest.index:
    worked = backtest.loc[company, 'within_predicted_range']
    text = (f"{company}: backtest validation result — the forecasting method "
            f"{'worked correctly' if str(worked)=='True' else 'FAILED'} when tested against real historical outcomes.")
    documents.append(text)
    ids.append(f"backtest_{doc_id}")
    doc_id += 1

collection.add(documents=documents, ids=ids)
print(f"Knowledge base built with {len(documents)} facts.")