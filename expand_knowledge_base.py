import chromadb
import pandas as pd

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="stock_knowledge")

COMPANIES = ["Amazon", "Apple", "Google", "Microsoft", "Accenture", "Nvidia"]
TICKERS = {"Amazon":"AMZN","Apple":"AAPL","Google":"GOOGL","Microsoft":"MSFT","Accenture":"ACN","Nvidia":"NVDA"}

documents = []
ids = []
doc_id = 2000

# --- Sector data ---
try:
    sp500 = pd.read_csv("data/sp500_companies.csv")
    for company, ticker in TICKERS.items():
        row = sp500[sp500['symbol'] == ticker]
        if not row.empty:
            r = row.iloc[0]
            text = (f"{company} sector information: sector is {r['sector']}, "
                    f"sub-industry is {r['sub_industry']}, headquartered in {r['headquarters']}, "
                    f"founded in {r['founded']}.")
            documents.append(text); ids.append(f"sector_{doc_id}"); doc_id += 1
    print(f"Added sector facts for {len(TICKERS)} companies.")
except FileNotFoundError:
    print("sp500_companies.csv not found in data/ — skipping sector facts.")

# --- Macro / interest rate data, focused on the 2022 downturn ---
try:
    yield_curve = pd.read_csv("data/yield_curve.csv", parse_dates=['date'])
    macro = pd.read_csv("data/macro_indicators.csv", parse_dates=['date'])

    period_2022 = yield_curve[(yield_curve['date'] >= '2022-01-01') & (yield_curve['date'] <= '2022-12-31')]
    if not period_2022.empty:
        start_10y = period_2022['gs10'].iloc[0]
        end_10y = period_2022['gs10'].iloc[-1]
        text = (f"Macro context for 2022: the 10-year US Treasury yield rose from {start_10y}% "
                f"in January 2022 to {end_10y}% by December 2022, a significant rate increase. "
                f"This directly supports this project's finding that 2022 was a broad market "
                f"downturn driven by rising interest rates, affecting nearly all six companies studied.")
        documents.append(text); ids.append(f"macro_{doc_id}"); doc_id += 1

    cpi_2022 = macro[(macro['date'] >= '2022-01-01') & (macro['date'] <= '2022-12-31')]
    if not cpi_2022.empty and 'cpi_yoy_pct' in cpi_2022.columns:
        max_cpi = cpi_2022['cpi_yoy_pct'].max()
        text = (f"Macro context for 2022: inflation (CPI year-over-year) peaked around {max_cpi}% "
                f"during 2022, prompting the interest rate increases referenced above.")
        documents.append(text); ids.append(f"macro_{doc_id}"); doc_id += 1

    print("Added macro/interest-rate context facts for 2022.")
except FileNotFoundError:
    print("yield_curve.csv or macro_indicators.csv not found in data/ — skipping macro facts.")
except Exception as e:
    print(f"Macro data had an unexpected format ({e}) — skipping macro facts, everything else still works.")

if documents:
    collection.add(documents=documents, ids=ids)
    print(f"Added {len(documents)} new facts to the knowledge base.")
else:
    print("No new facts added.")