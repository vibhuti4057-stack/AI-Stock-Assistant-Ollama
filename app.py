import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import itertools

st.set_page_config(page_title="Beyond the Chart", page_icon="📈")

# ---------- Load data from CSV (no version/pickle issues) ----------
df = pd.read_csv("saved_models/df.csv", parse_dates=['date'])
summary = pd.read_csv("saved_models/summary.csv", index_col=0)
risk_metrics = pd.read_csv("saved_models/risk_metrics.csv", index_col=0)
backtest_df = pd.read_csv("saved_models/backtest_df.csv", index_col=0)

with open("saved_models/companies.txt") as f:
    COMPANIES = f.read().split(",")

feature_cols = ['close_lag1', 'close_lag2', 'close_lag3', 'ma_5', 'volume_lag1']

def build_features(company_df):
    d = company_df.sort_values('date').copy()
    d['close_lag1'] = d['close'].shift(1)
    d['close_lag2'] = d['close'].shift(2)
    d['close_lag3'] = d['close'].shift(3)
    d['ma_5'] = d['close'].rolling(5).mean()
    d['volume_lag1'] = d['volume'].shift(1)
    d['target_next_close'] = d['close'].shift(-1)
    return d.dropna()

# ---------- Retrain price prediction models fresh (fast, cached) ----------
@st.cache_resource
def train_all_models():
    trained = {}
    for company in COMPANIES:
        company_df = df[df['company'] == company]
        prepared = build_features(company_df)
        X = prepared[feature_cols]
        y = prepared['target_next_close']
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        trained[company] = model
    return trained

trained_models = train_all_models()

# ---------- Retrain intent classifier fresh (fast, cached) ----------
@st.cache_resource
def train_intent_classifier():
    templates = {
        "predict_price": ["what will {c} price be tomorrow", "predict {c} stock", "forecast {c} price",
            "what is the future price of {c}", "give me a price prediction for {c}",
            "where will {c} stock go next", "estimate {c} closing price", "what's the projected price of {c}"],
        "risk": ["how risky is {c}", "what is the volatility of {c}", "is {c} a safe stock",
            "show me the risk of {c}", "how much can {c} drop", "what is the drawdown for {c}",
            "how volatile is {c}", "what is the downside risk of {c}"],
        "trend_chart": ["show me the trend of {c}", "plot {c} price history", "graph of {c} over time",
            "show {c} chart", "visualize {c} performance", "display {c} price movement"],
        "opinion": ["should i invest in {c}", "is {c} a good buy", "what do you think about {c} stock",
            "give your opinion on {c}", "is {c} worth buying", "would you recommend {c}",
            "is now a good time to buy {c}"],
    }
    training_examples = []
    for intent, phrases in templates.items():
        for phrase in phrases:
            for c in COMPANIES:
                training_examples.append((phrase.format(c=c), intent))
    compare_templates = ["compare {a} and {b}", "which is better {a} or {b}", "{a} vs {b} performance", "how does {a} compare to {b}"]
    for phrase in compare_templates:
        for a, b in itertools.permutations(COMPANIES, 2):
            training_examples.append((phrase.format(a=a, b=b), "compare"))

    texts = [t[0] for t in training_examples]
    labels = [t[1] for t in training_examples]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, labels)
    return model, vectorizer

intent_model, vectorizer = train_intent_classifier()

# ---------- Core logic functions ----------
def get_latest_features(company):
    company_df = df[df['company'] == company].sort_values('date')
    prepared = build_features(company_df)
    return prepared[feature_cols].iloc[[-1]]

def predict_next_price(company):
    model = trained_models[company]
    features = get_latest_features(company)
    return round(model.predict(features)[0], 2)

def classify_intent(question):
    X = vectorizer.transform([question.lower()])
    return intent_model.predict(X)[0]

def extract_company(question):
    question = question.lower()
    for c in COMPANIES:
        if c.lower() in question:
            return c
    return None

def handle_predict_price(company):
    predicted_price = predict_next_price(company)
    text = f"Based on recent price patterns, the model's next-session estimate for {company} is ${predicted_price:.2f}."
    numbers = {"predicted_price": predicted_price}
    fig, ax = plt.subplots(figsize=(8,4))
    sub = df[df['company']==company].tail(60)
    ax.plot(sub['date'], sub['close'])
    ax.axhline(predicted_price, color='red', linestyle='--', label='Predicted next price')
    ax.set_title(f"{company} — Recent Price & Prediction")
    ax.legend()
    return text, numbers, fig

def handle_risk(company):
    vol = summary.loc[company, 'volatility_std_pct']
    dd = risk_metrics.loc[company, 'Max_Drawdown_%']
    sharpe = summary.loc[company, 'sharpe_like_ratio']
    text = f"{company} has a daily volatility of {vol}%, a maximum historical drawdown of {dd}%, and a risk-adjusted (Sharpe-like) score of {sharpe}."
    numbers = {"volatility_%": vol, "max_drawdown_%": dd, "sharpe_like": sharpe}
    fig, ax = plt.subplots(figsize=(8,4))
    ax.boxplot(df[df['company']==company]['daily_return'].dropna()*100, vert=True)
    ax.set_title(f"{company} — Daily Return Spread")
    return text, numbers, fig

def handle_trend_chart(company):
    sub = df[df['company']==company]
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(sub['date'], sub['close'])
    ax.set_title(f"{company} — 10-Year Price Trend")
    text = f"Here is {company}'s full price trend over the last 10 years."
    numbers = {"current_price": round(sub['close'].iloc[-1], 2)}
    return text, numbers, fig

def handle_compare(companies_mentioned):
    a, b = companies_mentioned[0], companies_mentioned[1]
    text = f"{a} CAGR: {risk_metrics.loc[a,'CAGR_%']}% vs {b} CAGR: {risk_metrics.loc[b,'CAGR_%']}%."
    numbers = {a: risk_metrics.loc[a,'CAGR_%'], b: risk_metrics.loc[b,'CAGR_%']}
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar([a,b], [risk_metrics.loc[a,'CAGR_%'], risk_metrics.loc[b,'CAGR_%']])
    ax.set_title("CAGR Comparison")
    return text, numbers, fig

def handle_opinion(company):
    cagr = risk_metrics.loc[company, 'CAGR_%']
    model_worked = backtest_df.loc[company, 'within_predicted_range']
    first_opinion = f"Based on its 10-year CAGR of {cagr}%, {company} has shown {'strong' if cagr > 20 else 'weak'} historical growth."
    if str(model_worked) in ["True", "1", "1.0"]:
        second_opinion = f"Our backtest confirmed this style of projection worked reasonably well for {company} historically, so this view carries moderate confidence."
    else:
        second_opinion = f"Caution: our backtest showed this style of projection FAILED to predict {company}'s actual performance historically — real results exceeded even the optimistic estimate. Treat the first opinion as a rough baseline only, not a reliable forecast."
    text = first_opinion + " " + second_opinion
    numbers = {"cagr_%": cagr, "backtest_reliable": str(model_worked)}
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(["CAGR %"], [cagr], color='green' if cagr>20 else 'orange')
    ax.set_title(f"{company} — Growth Snapshot")
    return text, numbers, fig

def answer_question(question):
    intent = classify_intent(question)
    company = extract_company(question)
    if intent == "compare":
        mentioned = [c for c in COMPANIES if c.lower() in question.lower()]
        if len(mentioned) < 2:
            return "Please mention two companies to compare.", {}, None
        return handle_compare(mentioned)
    if company is None:
        return "I couldn't identify which company you're asking about. Try naming one directly.", {}, None
    if intent == "predict_price":
        return handle_predict_price(company)
    elif intent == "risk":
        return handle_risk(company)
    elif intent == "trend_chart":
        return handle_trend_chart(company)
    elif intent == "opinion":
        return handle_opinion(company)
    else:
        return "I'm not sure how to answer that yet.", {}, None

# ---------- The website itself ----------
st.title("Beyond the Chart — Stock Analysis Assistant")
st.write("Ask about **prediction, risk, trend, comparison, or opinion** for: Amazon, Apple, Google, Microsoft, Accenture, Nvidia")

question = st.text_input("Ask your question:")

if question:
    text, numbers, fig = answer_question(question)
    st.subheader("Answer")
    st.write(text)
    if numbers:
        st.subheader("Numbers")
        st.json(numbers)
    if fig:
        st.subheader("Chart")
        st.pyplot(fig)