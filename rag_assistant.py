import chromadb
import ollama

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="stock_knowledge")

def ask_grounded_question(question):
    results = collection.query(query_texts=[question], n_results=4)
    relevant_facts = "\n".join(results['documents'][0])

    prompt = f"""You are a financial analysis assistant. Answer the user's question 
using ONLY the facts below, which come from a validated 10-year statistical study. 
Do not invent numbers not present in these facts.

FACTS:
{relevant_facts}

QUESTION: {question}

ANSWER:"""

    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']

if __name__ == "__main__":
    answer = ask_grounded_question("Should I invest in Nvidia? What does the backtest say?")
    print(answer)