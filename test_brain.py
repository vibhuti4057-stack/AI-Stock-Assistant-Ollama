import ollama

response = ollama.chat(model='llama3.2', messages=[
    {'role': 'user', 'content': 'Explain what CAGR means in one sentence.'}
])
print(response['message']['content'])