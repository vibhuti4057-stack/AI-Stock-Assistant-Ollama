import threading
import ollama

class StreamState:
    def __init__(self):
        self.text = ""
        self.done = False
        self.stop_requested = False

from global_state import is_stop_requested

def stream_response(prompt, model_name, state: StreamState):
    try:
        stream = ollama.chat(model=model_name, messages=[{'role': 'user', 'content': prompt}], stream=True)
        for chunk in stream:
            if state.stop_requested or is_stop_requested():
                break
            state.text += chunk['message']['content']
    except Exception as e:
        state.text += f"\n[Error: {e}]"
    state.done = True

def start_streaming(prompt, model_name):
    state = StreamState()
    thread = threading.Thread(target=stream_response, args=(prompt, model_name, state))
    thread.start()
    return state, thread