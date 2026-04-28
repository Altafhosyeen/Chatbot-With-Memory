"""
Chatbot with Memory — local LLM via Ollama
Usage:
  python chatbot.py                    # interactive CLI
  python chatbot.py --model mistral    # choose model
  python chatbot.py --web              # Gradio web UI
Requires: ollama running locally (https://ollama.ai)
  ollama pull mistral
"""
import os, argparse, json
from datetime import datetime
from collections import deque

os.makedirs("logs", exist_ok=True)

try:
    import requests
except ImportError:
    print("[ERROR] pip install requests"); exit(1)

OLLAMA_URL = "http://localhost:11434/api/chat"
LOG_FILE   = os.path.join("logs", f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

SYSTEM_PROMPT = """You are a helpful, friendly AI assistant with memory of the conversation.
You remember what the user has told you and refer back to it naturally.
Be concise but thorough. If you don't know something, say so honestly."""

def chat_ollama(model, messages):
    payload = {"model": model, "messages": messages, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "[ERROR] Ollama not running. Start with: ollama serve"
    except Exception as e:
        return f"[ERROR] {e}"

def run_cli(model="mistral", max_history=20):
    history = [{"role":"system","content":SYSTEM_PROMPT}]
    log     = []
    print(f"\n[Chatbot] Model: {model}  |  'clear' to reset  |  'save' to export  |  'q' to quit\n")
    print("─"*60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input: continue
        if user_input.lower() == 'q': break
        if user_input.lower() == 'clear':
            history = [{"role":"system","content":SYSTEM_PROMPT}]
            log.clear(); print("[Memory cleared]"); continue
        if user_input.lower() == 'save':
            with open(LOG_FILE,"w") as f: json.dump(log,f,indent=2)
            print(f"[Saved] {LOG_FILE}"); continue

        history.append({"role":"user","content":user_input})
        # Keep context window manageable
        ctx = [history[0]] + history[max(1,len(history)-max_history):]

        print("\nAssistant: ", end="", flush=True)
        response = chat_ollama(model, ctx)
        print(response)

        history.append({"role":"assistant","content":response})
        log.append({"ts":datetime.now().isoformat(),"user":user_input,"assistant":response})

    # Auto-save on exit
    if log:
        with open(LOG_FILE,"w") as f: json.dump(log,f,indent=2)
        print(f"\n[Session saved] {LOG_FILE}")

def run_web(model="mistral"):
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] pip install gradio"); return

    history_store = [{"role":"system","content":SYSTEM_PROMPT}]

    def respond(message, chat_history):
        history_store.append({"role":"user","content":message})
        ctx = [history_store[0]] + history_store[max(1,len(history_store)-20):]
        reply = chat_ollama(model, ctx)
        history_store.append({"role":"assistant","content":reply})
        chat_history.append((message, reply))
        return "", chat_history

    def clear_fn():
        history_store.clear()
        history_store.append({"role":"system","content":SYSTEM_PROMPT})
        return []

    with gr.Blocks(title="Chatbot with Memory") as demo:
        gr.Markdown(f"# Chatbot with Memory\nModel: `{model}` via Ollama")
        chatbox = gr.Chatbot(height=500)
        msg     = gr.Textbox(placeholder="Type your message...", show_label=False)
        with gr.Row():
            send  = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear Memory")
        send.click(respond,  [msg, chatbox], [msg, chatbox])
        msg.submit(respond,  [msg, chatbox], [msg, chatbox])
        clear.click(clear_fn, outputs=chatbox)
    demo.launch()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--web",   action="store_true")
    args = ap.parse_args()
    if args.web: run_web(args.model)
    else:        run_cli(args.model)
