"""
FAQ Chatbot (CustomTkinter + spaCy + scikit‑learn)

➡️  Now reads its FAQ list exclusively from **faq_data.json** (same folder).  
    If the file is missing, the script raises a clear error so you remember to ship it.

Quick setup (inside venv):
    pip install customtkinter spacy scikit-learn nltk
    python -m spacy download en_core_web_sm

Place **faq_data.json** next to this script before running.
"""

import customtkinter as ctk
import json
import pathlib
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# 1.  Load FAQ JSON (no fallback hard‑coded list)                           
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
FAQ_PATH   = SCRIPT_DIR / r"C:\\Users\\HP\\Desktop\\FYP\\faq product data.json"

if not FAQ_PATH.exists():
    raise FileNotFoundError(
        f"faq_data.json not found in {SCRIPT_DIR}.\n"
        "→ Copy the JSON file you generated earlier into this folder "
        "or change FAQ_PATH accordingly."
    )

faq_items = json.loads(FAQ_PATH.read_text(encoding="utf-8"))

questions = [item["question"] for item in faq_items]
answers   = [item["answer"]   for item in faq_items]

# ---------------------------------------------------------------------------
# 2.  NLP pipeline                                                          
# ---------------------------------------------------------------------------

nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(text: str):
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]

vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer)
faq_matrix = vectorizer.fit_transform(questions)

def answer_query(query: str):
    vec    = vectorizer.transform([query])
    scores = cosine_similarity(vec, faq_matrix).flatten()
    best   = int(np.argmax(scores))
    return answers[best], float(scores[best])

# ---------------------------------------------------------------------------
# 3.  CustomTkinter GUI                                                     
# ---------------------------------------------------------------------------

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class FAQChatbot(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("FAQ Chatbot")
        self.geometry("500x600")

        # transcript box
        self.chat_box = ctk.CTkTextbox(self, wrap="word", state="disabled")
        self.chat_box.pack(fill="both", expand=True, padx=12, pady=12)

        # input row
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=12, pady=(0, 12))

        self.entry = ctk.CTkEntry(frame)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 8), pady=8)
        self.entry.bind("<Return>", self.on_send)

        ctk.CTkButton(frame, text="Send", command=self.on_send).pack(side="right", pady=8)

        self.append("Bot", "Hi! Ask me anything 😊")

    def append(self, who: str, msg: str):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"{who}: {msg}\n\n")
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")

    def on_send(self, *_):
        q = self.entry.get().strip()
        if not q:
            return
        self.entry.delete(0, "end")
        self.append("You", q)
        a, s = answer_query(q)
        self.append("Bot", f"{a}  (sim ≈ {s:.2f})")


if __name__ == "__main__":
    FAQChatbot().mainloop()

