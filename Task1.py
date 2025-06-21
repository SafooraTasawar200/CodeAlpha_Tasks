import customtkinter as ctk
from googletrans import Translator, LANGUAGES
from gtts import gTTS
from gtts.lang import tts_langs
import pyperclip
from pyperclip import PyperclipException          # ← NEW
from tkinter import messagebox                    # ← already used
import playsound, os

# ── gTTS‑supported language codes ─────────────────────────────────────────────
gtts_langs = set(tts_langs().keys())

# ── CustomTkinter base window ─────────────────────────────────────────────────
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("🌍 Modern Language Translator")
app.geometry("750x600")

translator = Translator()
language_codes = list(LANGUAGES.keys())

# ── Functions ─────────────────────────────────────────────────────────────────
def translate_text():
    text = input_box.get("1.0", "end-1c").strip()
    if not text:
        output_box.delete("1.0", "end")
        output_box.insert("1.0", "❗ Please type text to translate.")
        return
    try:
        translated = translator.translate(text, src=source_lang.get(), dest=target_lang.get())
        output_box.delete("1.0", "end")
        output_box.insert("1.0", translated.text)
    except Exception as e:
        output_box.delete("1.0", "end")
        output_box.insert("1.0", f"❌ Error: {e}")

def copy_text():
    """Try pyperclip first; if that fails, fall back to Tk’s own clipboard."""
    text = output_box.get("1.0", "end-1c").strip()
    if not text:
        return
    try:
        pyperclip.copy(text)
    except PyperclipException:                     # e.g. no xclip on Linux
        app.clipboard_clear()                      # Tk fallback: clear‑append
        app.clipboard_append(text)                 # ﻿:contentReference[oaicite:0]{index=0}
    messagebox.showinfo("Copied", "Translated text copied to clipboard!")

def speak_text():
    text = output_box.get("1.0", "end-1c").strip()
    if not text:
        return
    lang = target_lang.get()
    if lang not in gtts_langs:                      # avoid gTTS ValueError
        messagebox.showwarning(
            "TTS language unsupported",
            f"gTTS can’t speak “{lang}”. Falling back to English.",
        )
        lang = "en"
    try:
        tts = gTTS(text=text, lang=lang)
        tmp = "temp_tts.mp3"
        tts.save(tmp)
        playsound.playsound(tmp)
        os.remove(tmp)
    except Exception as e:
        messagebox.showerror("TTS Error", str(e))

# ── UI layout ─────────────────────────────────────────────────────────────────
title = ctk.CTkLabel(app, text="🌐 Language Translator Tool",
                     font=ctk.CTkFont(size=22, weight="bold"))
title.pack(pady=15)

frame_dd = ctk.CTkFrame(app)
frame_dd.pack(pady=10)

source_lang = ctk.StringVar(value="en")
target_lang = ctk.StringVar(value="fr")

ctk.CTkLabel(frame_dd, text="Source").grid(row=0, column=0, pady=(0, 4))
ctk.CTkLabel(frame_dd, text="Target").grid(row=0, column=1, pady=(0, 4))

ctk.CTkOptionMenu(frame_dd, values=language_codes, variable=source_lang, width=170)\
   .grid(row=1, column=0, padx=10)
ctk.CTkOptionMenu(frame_dd, values=language_codes, variable=target_lang, width=170)\
   .grid(row=1, column=1, padx=10)

ctk.CTkLabel(app, text="Enter Text").pack()
input_box = ctk.CTkTextbox(app, height=100, font=("Arial", 12))
input_box.pack(padx=20, pady=10, fill="x")

ctk.CTkButton(app, text="🔁 Translate", fg_color="#3498db", hover_color="#2980b9",
              command=translate_text).pack(pady=10)

ctk.CTkLabel(app, text="Translated Text").pack()
output_box = ctk.CTkTextbox(app, height=100, font=("Arial", 12))
output_box.pack(padx=20, pady=10, fill="x")

frame_btn = ctk.CTkFrame(app, fg_color="transparent")
frame_btn.pack(pady=15)

ctk.CTkButton(frame_btn, text="📋 Copy", fg_color="#2ecc71", hover_color="#27ae60",
              command=copy_text, width=120).grid(row=0, column=0, padx=20)
ctk.CTkButton(frame_btn, text="🔊 Speak", fg_color="#e67e22", hover_color="#d35400",
              command=speak_text, width=120).grid(row=0, column=1, padx=20)

app.mainloop()

