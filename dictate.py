#!/usr/bin/env python3
"""
Dictation & Translation — macOS menu bar app.
Fully offline after first run. All processing on-device via mlx-whisper + opus-mt.

Hold a hotkey to record, release to transcribe/translate and type at cursor.
Just run: python3 dictate.py — first run bootstraps everything automatically.
"""

import atexit
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = (
    os.path.expanduser("~/Library/Application Support/Dictation")
    if ".app/Contents" in SCRIPT_DIR
    else SCRIPT_DIR
)
VENV_DIR = os.path.join(DATA_DIR, ".venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")
MODELS_DIR = os.path.join(DATA_DIR, "models")

PACKAGES = {  # pip name → import name
    "mlx-whisper": "mlx_whisper",
    "sounddevice": "sounddevice",
    "numpy": "numpy",
    "pynput": "pynput",
    "rumps": "rumps",
    "ctranslate2": "ctranslate2",
    "transformers": "transformers",
    "sentencepiece": "sentencepiece",
    "protobuf": "google.protobuf",
}


# ── Bootstrap & singleton ───────────────────────────────────────────

def _bootstrap():
    """Create venv and install packages if needed, then re-exec."""
    if sys.prefix == sys.base_prefix:
        if not os.path.exists(VENV_PYTHON):
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        os.execv(VENV_PYTHON, [VENV_PYTHON, __file__] + sys.argv[1:])

    missing = [p for p, m in PACKAGES.items() if not _importable(m)]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        os.execv(sys.executable, [sys.executable, __file__] + sys.argv[1:])


def _importable(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _ensure_single_instance():
    """Kill any previous instance, write our PID."""
    os.makedirs(DATA_DIR, exist_ok=True)
    pid_file = os.path.join(DATA_DIR, ".dictation.pid")
    if os.path.exists(pid_file):
        try:
            old_pid = int(open(pid_file).read().strip())
            os.kill(old_pid, 9)
        except (ProcessLookupError, ValueError, PermissionError, OSError):
            pass
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(pid_file) and os.unlink(pid_file))


_bootstrap()
_ensure_single_instance()

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

# ── Imports (available after bootstrap) ──────────────────────────────

import tempfile
import threading
import time
import wave

import numpy as np
import rumps
import sounddevice as sd
from pynput import keyboard as kb

# ── Constants ────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
MAX_RECORDING_SECS = 120

HOTKEYS = {
    "Right Option (⌥)": kb.Key.alt_r,
    "Right Command (⌘)": kb.Key.cmd_r,
    "F5": kb.Key.f5,
    "F6": kb.Key.f6,
}

WHISPER_MODELS = {
    "Tiny (fastest)": "mlx-community/whisper-tiny",
    "Small (balanced)": "mlx-community/whisper-small-mlx",
    "Medium (accurate)": "mlx-community/whisper-medium-mlx",
}

LANGUAGES = [
    ("Auto-detect", None),
    ("English", "en"),
    ("Russian", "ru"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Ukrainian", "uk"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
]

HALLUCINATIONS = frozenset({
    "thank you", "thank you for watching", "thanks for watching",
    "subscribe", "subscribe to my channel", "please subscribe",
    "like and subscribe", "thank you for listening",
    "the end", "you", "bye", "goodbye", "so",
    "thanks for watching!", "thank you.",
    "i'm going to go ahead and do that",
    "...", ".", "",
})


# ── App ──────────────────────────────────────────────────────────────

class DictationApp(rumps.App):
    def __init__(self):
        super().__init__("🎙", quit_button=None)
        self.mode = "dictation"
        self.whisper_key = "Tiny (fastest)"
        self.speak_lang = None
        self.type_lang = "en"
        self.hotkey_name = "Right Option (⌥)"
        self.hotkey = HOTKEYS[self.hotkey_name]

        self._lock = threading.Lock()
        self.recording = False
        self.audio_chunks = []
        self.stream = None
        self.busy = False
        self.model_ready = False
        self.translators = {}
        self.tokenizers = {}

        self._build_menu()
        threading.Thread(target=self._listen_keys, daemon=True).start()
        threading.Thread(target=self._load_whisper, daemon=True).start()

    # ── Menu ─────────────────────────────────────────────────────────

    def _radio_menu(self, title, options, default, callback):
        """Build a submenu with radio-style checkmarks."""
        menu = rumps.MenuItem(title)
        items = {}
        for name in options:
            item = rumps.MenuItem(name, callback=callback)
            item.state = int(name == default)
            items[name] = item
            menu.add(item)
        return menu, items

    def _select_radio(self, items, sender):
        for item in items.values():
            item.state = 0
        sender.state = 1

    def _build_menu(self):
        self.status_item = rumps.MenuItem("Loading model...")

        self.mode_items = {
            "Dictation Mode": rumps.MenuItem("Dictation Mode", callback=self._set_mode),
            "Translation Mode": rumps.MenuItem("Translation Mode", callback=self._set_mode),
        }
        self.mode_items["Dictation Mode"].state = 1

        self.model_menu, self.model_items = self._radio_menu(
            "Whisper Model", WHISPER_MODELS, self.whisper_key, self._on_model)
        self.speak_menu, self.speak_items = self._radio_menu(
            "Speak Language", dict(LANGUAGES), "Auto-detect", self._on_speak_lang)
        type_langs = [(n, c) for n, c in LANGUAGES if c]
        self.type_menu, self.type_items = self._radio_menu(
            "Type Language", dict(type_langs), "English", self._on_type_lang)
        self.hotkey_menu, self.hotkey_items = self._radio_menu(
            "Hotkey", HOTKEYS, self.hotkey_name, self._on_hotkey)

        self.menu = [
            self.status_item, None,
            self.mode_items["Dictation Mode"],
            self.mode_items["Translation Mode"], None,
            self.model_menu, self.speak_menu,
            self.type_menu, self.hotkey_menu, None,
            rumps.MenuItem("Quit", callback=lambda _: rumps.quit_application()),
        ]

    # ── Menu callbacks ───────────────────────────────────────────────

    def _set_mode(self, sender):
        is_translate = sender.title == "Translation Mode"
        self.mode = "translate" if is_translate else "dictation"
        self.mode_items["Dictation Mode"].state = int(not is_translate)
        self.mode_items["Translation Mode"].state = int(is_translate)
        if not self.recording and not self.busy:
            self.title = "🌐" if is_translate else "🎙"

    def _on_model(self, sender):
        if sender.title == self.whisper_key:
            return
        self._select_radio(self.model_items, sender)
        self.whisper_key = sender.title
        self.model_ready = False
        self.status_item.title = "Loading model..."
        threading.Thread(target=self._load_whisper, daemon=True).start()

    def _on_speak_lang(self, sender):
        self._select_radio(self.speak_items, sender)
        self.speak_lang = dict(LANGUAGES)[sender.title]

    def _on_type_lang(self, sender):
        self._select_radio(self.type_items, sender)
        self.type_lang = dict((n, c) for n, c in LANGUAGES if c)[sender.title]

    def _on_hotkey(self, sender):
        self._select_radio(self.hotkey_items, sender)
        self.hotkey_name = sender.title
        self.hotkey = HOTKEYS[sender.title]
        if self.model_ready:
            self._set_ready()

    def _set_ready(self):
        self.status_item.title = f"Ready — Hold {self.hotkey_name}"

    # ── Whisper ──────────────────────────────────────────────────────

    def _load_whisper(self):
        self.status_item.title = f"Loading {self.whisper_key}..."
        try:
            import mlx_whisper  # noqa: F401
            self.model_ready = True
            self._set_ready()
        except Exception as e:
            self.status_item.title = "Error loading model"

    # ── Keyboard ─────────────────────────────────────────────────────

    def _listen_keys(self):
        def on_press(key):
            if key == self.hotkey:
                self._start_recording()

        def on_release(key):
            if key == self.hotkey and self.recording:
                threading.Thread(target=self._process, daemon=True).start()

        with kb.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    # ── Recording ────────────────────────────────────────────────────

    def _start_recording(self):
        with self._lock:
            if self.recording or self.busy or not self.model_ready:
                return
            self.audio_chunks = []
            self.recording = True
            try:
                self.stream = sd.InputStream(
                    samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    callback=self._audio_cb, blocksize=1024,
                )
                self.stream.start()
            except Exception:
                self.recording = False
                self._close_stream()
                return

        self.title = "🔴"
        self.status_item.title = "Recording..."
        _play_sound("Tink")

        def timeout():
            time.sleep(MAX_RECORDING_SECS)
            if self.recording:
                self._process()

        threading.Thread(target=timeout, daemon=True).start()

    def _audio_cb(self, indata, frames, time_info, status):
        if self.recording:
            self.audio_chunks.append(indata.copy())

    def _close_stream(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    # ── Processing ───────────────────────────────────────────────────

    def _process(self):
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            self.busy = True
            self._close_stream()
            chunks = list(self.audio_chunks)
            self.audio_chunks.clear()

        self.title = "⏳"
        self.status_item.title = "Processing..."
        _play_sound("Pop")

        try:
            if not chunks:
                return
            audio = np.concatenate(chunks, axis=0).flatten()
            if len(audio) / SAMPLE_RATE < 0.3:
                return

            text = self._transcribe(audio)

            if text and text.lower().strip(".,!? ") not in HALLUCINATIONS:
                _type_text(text)
                self._set_ready()
            else:
                self.status_item.title = "No speech detected"
        except Exception as e:
            self.status_item.title = "Error"
        finally:
            with self._lock:
                self.busy = False
            self.title = "🌐" if self.mode == "translate" else "🎙"

    def _transcribe(self, audio):
        """Transcribe or translate audio. Returns text to type."""
        import mlx_whisper
        repo = WHISPER_MODELS[self.whisper_key]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            _write_wav(wav_path, audio)

            if self.mode == "dictation":
                return self._whisper(mlx_whisper, wav_path, repo, "transcribe")

            speak, target = self.speak_lang, self.type_lang
            if speak == target:
                return self._whisper(mlx_whisper, wav_path, repo, "transcribe")
            if target == "en":
                return self._whisper(mlx_whisper, wav_path, repo, "translate")

            # Non-English target: whisper → English → opus-mt → target
            task = "transcribe" if speak == "en" else "translate"
            english = self._whisper(mlx_whisper, wav_path, repo, task,
                                    lang="en" if speak == "en" else speak)
            if not english or english.lower().strip(".,!? ") in HALLUCINATIONS:
                return ""
            self.status_item.title = "Translating..."
            return self._opus_translate(english, "en", target) or english
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def _whisper(self, mlx_whisper, wav_path, repo, task, lang=None):
        kw = {"path_or_hf_repo": repo, "task": task}
        if lang is None:
            lang = self.speak_lang
        if lang:
            kw["language"] = lang
        return mlx_whisper.transcribe(wav_path, **kw).get("text", "").strip()

    # ── Translation (opus-mt) ────────────────────────────────────────

    def _opus_translate(self, text, src, tgt):
        pair = (src, tgt)
        if pair not in self.translators:
            try:
                self._load_opus(src, tgt)
            except Exception:
                rumps.notification("Dictation", "Translation error",
                                   f"Could not load {src}→{tgt} model.")
                return None

        tokenizer = self.tokenizers[pair]
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
        result = self.translators[pair].translate_batch([tokens])
        return tokenizer.decode(
            tokenizer.convert_tokens_to_ids(result[0].hypotheses[0]))

    def _load_opus(self, src, tgt):
        import ctranslate2
        from transformers import MarianTokenizer

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_dir = os.path.join(MODELS_DIR, f"opus-mt-{src}-{tgt}")
        tok_dir = os.path.join(MODELS_DIR, f"tokenizer-{src}-{tgt}")
        hf_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"

        if not os.path.exists(model_dir):
            ctranslate2.converters.TransformersConverter(hf_name).convert(
                model_dir, quantization="int8")
        if not os.path.exists(tok_dir):
            MarianTokenizer.from_pretrained(hf_name).save_pretrained(tok_dir)

        pair = (src, tgt)
        self.translators[pair] = ctranslate2.Translator(model_dir, device="cpu")
        self.tokenizers[pair] = MarianTokenizer.from_pretrained(tok_dir)


# ── Helpers ──────────────────────────────────────────────────────────

def _write_wav(path, audio):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())


def _type_text(text):
    text = text.strip()
    if not text:
        return
    old = subprocess.run(["pbpaste"], capture_output=True, timeout=5).stdout
    subprocess.run(["pbcopy"], input=text.encode(), check=True, timeout=5)
    subprocess.run(["osascript", "-e",
                     'tell application "System Events" to keystroke "v" using command down'],
                    capture_output=True, timeout=5)
    time.sleep(0.1)
    subprocess.run(["pbcopy"], input=old, check=True, timeout=5)


def _play_sound(name):
    subprocess.Popen(["afplay", f"/System/Library/Sounds/{name}.aiff"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    DictationApp().run()
