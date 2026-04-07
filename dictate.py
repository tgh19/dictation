#!/usr/bin/env python3
"""
Dictation & Translation — macOS menu bar app. Fully offline after first run.
Hold Right-Option key to record, release to transcribe/translate and type at cursor.

Features:
  - Menu bar icon with model selection (tiny/small/medium)
  - Dictation mode: speak in any language, type what you said
  - Translation mode: speak in one language, type in another
  - All processing on-device via mlx-whisper + opus-mt

Just run: python3 dictate.py
First run bootstraps a venv and downloads models automatically.
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# When running from .app bundle, store data in Application Support
if ".app/Contents" in SCRIPT_DIR:
    DATA_DIR = os.path.expanduser("~/Library/Application Support/Dictation")
    os.makedirs(DATA_DIR, exist_ok=True)
else:
    DATA_DIR = SCRIPT_DIR

VENV_DIR = os.path.join(DATA_DIR, ".venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")
MODELS_DIR = os.path.join(DATA_DIR, "models")

REQUIRED_PACKAGES = [
    "mlx-whisper",
    "sounddevice",
    "numpy",
    "pynput",
    "rumps",
    "ctranslate2",
    "transformers",
    "sentencepiece",
    "protobuf",
]


def bootstrap():
    """Ensure venv and packages are ready. Auto-fixes anything missing."""
    # If not running inside our venv, create it and re-exec
    if sys.prefix == sys.base_prefix or not sys.executable.startswith(VENV_DIR):
        if not os.path.exists(VENV_PYTHON):
            print("Creating virtual environment (first run)...")
            subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        print("Activating virtual environment...")
        os.execv(VENV_PYTHON, [VENV_PYTHON, __file__] + sys.argv[1:])

    # Inside venv — check packages
    missing = []
    for pkg in REQUIRED_PACKAGES:
        import_name = pkg.replace("-", "_")
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing packages: {', '.join(missing)}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip",
        ])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", *missing,
        ])
        os.execv(sys.executable, [sys.executable, __file__] + sys.argv[1:])

    # Pre-download default whisper model
    whisper_marker = os.path.join(MODELS_DIR, ".whisper-tiny-ready")
    if not os.path.exists(whisper_marker):
        print("Downloading whisper-tiny model (~75MB, first run only)...")
        import tempfile
        import wave
        import numpy as np
        import mlx_whisper

        silent = np.zeros(16000, dtype=np.float32)
        audio_int16 = (silent * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            with wave.open(f.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())
            mlx_whisper.transcribe(f.name, path_or_hf_repo="mlx-community/whisper-tiny")
        os.makedirs(MODELS_DIR, exist_ok=True)
        open(whisper_marker, "w").close()
        print("  Model cached.")


# ── Bootstrap before importing anything heavy ────────────────────────
bootstrap()

# ── Now safe to import ───────────────────────────────────────────────
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

import tempfile
import threading
import time
import wave

import numpy as np
import sounddevice as sd
import rumps
from pynput import keyboard as kb

SAMPLE_RATE = 16000
CHANNELS = 1
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

TYPE_LANGUAGES = [lang for lang in LANGUAGES if lang[1] is not None]

HALLUCINATIONS = {
    "thank you", "thank you for watching", "thanks for watching",
    "subscribe", "subscribe to my channel", "please subscribe",
    "like and subscribe", "thank you for listening",
    "the end", "you", "bye", "goodbye", "so",
    "thanks for watching!", "thank you.",
    "i'm going to go ahead and do that",
    "...", ".", "",
}


class DictationApp(rumps.App):
    def __init__(self):
        super().__init__("🎙", quit_button=None)

        self.mode = "dictation"
        self.whisper_model_key = "Tiny (fastest)"
        self.speak_lang = None  # None = auto-detect
        self.type_lang = "en"
        self.hotkey_name = "Right Option (⌥)"
        self.hotkey = HOTKEYS[self.hotkey_name]

        self._lock = threading.Lock()
        self.recording = False
        self.audio_chunks = []
        self.stream = None
        self.busy = False
        self.model_ready = False

        # Translation models loaded on demand
        self.translators = {}
        self.tokenizers = {}

        self._build_menu()

        threading.Thread(target=self._keyboard_listener, daemon=True).start()
        threading.Thread(target=self._load_whisper, daemon=True).start()

    # ── Menu ─────────────────────────────────────────────────────────

    def _build_menu(self):
        self.status_item = rumps.MenuItem("Status: Loading model...")

        self.dictation_item = rumps.MenuItem("Dictation Mode", callback=self._set_dictation)
        self.dictation_item.state = 1
        self.translate_item = rumps.MenuItem("Translation Mode", callback=self._set_translate)

        self.model_menu = rumps.MenuItem("Whisper Model")
        self.model_items = {}
        for name in WHISPER_MODELS:
            item = rumps.MenuItem(name, callback=self._select_model)
            item.state = 1 if name == self.whisper_model_key else 0
            self.model_items[name] = item
            self.model_menu.add(item)

        self.speak_menu = rumps.MenuItem("Speak Language")
        self.speak_items = {}
        for name, code in LANGUAGES:
            item = rumps.MenuItem(name, callback=self._select_speak_lang)
            item.state = 1 if code is None else 0
            self.speak_items[name] = item
            self.speak_menu.add(item)

        self.type_menu = rumps.MenuItem("Type Language")
        self.type_items = {}
        for name, code in TYPE_LANGUAGES:
            item = rumps.MenuItem(name, callback=self._select_type_lang)
            item.state = 1 if code == "en" else 0
            self.type_items[name] = item
            self.type_menu.add(item)

        self.hotkey_menu = rumps.MenuItem("Hotkey")
        self.hotkey_items = {}
        for name in HOTKEYS:
            item = rumps.MenuItem(name, callback=self._select_hotkey)
            item.state = 1 if name == self.hotkey_name else 0
            self.hotkey_items[name] = item
            self.hotkey_menu.add(item)

        self.menu = [
            self.status_item,
            None,
            self.dictation_item,
            self.translate_item,
            None,
            self.model_menu,
            self.speak_menu,
            self.type_menu,
            self.hotkey_menu,
            None,
            rumps.MenuItem("Quit", callback=lambda _: rumps.quit_application()),
        ]

    def _set_dictation(self, _):
        self.mode = "dictation"
        self.dictation_item.state = 1
        self.translate_item.state = 0
        if not self.recording and not self.busy:
            self.title = "🎙"

    def _set_translate(self, _):
        self.mode = "translate"
        self.dictation_item.state = 0
        self.translate_item.state = 1
        if not self.recording and not self.busy:
            self.title = "🌐"

    def _select_model(self, sender):
        old_key = self.whisper_model_key
        self.whisper_model_key = sender.title
        for item in self.model_items.values():
            item.state = 0
        sender.state = 1
        if old_key != sender.title:
            self.model_ready = False
            self.status_item.title = "Status: Loading model..."
            threading.Thread(target=self._load_whisper, daemon=True).start()

    def _select_hotkey(self, sender):
        for item in self.hotkey_items.values():
            item.state = 0
        sender.state = 1
        self.hotkey_name = sender.title
        self.hotkey = HOTKEYS[sender.title]
        self._update_status_ready()

    def _select_speak_lang(self, sender):
        for item in self.speak_items.values():
            item.state = 0
        sender.state = 1
        for name, code in LANGUAGES:
            if name == sender.title:
                self.speak_lang = code
                break

    def _select_type_lang(self, sender):
        for item in self.type_items.values():
            item.state = 0
        sender.state = 1
        for name, code in TYPE_LANGUAGES:
            if name == sender.title:
                self.type_lang = code
                break

    def _update_status_ready(self):
        self.status_item.title = f"Status: Ready — Hold {self.hotkey_name} to dictate"

    # ── Model Loading ────────────────────────────────────────────────

    def _load_whisper(self):
        model_repo = WHISPER_MODELS[self.whisper_model_key]
        self.status_item.title = f"Status: Loading {self.whisper_model_key}..."
        try:
            import mlx_whisper
            silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                _write_wav(f.name, silent)
                mlx_whisper.transcribe(f.name, path_or_hf_repo=model_repo)
            self.model_ready = True
            self._update_status_ready()
            print(f"Whisper model '{self.whisper_model_key}' loaded.")
        except Exception as e:
            self.status_item.title = "Status: Error loading model"
            print(f"Model load error: {e}")

    # ── Keyboard ─────────────────────────────────────────────────────

    def _keyboard_listener(self):
        def on_press(key):
            if key == self.hotkey:
                self._start_recording()

        def on_release(key):
            if key == self.hotkey and self.recording:
                threading.Thread(target=self._stop_and_process, daemon=True).start()

        with kb.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    # ── Recording ────────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
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

    def _start_recording(self):
        with self._lock:
            if self.recording or self.busy or not self.model_ready:
                return
            self.audio_chunks = []
            self.recording = True
            try:
                self.stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    callback=self._audio_callback,
                    blocksize=1024,
                )
                self.stream.start()
            except Exception as e:
                self.recording = False
                self._close_stream()
                print(f"Recording error: {e}")
                return

        self.title = "🔴"
        self.status_item.title = "Status: Recording..."
        _play_sound("Tink")

        def _timeout():
            time.sleep(MAX_RECORDING_SECS)
            with self._lock:
                if self.recording:
                    print(f"Recording hit {MAX_RECORDING_SECS}s limit.")
            self._stop_and_process()

        threading.Thread(target=_timeout, daemon=True).start()

    # ── Transcription & Translation ──────────────────────────────────

    def _stop_and_process(self):
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            self.busy = True
            self._close_stream()
            chunks = list(self.audio_chunks)
            self.audio_chunks.clear()

        self.title = "⏳"
        self.status_item.title = "Status: Processing..."
        _play_sound("Pop")

        try:
            if not chunks:
                return

            audio = np.concatenate(chunks, axis=0).flatten()
            duration = len(audio) / SAMPLE_RATE
            if duration < 0.3:
                return

            import mlx_whisper
            model_repo = WHISPER_MODELS[self.whisper_model_key]

            wav_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                _write_wav(wav_path, audio)

                t0 = time.time()

                if self.mode == "dictation":
                    kw = {"path_or_hf_repo": model_repo, "task": "transcribe"}
                    if self.speak_lang:
                        kw["language"] = self.speak_lang
                    result = mlx_whisper.transcribe(wav_path, **kw)
                    text = result.get("text", "").strip()

                else:  # translate mode
                    text = self._process_translation(wav_path, model_repo, mlx_whisper)

                elapsed = time.time() - t0
            finally:
                if wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)

            if text and text.lower().strip(".,!? ") not in HALLUCINATIONS:
                print(f"[{elapsed:.2f}s] {text}")
                _type_text(text)
                self._update_status_ready()
            else:
                self.status_item.title = "Status: No speech detected"

        except Exception as e:
            print(f"Error: {e}")
            self.status_item.title = "Status: Error"
        finally:
            with self._lock:
                self.busy = False
            self.title = "🌐" if self.mode == "translate" else "🎙"

    def _process_translation(self, wav_path, model_repo, mlx_whisper):
        """Handle translation mode logic. Returns final text to type."""
        speak = self.speak_lang
        target = self.type_lang

        if speak == target:
            # Same language — just transcribe
            kw = {"path_or_hf_repo": model_repo, "task": "transcribe"}
            if speak:
                kw["language"] = speak
            result = mlx_whisper.transcribe(wav_path, **kw)
            return result.get("text", "").strip()

        if target == "en":
            # Any → English: whisper's built-in translate
            kw = {"path_or_hf_repo": model_repo, "task": "translate"}
            if speak:
                kw["language"] = speak
            result = mlx_whisper.transcribe(wav_path, **kw)
            return result.get("text", "").strip()

        # Target is non-English — need opus-mt
        # Step 1: Get English text
        if speak == "en":
            result = mlx_whisper.transcribe(
                wav_path, path_or_hf_repo=model_repo,
                task="transcribe", language="en",
            )
        else:
            kw = {"path_or_hf_repo": model_repo, "task": "translate"}
            if speak:
                kw["language"] = speak
            result = mlx_whisper.transcribe(wav_path, **kw)

        english_text = result.get("text", "").strip()
        if not english_text or english_text.lower().strip(".,!? ") in HALLUCINATIONS:
            return ""

        # Step 2: Translate English → target via opus-mt
        self.status_item.title = "Status: Translating..."
        translated = self._translate_opus(english_text, "en", target)
        return translated if translated else english_text

    def _translate_opus(self, text, src, tgt):
        """Translate using Helsinki-NLP opus-mt models via ctranslate2."""
        pair = (src, tgt)

        if pair not in self.translators:
            try:
                self._load_opus_model(src, tgt)
            except Exception as e:
                import traceback; traceback.print_exc()
                rumps.notification(
                    "Dictation", "Translation model error",
                    f"Could not load {src}→{tgt} model. Check internet for first download.",
                )
                return None

        try:
            translator = self.translators[pair]
            tokenizer = self.tokenizers[pair]
            tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
            result = translator.translate_batch([tokens])
            return tokenizer.decode(
                tokenizer.convert_tokens_to_ids(result[0].hypotheses[0])
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            return None

    def _load_opus_model(self, src, tgt):
        """Download (if needed) and load an opus-mt translation model."""
        import ctranslate2
        from transformers import MarianTokenizer

        hf_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        os.makedirs(MODELS_DIR, exist_ok=True)

        model_dir = os.path.join(MODELS_DIR, f"opus-mt-{src}-{tgt}")
        tok_dir = os.path.join(MODELS_DIR, f"tokenizer-{src}-{tgt}")

        if not os.path.exists(model_dir):
            print(f"Downloading translation model {hf_name}...")
            converter = ctranslate2.converters.TransformersConverter(hf_name)
            converter.convert(model_dir, quantization="int8")

        if not os.path.exists(tok_dir):
            print(f"Caching tokenizer for {hf_name}...")
            tok = MarianTokenizer.from_pretrained(hf_name)
            tok.save_pretrained(tok_dir)

        pair = (src, tgt)
        self.translators[pair] = ctranslate2.Translator(model_dir, device="cpu")
        self.tokenizers[pair] = MarianTokenizer.from_pretrained(tok_dir)
        print(f"Translation model {src}→{tgt} loaded.")


# ── Helpers ──────────────────────────────────────────────────────────

def _write_wav(path, audio_data):
    audio_int16 = (audio_data * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())


def _type_text(text):
    """Type text at cursor position via clipboard paste (restores clipboard after)."""
    text = text.strip()
    if not text:
        return
    old_clip = subprocess.run(["pbpaste"], capture_output=True, timeout=5).stdout
    subprocess.run(["pbcopy"], input=text.encode(), check=True, timeout=5)
    subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to keystroke "v" using command down'],
        capture_output=True, timeout=5,
    )
    time.sleep(0.1)
    subprocess.run(["pbcopy"], input=old_clip, check=True, timeout=5)


def _play_sound(name):
    path = f"/System/Library/Sounds/{name}.aiff"
    subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    DictationApp().run()
