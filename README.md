# Dictation

**Offline voice-to-text and translation for macOS, right from your menu bar.**

![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.13-3776AB?style=flat&logo=python&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-optimized-FF6B00?style=flat&logo=apple&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

Hold a hotkey, speak, release — your words appear at the cursor. No cloud, no API keys, no internet required after first run. All processing happens on-device using [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) and [Opus-MT](https://github.com/Helsinki-NLP/Opus-MT).

---

## Features

| Feature | Details |
|---|---|
| **Dictation** | Hold hotkey to record, release to transcribe and type at cursor |
| **Translation** | Speak in one language, type in another (14 languages supported) |
| **Fully offline** | Models download once, then everything runs locally |
| **Apple Silicon** | Hardware-accelerated inference via MLX |
| **Menu bar app** | Unobtrusive — lives in your macOS menu bar |
| **Zero config** | Single file, self-bootstrapping — just run it |

## Supported Languages

Auto-detect, English, Russian, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Ukrainian, Arabic, and Hindi.

## Quick Start

**Requirements:** macOS with Apple Silicon (M1/M2/M3/M4), Python 3.13+

```bash
# Clone the repo
git clone https://github.com/tgh19/dictation.git
cd dictation

# Run it — first launch installs everything automatically
python3 dictate.py
```

That's it. On first run, Dictation will:
1. Create a virtual environment
2. Install all dependencies
3. Download the Whisper model (~75 MB for Tiny)

Subsequent launches start in seconds.

## Usage

### Dictation Mode (default)

1. Look for the **🎙** icon in your menu bar
2. **Hold** the hotkey (default: Right Option ⌥)
3. **Speak**
4. **Release** — text is typed at your cursor

### Translation Mode

1. Click the menu bar icon and select **Translation Mode**
2. Set your **Speak Language** (what you'll say) and **Type Language** (what gets typed)
3. Hold the hotkey, speak, release — translated text appears at your cursor

### Menu Bar Options

| Option | Description |
|---|---|
| **Dictation / Translation Mode** | Switch between modes |
| **Whisper Model** | Tiny (fastest), Small (balanced), Medium (accurate) |
| **Speak Language** | Language you're speaking (or auto-detect) |
| **Type Language** | Language to type in (translation mode) |
| **Hotkey** | Right Option, Right Command, F5, or F6 |

## How It Works

```
🎤 Microphone → WAV → MLX Whisper → Text → ⌨️ Paste at cursor
                                  ↘ Opus-MT → Translated text ↗
```

- **Speech-to-text:** [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) runs OpenAI's Whisper models natively on Apple Silicon via MLX
- **Translation:** [CTranslate2](https://github.com/OpenNMT/CTranslate2) runs Helsinki-NLP's Opus-MT models for language pairs not covered by Whisper's built-in translate task
- **Typing:** Uses the clipboard to paste text at cursor position, then restores the original clipboard contents
- **Hallucination filtering:** Common Whisper hallucinations ("thank you for watching", etc.) are automatically suppressed

## Permissions

macOS will ask for:
- **Microphone** — to record your voice
- **Accessibility** — to listen for hotkeys and simulate keystrokes

## Project Structure

```
dictation/
├── dictate.py      # The entire app — single file
├── models/         # Downloaded translation models (gitignored)
└── .venv/          # Auto-created virtual environment (gitignored)
```

## Troubleshooting

| Problem | Solution |
|---|---|
| Nothing happens when I press the hotkey | Grant Accessibility permission in System Settings → Privacy & Security |
| No audio captured | Grant Microphone permission for your terminal / Python |
| Model download fails | Check internet connection; models are cached after first download |
| App won't start (already running) | The app enforces a single instance — the old one is automatically stopped |

## Dependencies

All installed automatically in an isolated virtual environment:

- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) — Speech recognition
- [sounddevice](https://python-sounddevice.readthedocs.io/) — Audio capture
- [pynput](https://pynput.readthedocs.io/) — Global hotkey listener
- [rumps](https://github.com/jaredks/rumps) — macOS menu bar framework
- [ctranslate2](https://github.com/OpenNMT/CTranslate2) — Translation inference
- [transformers](https://huggingface.co/docs/transformers/) — Tokenizers for Opus-MT
- [numpy](https://numpy.org/) — Audio processing
