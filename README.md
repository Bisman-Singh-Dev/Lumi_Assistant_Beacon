# Lumi - HealthThread Beacon assistant

A voice and camera health assistant in pure Python, for a Windows laptop or a Raspberry Pi.

- **Diagnosis Lumi**: say "Wake up Lumi" (or type) and describe your symptoms. Lumi asks a follow-up question if needed, then shows and speaks a structured answer: possible problem, AI assessment, next steps, precautions, specialist and urgency. It then goes back to the home screen.
- **Camera Lumi**: say "Camera Lumi", then give a thumbs-up or press Capture (or upload a photo). The photo is analysed straight away by a vision model. Skin photos get a dermatological assessment.
- **Emergency Lumi**: say "Emergency Lumi" (no wake-up needed). The emergency screen and camera open, verified emergency numbers are shown (India: 112, 108), and urgent guidance comes first. The screen stays until you press "I'm safe".

Lumi gives AI health information, not a medical diagnosis.

## Setup

```bash
pip install -r requirements.txt
cp lumi.env.example lumi.env   # then add your Deepgram and Groq keys
python lumi.py --windowed
```

On a Raspberry Pi, also install: `sudo apt install -y libportaudio2 espeak-ng python3-tk`

- Check everything works: `python lumi.py --selftest`
- List microphones: `python lumi.py --list-mics`

## Voice commands

Wake up Lumi | Camera Lumi | Emergency Lumi | Stop Lumi | Repeat | Close camera | Go home | I'm safe

## Keys

W talk, T type, C camera, U upload, E emergency, R repeat, S stop, Space capture, X close camera, H home, V last result, Y first-aid video, Esc quit

## Services used

- Deepgram: speech-to-text (nova-3) and voice (Aura-2)
- Groq: gpt-oss-120b (symptoms) and qwen3.8-27b (photos)
- MediaPipe: thumbs-up gesture recognition (the model downloads on first use)
