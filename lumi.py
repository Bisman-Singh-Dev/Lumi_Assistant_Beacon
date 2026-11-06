#!/usr/bin/env python3
"""
Lumi - HealthThread Beacon medical assistant (pure Python: Windows laptop + Raspberry Pi)

Home screen -> three modes that share one result layout
  Diagnosis Lumi : "Wake up Lumi" (or Type) -> symptoms -> follow-up questions
                   -> 6-card assessment -> spoken -> back to home automatically
  Camera Lumi    : "Camera Lumi" / Upload -> photo -> Retake or Analyze
                   -> skin photos get a dermatological assessment
                   -> spoken -> back to home automatically
  Emergency Lumi : "Emergency Lumi" (no wake-up) -> emergency screen + camera,
                   voice / typing / photo -> immediate guidance -> stays until
                   you press "I'm safe - exit"

Voice: Wake up Lumi | Camera Lumi | Emergency Lumi | Stop Lumi | Repeat |
       Retake | Analyze | Close camera | Go home | I'm safe
Keys:  W talk  T type  C camera  U upload  E emergency  R repeat  S stop
       Space capture  Enter analyze  Backspace retake  X close camera
       H home  V last result  Y first-aid video  Esc quit

Setup: keys in lumi.env (DEEPGRAM_API_KEY, DEEPGRAM_API_KEY_2, GROQ_API_KEY),
       optional LUMI_COUNTRY (otherwise detected from the time zone).
  pip install pygame-ce sounddevice numpy requests opencv-python mediapipe
  (Raspberry Pi: also  sudo apt install -y libportaudio2 espeak-ng python3-tk)
Run:   python Lumi.py.py --windowed      python Lumi.py.py --selftest
"""

import argparse
import base64
import io
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import threading
import time
import wave
import webbrowser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import requests
import pygame

try:
    import sounddevice as sd
    SD_IMPORT_ERROR = None
except Exception as _e:
    sd, SD_IMPORT_ERROR = None, _e

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MP_IMPORT_ERROR = None
except Exception as _e:
    mp, MP_IMPORT_ERROR = None, _e


# ============================================================================
# CONFIG (keys come from lumi.env / environment variables, never the code)
# ============================================================================
HERE = os.path.dirname(os.path.abspath(__file__))


def load_env_file(path):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not os.environ.get(k):
                os.environ[k] = v


load_env_file(os.path.join(HERE, "lumi.env"))

DEEPGRAM_API_KEYS = list(dict.fromkeys(
    k for k in (os.getenv("DEEPGRAM_API_KEY", ""), os.getenv("DEEPGRAM_API_KEY_2", "")) if k))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

STT_MODEL = "nova-3"
TTS_MODELS = [os.getenv("LUMI_TTS_MODEL", "aura-2-thalia-en"), "aura-asteria-en"]
TEXT_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
VISION_MODEL = os.getenv("LUMI_VISION_MODEL", "qwen/qwen3.8-27b")  # verified: accepts images
CAMERA_INDEX = int(os.getenv("LUMI_CAMERA", "0"))
MIC_MIN_RMS = float(os.getenv("LUMI_MIC_MIN_RMS", "150"))
THUMB_HOLD_SECONDS = 0.6
THUMB_MIN_SCORE = 0.6
REVIEW_AUTO_SECONDS = 4.0  # photo preview: analyze automatically unless "Retake"
EMERGENCY_VIDEO = "https://youtu.be/AhANvBB9hz0?si=mRPZZFV2MgIbLO_l"
GESTURE_MODEL_PATH = os.path.join(HERE, "gesture_recognizer.task")
GESTURE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                     "gesture_recognizer/float16/1/gesture_recognizer.task")
HISTORY_FILE = os.path.join(HERE, "lumi_history.json")
PHOTO_DIR = os.path.join(HERE, "lumi_photos")
DEBUG = os.getenv("LUMI_DEBUG") == "1"

# Verified emergency numbers. Lumi never asks the AI for phone numbers.
EMERGENCY_NUMBERS = {
    "India": [("112", "Emergency response (police, fire, ambulance)"),
              ("108", "Ambulance, where available in your state")],
    "United States": [("911", "Police, fire and ambulance")],
    "Canada": [("911", "Police, fire and ambulance")],
    "United Kingdom": [("999", "Police, fire and ambulance"), ("112", "Also works for emergencies")],
    "Australia": [("000", "Police, fire and ambulance"), ("112", "Emergency calls from mobiles")],
    "European Union": [("112", "European emergency number")],
}
SPOKEN_PURPOSE = {"112": "emergency response", "108": "an ambulance", "911": "emergencies",
                  "999": "emergencies", "000": "emergencies"}


def detect_country():
    """LUMI_COUNTRY wins; otherwise the system time zone. Never guesses beyond that."""
    env = os.getenv("LUMI_COUNTRY", "").strip()
    if env:
        return env, "set in lumi.env"
    parts = list(time.tzname)
    for p in ("/etc/timezone",):
        try:
            parts.append(open(p).read())
        except Exception:
            pass
    try:
        parts.append(os.path.realpath("/etc/localtime"))
    except Exception:
        pass
    try:
        parts.append(str(datetime.now().astimezone().tzinfo))
    except Exception:
        pass
    s = " ".join(parts)
    rules = [("India", ("Kolkata", "Calcutta", "India")),
             ("United Kingdom", ("Europe/London", "GMT Standard Time", "British")),
             ("Australia", ("Australia/", "AUS Eastern", "AUS Central")),
             ("Canada", ("America/Toronto", "America/Vancouver", "America/Edmonton", "America/Winnipeg",
                         "America/Halifax", "America/Montreal", "America/Regina", "America/St_Johns")),
             ("United States", ("America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
                                "America/Phoenix", "America/Anchorage", "Pacific/Honolulu")),
             ("European Union", ("Europe/",))]
    for country, keys in rules:
        if any(k in s for k in keys):
            return country, "from your time zone"
    return None, "unknown"


COUNTRY, COUNTRY_SOURCE = detect_country()
CONTACTS = EMERGENCY_NUMBERS.get(COUNTRY, []) if COUNTRY else []
PRIMARY_NUMBER = CONTACTS[0][0] if CONTACTS else None


def primary_spoken():
    return PRIMARY_NUMBER or "your local emergency number"


# ============================================================================
# PROMPTS
# ============================================================================
BASE_RULES = """You are Lumi, a careful, warm health assistant inside HealthThread Beacon. You are not a doctor.
- Use simple everyday words. Briefly explain any medical word you must use.
- Never present a possibility as a confirmed diagnosis. Use "possibly" or "may be".
- Never name prescription medicines and never give doses.
- If anything could be serious, set urgency to Prompt or Emergency and do not suggest waiting at home. For Emergency the specialist is "Emergency department".
- Specialists: Dermatologist (skin), Cardiologist (heart), Neurologist (brain and nerves), Orthopedic specialist (bones and joints), Ophthalmologist (eyes), Dentist (teeth), Gastroenterologist (digestion), General physician (general or unclear), qualified mental health professional (mental health).
- Never include phone numbers. The app adds verified emergency numbers itself.
- Plain text in every string: no markdown, asterisks or emojis. Keep strings short and easy to say aloud."""

RESULT_SCHEMA = """Return ONLY one JSON object with these keys:
{
 "concern": "the identified problem in plain words, as a possibility",
 "assessment": "one to three simple sentences explaining what this may mean",
 "possible_conditions": [{"name": "...", "why": "which symptoms or visible signs point to it"}],
 "confidence": "Low or Moderate or High",
 "steps": [{"title": "short title", "description": "what to do"}],
 "precautions": ["things to avoid until a doctor has checked"],
 "specialist": {"name": "...", "reason": "why this specialist"},
 "urgency": "Routine or Prompt or Emergency",
 "urgency_reason": "one sentence on why",
 "first_instruction": "the single most important thing to do right now",
 "call_now_if": "warning signs that mean calling emergency services immediately"%s
}"""
IMAGE_KEYS = """,
 "visible_features": ["only things actually visible in the photo"],
 "limitations": "what cannot be known from this photo\""""

SYMPTOM_TASK = """TASK: symptom-based assessment. Give 3 to 5 steps (what to do), 2 to 4 precautions (what to avoid), and up to 3 possible conditions."""

EMERGENCY_TASK = """TASK: EMERGENCY. The user may be in danger right now. Focus on immediate safety.
- concern: the reported emergency. steps: 3 to 5 immediate, safe actions, most important first. precautions: what NOT to do.
- specialist: the emergency service to use (usually "Ambulance" or "Emergency department").
- If it could be life threatening, urgency is Emergency.
- Never say you have called anyone or sent an alert."""

DERM_TASK = """TASK: DERMATOLOGICAL ASSESSMENT of a skin photo.
- visible_features: list only what you can actually see: redness, swelling, irritation, discolouration, rash, spots, bumps, scaling, blisters, inflammation, and where and how widespread.
- possible_conditions: up to 3, each with "why" naming the visible features that support it.
- If no meaningful skin problem is visible, set concern to "No clear skin problem visible", leave possible_conditions empty and urgency Routine.
- If the photo is blurry, dark or far away, say so and set confidence Low. Do not guess.
- specialist is usually Dermatologist. Use urgency Prompt for signs that could be serious (spreading fast, very painful, blistering, signs of infection, a changing mole) and Emergency for swelling of the face or lips or trouble breathing.
- limitations: explain that a photo alone cannot reliably give a skin diagnosis."""

PHOTO_TASK = """TASK: VISUAL ASSESSMENT of a photo that shows a possible health issue that is not mainly about skin.
- visible_features: only what is actually visible. A photo cannot show blood pressure, blood sugar, oxygen level or internal disease and cannot replace an examination.
- limitations: what cannot be known from this photo."""

EMERGENCY_PHOTO_TASK = EMERGENCY_TASK + """
- This is a photo. Describe in visible_features only what is actually visible, and say in limitations what cannot be known from it."""

CLASSIFY_PROMPT = """Classify this photo for a health assistant. Return ONLY JSON:
{"image_type": "skin" or "other_health" or "not_health" or "unclear", "reason": "one short, plain sentence"}
- skin: skin is shown close enough to judge how it looks (rash, spots, redness, a mole, a burn, or plain skin).
- other_health: a visible health issue that is not mainly about skin (an eye, a swollen joint, an open wound, a medicine).
- not_health: no body part or health issue is shown (objects, rooms, pets, screens), or a person is shown from a distance with nothing health related visible.
- unclear: too blurry, dark or far away to judge."""

TRIAGE_PROMPT = """You help a health assistant decide the next step after the user speaks.
Return ONLY JSON: {"kind": "health" or "other", "reply": "...", "urgent": true or false, "questions": ["..."]}
- kind is "other" if the user is not describing a health problem (for example maths, chit-chat, general questions). Then put a short, friendly one or two sentence answer in reply, with no markdown, and leave questions empty.
- For kind "health", reply is an empty string.
- urgent is true for possible emergencies: chest pain, trouble breathing, severe bleeding, stroke signs, fainting, seizures, severe allergic reaction, thoughts of self harm, serious injury. If urgent, questions must be empty.
- Otherwise ask at most two short questions, only if the answer would really change the advice (how long, getting worse, other symptoms, treatment already tried). If the description already covers this, return an empty list."""

RED_FLAGS = ("chest pain", "can't breathe", "cannot breathe", "cant breathe", "difficulty breathing",
             "trouble breathing", "not breathing", "unconscious", "fainted", "passed out", "severe bleeding",
             "bleeding heavily", "won't stop bleeding", "stroke", "face drooping", "slurred speech", "seizure",
             "suicid", "kill myself", "overdose", "poison", "choking", "heart attack", "severe burn",
             "head injury", "throat is swelling", "anaphylaxis")


def log(*a):
    print("[Lumi]", *a, flush=True)


class ApiError(Exception):
    pass


# ============================================================================
# TEXT HELPERS + COMMANDS
# ============================================================================
def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def clean_for_speech(text):
    text = strip_thinking(text)
    text = text.replace("*", "").replace("#", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[_~`]", "", text)
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    return re.sub(r"\s+", " ", text).strip()


def split_for_speech(text, max_len=230):
    chunks, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        while len(s) > max_len:
            cut = max(s.rfind(",", 0, max_len), s.rfind(" ", 0, max_len))
            cut = cut if cut > 40 else max_len
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s']", " ", text.lower())).replace("'", "").strip()


LUMI_NAMES = {"lumi", "loomy", "lumie", "lumee", "lumy", "loomi", "lumia", "looming", "lumine", "luma"}


def has_lumi(n):
    return "lumi" in n or any(w in LUMI_NAMES for w in n.split())


EMERGENCY_PHRASES = ("emergency lumi", "lumi emergency", "emergency mode", "emergency room",
                     "call ambulance", "call an ambulance", "critical emergency")


def detect_command(n):
    """One command name or None. Order matters."""
    words = n.split()
    if not words:
        return None
    if any(p in n for p in ("im safe", "i am safe", "i got help", "i have help", "help has arrived",
                            "exit emergency", "end emergency", "cancel emergency", "stop emergency")):
        return "exit_emergency"
    if any(p in n for p in EMERGENCY_PHRASES) or n == "emergency" or (words[0] == "emergency" and has_lumi(n)):
        return "emergency"
    if any(p in n for p in ("close camera", "close the camera", "stop camera", "stop the camera",
                            "exit camera", "camera off", "turn off the camera", "turn off camera")):
        return "close_camera"
    if n in ("stop", "stop lumi", "lumi stop", "stop talking", "be quiet", "quiet", "stop it", "ok stop",
             "okay stop", "stop please") or n.startswith("stop lumi"):
        return "stop"
    if n in ("repeat", "repeat that", "repeat please", "say that again", "lumi repeat",
             "repeat lumi", "can you repeat", "can you repeat that", "please repeat"):
        return "repeat"
    if n in ("retake", "retake photo", "retake the photo", "take another", "take another one", "another photo",
             "try again", "redo", "retake it"):
        return "retake"
    if n in ("analyze", "analyse", "analyze it", "analyse it", "analyze photo", "analyse photo", "use photo",
             "use this photo", "use this one", "looks good", "yes", "yes please", "confirm", "go ahead", "ok", "okay"):
        return "confirm"
    if n in ("home", "go home", "go back", "go back home", "back to home", "main screen", "lumi home",
             "home screen", "back"):
        return "home"
    if any(p in n for p in ("camera lumi", "lumi camera", "open camera", "open the camera", "start camera",
                            "start the camera", "take a photo", "take photo", "take a picture")) \
            or ("camera" in words and len(words) <= 3):
        return "camera"
    if any(p in n for p in ("wake up", "wake lumi", "hey lumi", "hello lumi", "hi lumi", "okay lumi", "ok lumi")) \
            or (has_lumi(n) and len(words) <= 3):
        return "wake"
    return None


def strip_emergency_words(text):
    t = re.sub(r"(?i)\b(emergency|lumi|loomy|lumie)\b[,.!]?", " ", text)
    return re.sub(r"\s+", " ", t).strip(" ,.!")


def has_red_flag(text):
    t = text.lower()
    return any(f in t for f in RED_FLAGS)


def parse_json_loose(text):
    text = strip_thinking(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ApiError("The AI did not return a structured answer.")


def save_history(entry):
    data = []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        pass
    entry["time"] = datetime.now().isoformat(timespec="seconds")
    data.append(entry)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log("Could not save history:", e)


# ============================================================================
# AUDIO HELPERS
# ============================================================================
def to_wav(pcm, sr, boost=True):
    pcm = pcm.astype(np.float32)
    peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
    if boost and 0 < peak < 16000:
        pcm *= min(8.0, 16000.0 / peak)
    pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def read_wav(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16)[::ch] if sw == 2 else np.zeros(1, np.int16)
    return x, sr


def wav_envelope(wav_bytes, fps=30):
    try:
        x, sr = read_wav(wav_bytes)
        x = x.astype(np.float32)
        hop = max(1, sr // fps)
        k = len(x) // hop
        if k == 0:
            return None
        env = np.sqrt(np.mean(x[:k * hop].reshape(k, hop) ** 2, axis=1))
        return env / (env.max() + 1e-6)
    except Exception:
        return None


def chime_wav():
    sr, parts = 24000, []
    for f, d in ((660, 0.11), (990, 0.16)):
        t = np.arange(int(sr * d)) / sr
        env = np.minimum(1, t / 0.01) * np.exp(-t * 9)
        parts.append(np.sin(2 * np.pi * f * t) * env + 0.3 * np.sin(4 * np.pi * f * t) * env)
    return to_wav(np.concatenate(parts) * 9000, sr, boost=False)


# ============================================================================
# CLOUD APIs
# ============================================================================
def speech_to_text(wav):
    """Returns (text, provider, error). Deepgram (every key), then Groq Whisper."""
    errors = []
    for key in DEEPGRAM_API_KEYS:
        params = {"model": STT_MODEL, "smart_format": "true", "language": "en", "keyterm": "Lumi"}
        for _ in range(2):
            try:
                r = requests.post("https://api.deepgram.com/v1/listen", params=params, data=wav, timeout=20,
                                  headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"})
                if r.ok:
                    alt = r.json()["results"]["channels"][0]["alternatives"][0]
                    return alt.get("transcript", "").strip(), "deepgram", None
                if r.status_code == 400 and "keyterm" in params:
                    params.pop("keyterm")
                    continue
                errors.append(f"Deepgram {r.status_code}")
            except requests.RequestException as e:
                errors.append(f"Deepgram network error ({type(e).__name__})")
            break
    if GROQ_API_KEY:
        try:
            r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions", timeout=30,
                              headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                              files={"file": ("speech.wav", wav, "audio/wav")},
                              data={"model": "whisper-large-v3-turbo", "response_format": "json", "language": "en"})
            if r.ok:
                return r.json().get("text", "").strip(), "groq-whisper", None
            errors.append(f"Groq Whisper {r.status_code}")
        except requests.RequestException as e:
            errors.append(f"Groq network error ({type(e).__name__})")
    return "", None, ", ".join(errors) or "no speech-to-text key configured"


def text_to_speech(text):
    """Returns (wav_bytes, error)."""
    if not DEEPGRAM_API_KEYS:
        return None, "no Deepgram key in lumi.env"
    err = "unknown error"
    for key in DEEPGRAM_API_KEYS:
        for model in TTS_MODELS:
            try:
                r = requests.post("https://api.deepgram.com/v1/speak", timeout=30,
                                  params={"model": model, "encoding": "linear16", "sample_rate": "24000",
                                          "container": "wav"},
                                  headers={"Authorization": f"Token {key}", "Content-Type": "application/json"},
                                  json={"text": text[:1900]})
                if r.ok and len(r.content) > 100:
                    return r.content, None
                err = f"Deepgram {r.status_code}: {r.text[:120]}"
                if r.status_code in (401, 403):
                    break
            except requests.RequestException as e:
                err = f"network error ({type(e).__name__})"
    return None, err


def groq_chat(messages, model, max_tokens=3000, temperature=0.2, json_mode=False, extra=None):
    if not GROQ_API_KEY:
        raise ApiError("No Groq API key. Add GROQ_API_KEY to lumi.env.")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    body.update(extra or {})
    for attempt in range(3):
        last = attempt == 2
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=body, timeout=60,
                              headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
        except requests.ConnectionError:
            if last:
                raise ApiError("No internet connection. Could not reach Groq.")
            time.sleep(1.5)
            continue
        except requests.Timeout:
            if last:
                raise ApiError("Groq took too long to answer.")
            continue
        if r.status_code == 429:
            if last:
                raise ApiError("Groq rate limit reached. Please wait a minute and try again.")
            try:
                wait = float(r.headers.get("retry-after", "3"))
            except ValueError:
                wait = 3.0
            log(f"Groq rate limit, waiting {min(max(wait, 1.0), 15.0):.0f}s")
            time.sleep(min(max(wait, 1.0), 15.0))
            continue
        if r.status_code >= 500 and not last:
            time.sleep(1.5)
            continue
        if r.status_code in (401, 403):
            raise ApiError("Groq rejected the API key. Check GROQ_API_KEY in lumi.env.")
        if not r.ok:
            if "json_validate_failed" in r.text:
                raise ApiError("JSON_FAILED")
            try:
                msg = r.json().get("error", {}).get("message", r.text)
            except Exception:
                msg = r.text
            raise ApiError(f"Groq error {r.status_code}: {msg[:140]}")
        try:
            text = r.json()["choices"][0]["message"].get("content") or ""
        except Exception:
            raise ApiError("Groq sent back an invalid response.")
        text = strip_thinking(text)
        if not text:
            raise ApiError("Groq returned an empty answer.")
        return text
    raise ApiError("Groq could not be reached.")


def groq_json(messages, model, max_tokens=3000, extra=None):
    try:
        return parse_json_loose(groq_chat(messages, model, max_tokens, json_mode=True, extra=extra))
    except ApiError as e:
        if str(e) not in ("JSON_FAILED", "The AI did not return a structured answer."):
            raise
        log(f"{model}: strict JSON failed, retrying in plain mode")
        return parse_json_loose(groq_chat(messages, model, max_tokens, json_mode=False, extra=extra))


def _clean(s):
    return re.sub(r"\s+", " ", str(s or "").replace("*", "").replace("#", "")).strip()


def _strlist(v, n=5):
    return [_clean(x) for x in (v if isinstance(v, list) else [v] if v else []) if _clean(x)][:n]


def validate_result(d, kind):
    """Checks the AI's JSON, fills safe defaults, adds verified contacts. Raises ApiError if unusable."""
    if not isinstance(d, dict):
        raise ApiError("The AI answer was not in the expected format.")
    concern = _clean(d.get("concern") or d.get("possible_condition"))
    steps = []
    for s in (d.get("steps") if isinstance(d.get("steps"), list) else [])[:6]:
        title, desc = (_clean(s.get("title")), _clean(s.get("description"))) if isinstance(s, dict) else ("", _clean(s))
        if title or desc:
            steps.append({"step": len(steps) + 1, "title": title, "description": desc})
    assessment = _clean(d.get("assessment") or d.get("why"))
    if not concern or not (steps or assessment):
        raise ApiError("The AI answer was incomplete. Please try again.")
    conds = []
    for c in (d.get("possible_conditions") if isinstance(d.get("possible_conditions"), list) else [])[:3]:
        if isinstance(c, dict) and _clean(c.get("name")):
            conds.append({"name": _clean(c.get("name")), "why": _clean(c.get("why"))})
        elif isinstance(c, str) and _clean(c):
            conds.append({"name": _clean(c), "why": ""})
    conf = next((c for c in ("High", "Moderate", "Low") if c.lower() in str(d.get("confidence", "")).lower()), "Low")
    raw_u = str(d.get("urgency") or (d.get("emergency") or {}).get("urgency", "") if isinstance(d.get("emergency"), dict)
                else d.get("urgency", "")).lower()
    urgency = "Emergency" if "emergency" in raw_u else "Prompt" if ("prompt" in raw_u or "urgent" in raw_u) else "Routine"
    sp = d.get("specialist") if isinstance(d.get("specialist"), dict) else {}
    spec = {"name": _clean(sp.get("name")) or "General physician", "reason": _clean(sp.get("reason"))}
    if urgency == "Emergency" and not any(w in spec["name"].lower() for w in ("emergency", "ambulance")):
        spec = {"name": "Emergency department",
                "reason": f"This may be an emergency, so get care now. Afterwards: {spec['name']}."}
    return {
        "kind": kind, "concern": concern, "assessment": assessment, "possible_conditions": conds,
        "visible_features": _strlist(d.get("visible_features"), 6), "confidence": conf, "steps": steps,
        "precautions": _strlist(d.get("precautions"), 5), "specialist": spec, "urgency": urgency,
        "urgency_reason": _clean(d.get("urgency_reason")), "first_instruction": _clean(d.get("first_instruction")),
        "call_now_if": _clean(d.get("call_now_if")), "limitations": _clean(d.get("limitations")),
        "country": COUNTRY, "country_source": COUNTRY_SOURCE, "contacts": CONTACTS, "photo": None,
        "disclaimer": "AI-generated health information, not a confirmed medical diagnosis.",
    }


def triage(text):
    d = groq_json([{"role": "system", "content": TRIAGE_PROMPT}, {"role": "user", "content": text}],
                  TEXT_MODELS[0], max_tokens=1200, extra={"reasoning_effort": "low"})
    if not isinstance(d, dict):
        return {"kind": "health", "reply": "", "urgent": False, "questions": []}
    qs = [_clean(q) for q in (d.get("questions") or []) if _clean(q)][:2]
    kind = "other" if str(d.get("kind", "")).lower() == "other" and _clean(d.get("reply")) else "health"
    return {"kind": kind, "reply": _clean(d.get("reply")), "urgent": bool(d.get("urgent")), "questions": qs}


def analyze_symptoms(info, emergency=False):
    task = EMERGENCY_TASK if emergency else SYMPTOM_TASK
    messages = [{"role": "system", "content": BASE_RULES + "\n" + task + "\n" + RESULT_SCHEMA % ""},
                {"role": "user", "content": info}]
    last_err = None
    for model in TEXT_MODELS:
        try:
            extra = {"reasoning_effort": "low"} if "gpt-oss" in model else None
            return validate_result(groq_json(messages, model, extra=extra), "emergency" if emergency else "symptoms")
        except ApiError as e:
            last_err = e
            log(f"{model} failed: {e}")
    raise last_err


def _image_message(text, b64):
    return {"role": "user", "content": [{"type": "text", "text": text},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}


def classify_image(b64):
    d = groq_json([{"role": "system", "content": CLASSIFY_PROMPT}, _image_message("Classify this photo.", b64)],
                  VISION_MODEL, max_tokens=300)
    t = str(d.get("image_type", "")).lower() if isinstance(d, dict) else ""
    t = t if t in ("skin", "other_health", "not_health", "unclear") else "other_health"
    return t, _clean(d.get("reason")) if isinstance(d, dict) else ""


def analyze_image(jpeg_bytes, emergency=False, symptoms=None):
    """Real vision pipeline: classify the photo, then run the matching assessment."""
    b64 = base64.b64encode(jpeg_bytes).decode()
    extra_note = f" The user also reported these symptoms: {symptoms}" if symptoms else ""
    if emergency:
        kind, task = "emergency", EMERGENCY_PHOTO_TASK
        ask = "EMERGENCY. I may be hurt right now. Look at this photo and tell me what to do." + extra_note
    else:
        image_type, reason = classify_image(b64)
        log(f"Photo classified as {image_type}: {reason}")
        if image_type in ("not_health", "unclear"):
            unclear = image_type == "unclear"
            r = validate_result({
                "concern": "The photo is not clear enough" if unclear else "No health concern visible in this photo",
                "assessment": reason or ("I can't see enough detail to say anything useful."),
                "confidence": "Low", "urgency": "Routine",
                "steps": [{"title": "Try again", "description": "Take a close, well lit photo of the area that "
                                                                "worries you, or describe your symptoms to me."}],
                "specialist": {"name": "General physician", "reason": "If you are worried, a doctor can check in person."},
                "limitations": "I only assess what is visible. Nothing in this photo can be judged medically.",
            }, "not_health")
            r["specialist"] = None
            return r
        kind, task = ("skin", DERM_TASK) if image_type == "skin" else ("photo", PHOTO_TASK)
        ask = "Please assess this photo." + extra_note
    messages = [{"role": "system", "content": BASE_RULES + "\n" + task + "\n" + RESULT_SCHEMA % IMAGE_KEYS},
                _image_message(ask, b64)]
    # 1400 output tokens is plenty for the JSON and keeps us inside Groq's 8000 tokens/minute limit
    return validate_result(groq_json(messages, VISION_MODEL, max_tokens=1400), kind)


def compose_speech(r):
    parts = []
    emergency = r["kind"] == "emergency"
    if r["urgency"] == "Emergency":
        first = (r["first_instruction"] or "This could be an emergency").rstrip(" .!") + "."
        parts.append(f"{first} If this is serious, call {primary_spoken()} now.")
    parts.append(("The reported concern is: " if emergency else "Here is what I found. ") + r["concern"].rstrip(".") + ".")
    if r["assessment"]:
        parts.append(r["assessment"])
    if r["visible_features"]:
        parts.append("What I can see: " + ", ".join(f.rstrip(".") for f in r["visible_features"][:4]) + ".")
    if r["possible_conditions"]:
        parts.append("Possible causes include " + "; ".join(
            c["name"] + (f", because {c['why'][0].lower() + c['why'][1:]}" if c["why"] else "")
            for c in r["possible_conditions"]).rstrip(".") + ".")
    if r["steps"]:
        parts.append("Do this now." if emergency else "Here is what to do.")
        for s in r["steps"]:
            parts.append(f"Step {s['step']}. {s['title'] + '. ' if s['title'] else ''}{s['description']}")
    if r["precautions"]:
        parts.append(("Do not: " if emergency else "Until a doctor has checked, avoid: ")
                     + "; ".join(p.rstrip(".") for p in r["precautions"]) + ".")
    if r["specialist"]:
        parts.append(f"Get help from: {r['specialist']['name']}. {r['specialist']['reason']}")
    level = {"Routine": "This looks routine. Book a normal appointment if it does not get better.",
             "Prompt": "Please get this checked promptly, ideally within a day.",
             "Emergency": "This needs emergency care."}[r["urgency"]]
    parts.append(level)
    if r["contacts"] and (r["urgency"] != "Routine" or emergency):
        nums = ", or ".join(f"{n} for {SPOKEN_PURPOSE.get(n, p)}" for n, p in r["contacts"])
        parts.append(f"Emergency numbers in {r['country']}: call {nums}.")
    elif r["urgency"] != "Routine" or emergency:
        parts.append("In an emergency, call your local emergency number.")
    if r["call_now_if"]:
        parts.append(f"Call right away if {r['call_now_if'][0].lower() + r['call_now_if'][1:]}")
    if r["limitations"]:
        parts.append(r["limitations"])
    parts.append("Remember, this is AI health information, not a confirmed diagnosis.")
    return " ".join(p.strip() for p in parts if p.strip())


# ============================================================================
# SHARED STATE
# ============================================================================
class App:
    def __init__(self):
        # mode drives the logic: idle sleeping waking listening answering camera emergency
        self.mode = "idle"
        self.busy = None              # None thinking speaking
        self.processing = False
        self.status = 'Say "Wake up Lumi", or choose a mode below'
        self.heard = ""
        self.caption = ""
        self.result = None
        self.result_id = 0
        self.show_result = False      # result screen visible
        self.nav_id = 0               # bumped by every user navigation -> cancels auto-return
        self.leaving_t = -10          # exit-transition start
        self.review = None            # photo waiting for Analyze / Retake
        self.review_deadline = None
        self.confirm = None           # {"text", "yes"} modal
        self.last_spoken = ""
        self.questions = []
        self.pending_symptoms = ""
        self.last_symptoms = ""
        self.last_symptoms_t = 0
        self.error = ""
        self.error_until = 0
        self.notice = ""
        self.notice_until = 0
        self.mic_ok = False
        self.mic_name = ""
        self.mic_level = 0.0
        self.mic_silent = False
        self.hearing = False
        self.typing = False
        self.typed_text = ""
        self.camera_opening = False
        self.speech_end = 0
        self.speak_env = None
        self.speak_t0 = 0
        self.wake_t0 = -10
        self.gesture_t = -10
        self.flash_t = -10
        self.listen_started = 0
        self.last_activity = time.time()
        self.happy_until = 0
        self.frozen_rgb = None
        self.video_opened = False
        self.video_status = None
        self.request_iconify = False
        self.request_upload = False
        self.events = queue.Queue()
        self.stop_speech = threading.Event()
        self.running = True
        self.mixer_ok = False

    def show_error(self, msg, seconds=10):
        self.error, self.error_until = msg, time.time() + seconds
        log("ERROR:", msg)

    def show_notice(self, msg, seconds=12):
        self.notice, self.notice_until = msg, time.time() + seconds
        log("NOTICE:", msg)

    @property
    def emergency(self):
        return self.mode == "emergency"


# ============================================================================
# SPEAKER
# ============================================================================
class Speaker:
    def __init__(self, app):
        self.app = app
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        self.aplay = shutil.which("aplay")
        self._chime = chime_wav()

    def chime(self):
        if self.app.mixer_ok:
            try:
                pygame.mixer.Sound(file=io.BytesIO(self._chime)).play()
            except Exception as e:
                log("chime failed:", e)

    def stop(self):
        self.app.stop_speech.set()
        if self.app.mixer_ok:
            try:
                pygame.mixer.stop()
            except Exception:
                pass
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass

    def say(self, text, remember=False):
        """Blocking, one voice at a time. Returns True only if the audio played to the very end."""
        app = self.app
        clean = clean_for_speech(text)
        if not clean:
            return True
        if remember:
            app.last_spoken = clean
        chunks = split_for_speech(clean)
        with self.lock:
            app.stop_speech.clear()
            app.busy = "speaking"
            futures = [self.pool.submit(text_to_speech, c) for c in chunks]
            played_all = True
            try:
                for chunk, fut in zip(chunks, futures):
                    if app.stop_speech.is_set():
                        break
                    app.caption = chunk
                    wav, err = fut.result()
                    if app.stop_speech.is_set():
                        break
                    if wav:
                        ok = self.play_wav(wav)
                    else:
                        app.show_error(f"Voice output failed ({err}). Using the backup system voice.")
                        ok = self.fallback_say(chunk)
                    played_all = played_all and ok
            finally:
                for f in futures:
                    f.cancel()
                stopped = app.stop_speech.is_set()
                app.busy = None
                app.caption = ""
                app.speak_env = None
                app.speech_end = time.time()
            return played_all and not stopped

    def play_wav(self, wav):
        app = self.app
        app.speak_env, app.speak_t0 = wav_envelope(wav), time.time()
        if app.mixer_ok:
            ch = pygame.mixer.Sound(file=io.BytesIO(wav)).play()
            if ch is None:
                return False
            time.sleep(0.05)
            while ch.get_busy():  # real playback-finished signal from the audio channel
                if app.stop_speech.is_set():
                    ch.stop()
                    return False
                time.sleep(0.03)
            return True
        x, sr = read_wav(wav)
        if sd is not None:
            try:
                sd.play(x, sr)
                sd.wait()
                return not app.stop_speech.is_set()
            except Exception as e:
                log("sounddevice playback failed:", e)
        if self.aplay:
            path = os.path.join(HERE, "_lumi_tmp.wav")
            with open(path, "wb") as f:
                f.write(wav)
            return self.wait_process(subprocess.Popen([self.aplay, "-q", path]))
        app.show_error("No working speaker output found.")
        return False

    def fallback_say(self, text):
        if os.name == "nt":
            ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  "$s.Speak([Console]::In.ReadToEnd())")
            try:
                p = subprocess.Popen(["powershell", "-NoProfile", "-Command", ps], stdin=subprocess.PIPE,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                p.stdin.write(text.encode("utf-8"))
                p.stdin.close()
                return self.wait_process(p)
            except Exception as e:
                log("Windows voice failed:", e)
        elif self.espeak:
            return self.wait_process(subprocess.Popen([self.espeak, "-s", "150", text]))
        self.app.show_error("Voice output failed and no backup voice is installed. The text is on screen.")
        return False

    def wait_process(self, proc):
        while proc.poll() is None:
            if self.app.stop_speech.is_set():
                proc.terminate()
                return False
            time.sleep(0.05)
        return proc.returncode == 0


# ============================================================================
# EAR: microphone + voice activity detection (+ commands while Lumi talks)
# ============================================================================
def open_mic_stream(mic=None):
    if sd is None:
        raise RuntimeError(f"sounddevice not available ({SD_IMPORT_ERROR}). Raspberry Pi: sudo apt install libportaudio2")
    dev = None
    if mic not in (None, ""):
        if str(mic).isdigit():
            dev = int(mic)
        else:
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] > 0 and str(mic).lower() in d["name"].lower():
                    dev = i
                    break
            else:
                raise RuntimeError(f"no microphone matching '{mic}' (run with --list-mics)")
    info = sd.query_devices(dev, "input")
    for ch in (1, max(1, int(info["max_input_channels"]))):
        for sr in (16000, int(info["default_samplerate"]), 48000, 44100):
            try:
                sd.check_input_settings(device=dev, channels=ch, samplerate=sr, dtype="int16")
                block = int(sr * 0.03)
                return sd.InputStream(device=dev, channels=ch, samplerate=sr, dtype="int16",
                                      blocksize=block), sr, block, info["name"]
            except Exception:
                continue
    raise RuntimeError(f"microphone '{info['name']}' refused every sample rate")


class Ear(threading.Thread):
    daemon = True

    def __init__(self, app, speaker, mic):
        super().__init__()
        self.app, self.speaker, self.mic = app, speaker, mic

    def run(self):
        while self.app.running:
            try:
                self.listen()
            except Exception as e:
                self.app.mic_ok = False
                self.app.show_error(f"Microphone problem: {e}. Retrying...", 5)
                time.sleep(3)

    def listen_mode(self):
        app = self.app
        if app.typing:
            return None
        if app.busy == "speaking":
            return "barge"
        if app.busy is None and app.mode != "waking" and time.time() > app.speech_end + 0.5:
            return "normal"
        return None

    @staticmethod
    def is_echo(n, spoken):
        words = n.split()
        if not words:
            return True
        said = set(normalize(spoken).split())
        return sum(w in said for w in words) / len(words) >= 0.75

    def barge_check(self, pcm, sr):
        spoken = self.app.caption
        text, _, _ = speech_to_text(to_wav(pcm, sr))
        n = normalize(text) if text else ""
        cmd = detect_command(n) if n else None
        if cmd in ("confirm",):
            cmd = None  # "yes" / "ok" are too easy to hear by accident while Lumi talks
        if cmd and self.is_echo(n, spoken):
            log(f"While speaking heard {text!r} -> ignored (Lumi's own echo)")
            return
        log(f"While speaking heard {text!r} -> {cmd}")
        if cmd:
            self.speaker.stop()
            if cmd != "stop":
                self.app.events.put(("text", text))

    def listen(self):
        app = self.app
        stream, sr, block, name = open_mic_stream(self.mic)
        frame_s = block / sr
        log(f"Mic: {name} @ {sr} Hz")
        with stream:
            app.mic_ok, app.mic_name = True, name
            calib, t0, noise = [], time.time(), None
            pre = deque(maxlen=int(0.4 / frame_s))
            recent = deque(maxlen=int(3.0 / frame_s))
            buf, in_speech, silence, voiced, last_mode = [], False, 0.0, 0, None
            while app.running:
                data, _ = stream.read(block)
                x = data[:, 0].astype(np.float32)
                rms = float(np.sqrt(np.mean(x * x)) + 1e-6)
                recent.append(rms)
                if len(recent) == recent.maxlen:
                    app.mic_silent = max(recent) < 4
                if noise is None:
                    calib.append(rms)
                    if time.time() - t0 > 1.0:
                        noise = max(float(np.percentile(calib, 30)), 15.0)
                        log(f"Noise floor {noise:.0f}")
                    continue
                mode = self.listen_mode()
                thr = max(noise * 3.0, MIC_MIN_RMS) * (1.6 if mode == "barge" else 1.0)
                app.mic_level = min(1.0, rms / (thr * 3))
                if DEBUG:
                    print(f"rms={rms:7.0f} thr={thr:6.0f} {mode}", end="\r")
                if mode != last_mode:
                    in_speech, buf, voiced, app.hearing = False, [], 0, False
                    pre.clear()
                    last_mode = mode
                if mode is None:
                    continue
                if not in_speech:
                    pre.append(data.copy())
                    if rms > thr:
                        voiced += 1
                        if voiced >= 3:
                            in_speech, buf, silence = True, list(pre), 0.0
                            app.hearing = mode == "normal"
                            app.last_activity = time.time()
                    else:
                        voiced = 0
                        noise = 0.98 * noise + 0.02 * min(rms, noise * 2)
                    continue
                buf.append(data.copy())
                silence = 0.0 if rms > thr * 0.7 else silence + frame_s
                dur = len(buf) * frame_s
                if mode == "barge":
                    if dur > 3.5:
                        in_speech, buf, voiced = False, [], 0
                        pre.clear()
                    elif silence >= 0.6:
                        in_speech = False
                        if dur - silence >= 0.3:
                            threading.Thread(target=self.barge_check, args=(np.concatenate(buf)[:, 0], sr),
                                             daemon=True).start()
                        buf, voiced = [], 0
                        pre.clear()
                    continue
                end_after = 1.8 if app.mode in ("listening", "answering", "emergency") else 0.9
                if silence >= end_after or dur > 30:
                    in_speech, app.hearing = False, False
                    if dur - silence >= 0.35:
                        app.events.put(("utterance", np.concatenate(buf)[:, 0], sr))
                    buf, voiced = [], 0
                    pre.clear()


# ============================================================================
# CAMERA + MEDIAPIPE GESTURES (one implementation for every mode)
# ============================================================================
def ensure_gesture_model():
    if os.path.exists(GESTURE_MODEL_PATH) and os.path.getsize(GESTURE_MODEL_PATH) > 1_000_000:
        return
    log("Downloading MediaPipe gesture model...")
    r = requests.get(GESTURE_MODEL_URL, timeout=60)
    r.raise_for_status()
    with open(GESTURE_MODEL_PATH, "wb") as f:
        f.write(r.content)


def encode_jpeg(rgb, max_side=768, quality=82):
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if scale < 1:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    surf = pygame.image.frombuffer(np.ascontiguousarray(rgb).tobytes(), (w, h), "RGB")
    if scale < 1:
        surf = pygame.transform.smoothscale(surf, (int(w * scale), int(h * scale)))
    out = io.BytesIO()
    pygame.image.save(surf, out, "photo.jpg")
    return out.getvalue()


def load_image_rgb(path):
    if cv2 is not None:
        data = np.fromfile(path, dtype=np.uint8)  # works with non-ASCII Windows paths
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if img is None:
            raise ApiError("That file is not a photo Lumi can read (use JPG or PNG).")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        surf = pygame.image.load(path)
    except Exception:
        raise ApiError("That file is not a photo Lumi can read (use JPG or PNG).")
    return np.ascontiguousarray(pygame.surfarray.array3d(surf).swapaxes(0, 1))


class CameraWorker:
    def __init__(self, app):
        self.app = app
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.cap = None
        self.picam = None
        self.frame = None
        self.raw = None
        self.recognizer = None
        self.recognizer_error = None
        self.gesture = ("", 0.0)
        self.hand = False
        self.hold = 0.0
        self.hold_start = None
        self.armed_at = 0.0
        self.mode = None
        self.ts = 0
        self.last_open_error = ""

    @property
    def active(self):
        return self.running

    def load_recognizer(self):
        if self.recognizer is not None:
            return
        if mp is None:
            raise RuntimeError(f"mediapipe not installed ({MP_IMPORT_ERROR})")
        ensure_gesture_model()
        opts = mp_vision.GestureRecognizerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        self.recognizer = mp_vision.GestureRecognizer.create_from_options(opts)

    def open_device(self):
        tried = []
        if cv2 is not None:
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if os.name == "nt" else [cv2.CAP_V4L2, cv2.CAP_ANY]
            for idx in [CAMERA_INDEX] + [i for i in (0, 1, 2) if i != CAMERA_INDEX]:
                for be in backends:
                    cap = cv2.VideoCapture(idx, be)
                    if not cap.isOpened():
                        cap.release()
                        tried.append(f"{idx}/{be}: not opened")
                        continue
                    for size in ((640, 480), None):
                        if size:
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
                        for _ in range(15):
                            ok, f = cap.read()
                            if ok and f is not None and f.size:
                                self.cap = cap
                                log(f"Camera {idx} opened (backend {be}, {f.shape[1]}x{f.shape[0]})")
                                return True
                            time.sleep(0.05)
                    cap.release()
                    tried.append(f"{idx}/{be}: opened but sent no frames (busy in another app?)")
        else:
            tried.append("opencv-python is not installed")
        try:
            from picamera2 import Picamera2
            pc = Picamera2()
            pc.configure(pc.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
            pc.start()
            self.picam = pc
            return True
        except Exception as e:
            tried.append(f"picamera2: {e}")
        self.last_open_error = "; ".join(tried[:4])
        log("Camera open failed:", self.last_open_error)
        return False

    def start(self, mode):
        self.stop()
        self.recognizer_error = None
        try:
            self.load_recognizer()
        except Exception as e:
            self.recognizer_error = str(e)
            log("Gesture recognizer unavailable:", e)
        if not self.open_device():
            return False, ("Could not open the camera. Check it is connected, not used by another app (Zoom, Teams, "
                           "browser), and allowed in Settings > Privacy & security > Camera. You can Upload a photo "
                           "instead.")
        self.mode = mode
        self.hold, self.hold_start = 0.0, None
        self.armed_at = time.time() + 1.0
        self.frame = self.raw = None
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return True, None

    def read_raw(self):
        if self.cap is not None:
            ok, bgr = self.cap.read()
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
        if self.picam is not None:
            return np.ascontiguousarray(self.picam.capture_array()[:, :, 2::-1])
        return None

    def loop(self):
        fails = 0
        while self.running:
            rgb = self.read_raw()
            if rgb is None:
                fails += 1
                if fails > 60:
                    self.app.show_error("The camera stopped sending video. Close other apps using it and try again.")
                    self.running = False
                    break
                time.sleep(0.03)
                continue
            fails = 0
            with self.lock:
                self.raw = rgb
                self.frame = np.ascontiguousarray(rgb[:, ::-1])
            if self.recognizer is not None and self.armed_at != float("inf"):
                self.detect(rgb)
        self.release()

    def detect(self, rgb):
        self.ts = max(self.ts + 1, int(time.monotonic() * 1000))
        try:
            res = self.recognizer.recognize_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), self.ts)
        except Exception as e:
            log("gesture error:", e)
            return
        name, score = "", 0.0
        if res.gestures:
            c = res.gestures[0][0]
            name, score = c.category_name, float(c.score)
        self.gesture, self.hand = (name, score), bool(res.hand_landmarks)
        now = time.time()
        if name == "Thumb_Up" and score >= THUMB_MIN_SCORE and now >= self.armed_at:
            if self.hold_start is None:
                self.hold_start = now
            self.hold = min(1.0, (now - self.hold_start) / THUMB_HOLD_SECONDS)
            if self.hold >= 1.0:
                self.app.gesture_t = now
                self.capture(rgb)
        else:
            self.hold_start, self.hold = None, 0.0

    def capture(self, rgb=None):
        """Thread-safe: one photo until rearm(); hands it to the Brain for review."""
        with self.lock:
            if self.armed_at == float("inf") or not self.running:
                return False
            rgb = rgb if rgb is not None else self.raw
            if rgb is None:
                return False
            self.armed_at = float("inf")
        self.hold_start, self.hold = None, 0.0
        app = self.app
        app.flash_t = time.time()
        app.events.put(("photo", rgb.copy(), np.ascontiguousarray(rgb[:, ::-1]), "camera", time.time()))
        if app.busy == "speaking":
            app.stop_speech.set()
            if app.mixer_ok:
                try:
                    pygame.mixer.stop()
                except Exception:
                    pass
        log("Photo captured from camera")
        return True

    def rearm(self, delay):
        self.armed_at = time.time() + delay

    def get_frame(self):
        with self.lock:
            return self.frame

    def release(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            if self.picam is not None:
                try:
                    self.picam.stop()
                    self.picam.close()
                except Exception:
                    pass
            self.cap = self.picam = None

    def stop(self):
        self.running = False
        t = self.thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2)
        self.thread = None
        self.release()
        self.gesture, self.hand, self.hold = ("", 0.0), False, 0.0


def place_call(number):
    """Opens the device's phone handler (only after the user confirmed). Returns True if a handler started."""
    try:
        if os.name == "nt":
            os.startfile(f"tel:{number}")  # Windows Phone Link / Skype etc., if installed
            return True
        opener = shutil.which("xdg-open") or shutil.which("open")
        if opener:
            p = subprocess.run([opener, f"tel:{number}"], capture_output=True, timeout=8)
            return p.returncode == 0
    except Exception as e:
        log("Call handler failed:", e)
    return False


# ============================================================================
# BRAIN: one state machine for every workflow
# ============================================================================
class Brain(threading.Thread):
    daemon = True

    def __init__(self, app, speaker, camera):
        super().__init__()
        self.app, self.speaker, self.camera = app, speaker, camera
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.last_cmd = (None, 0)
        self.analysis_window = (0.0, 0.0)

    def stale(self, ev):
        """True for a photo that arrived while another photo was being analyzed (no duplicate requests)."""
        t = ev[-1] if isinstance(ev[-1], float) else time.time()
        start, end = self.analysis_window
        if start <= t <= end:
            log("Ignored a photo that arrived while another one was being analyzed")
            return True
        return False

    def run(self):
        app = self.app
        handlers = {
            "utterance": lambda e: self.on_utterance(e[1], e[2]),
            "text": lambda e: self.handle_text(e[1], typed=len(e) > 2 and e[2]),
            "wake": lambda e: self.wake(),
            "camera": lambda e: self.start_camera("standard"),
            "emergency": lambda e: self.start_emergency(),
            "close_camera": lambda e: self.close_camera(),
            "repeat": lambda e: self.repeat(),
            "photo": lambda e: None if self.stale(e) else self.on_photo(e[1], e[2], e[3]),
            "upload": lambda e: None if self.stale(e) else self.on_upload(e[1]),
            "analyze_photo": lambda e: self.analyze_review(),
            "retake": lambda e: self.retake(),
            "home": lambda e: self.go_home(),
            "exit_emergency": lambda e: self.exit_emergency(),
            "call": lambda e: self.call(e[1]),
        }
        while app.running:
            try:
                ev = app.events.get(timeout=0.25)
            except queue.Empty:
                self.tick()
                continue
            app.last_activity = time.time()
            app.processing = True
            try:
                fn = handlers.get(ev[0])
                if fn:
                    fn(ev)
            except Exception as e:
                log("Brain error:", repr(e))
                app.busy = None
                app.show_error(f"Something went wrong: {e}")
            finally:
                app.processing = False

    def tick(self):
        app, now = self.app, time.time()
        if app.review and app.review_deadline and now >= app.review_deadline and app.confirm is None:
            app.review_deadline = None
            self.analyze_review()
            return
        if app.busy or app.hearing or app.typing:
            return
        if app.mode == "answering" and now - app.listen_started > 25:
            self.finish_followup("(no answer given)")
        elif app.mode == "listening" and now - app.listen_started > 25:
            app.mode, app.status = "idle", 'I stopped listening. Say "Wake up Lumi" when you need me.'
        elif app.mode == "idle" and now - app.last_activity > 120 and not app.show_result:
            app.mode, app.status = "sleeping", 'Zzz... say "Wake up Lumi"'

    # ---------- input ----------
    def on_utterance(self, pcm, sr):
        text, provider, err = speech_to_text(to_wav(pcm, sr))
        log(f"Heard ({provider}): {text!r}" if text else f"Heard nothing ({err or 'silence'})")
        if not text:
            if err:
                self.app.show_error(f"Speech-to-text failed: {err}")
            return
        self.handle_text(text)

    def handle_text(self, text, typed=False):
        app = self.app
        app.heard = text
        n = normalize(text)
        cmd = None if typed and len(n.split()) > 3 else detect_command(n)
        if cmd in ("confirm", "retake") and not app.review:
            cmd = None if cmd == "confirm" else cmd
        if cmd:
            last, t = self.last_cmd
            if cmd == last and time.time() - t < 2.5:
                return
            self.last_cmd = (cmd, time.time())
            app.nav_id += 1
        if cmd == "exit_emergency":
            return self.exit_emergency() if app.emergency else None
        if cmd == "emergency":
            rest = strip_emergency_words(text)
            return self.start_emergency(rest if len(rest.split()) >= 3 else None)
        if cmd == "stop":
            return self.speaker.stop()
        if cmd == "repeat":
            return self.repeat()
        if cmd == "retake":
            return self.retake() if app.review else None
        if cmd == "confirm":
            return self.analyze_review()
        if cmd == "home":
            return self.go_home() if not app.emergency else None
        if cmd == "close_camera":
            return self.close_camera() if self.camera.active else None
        if cmd == "camera":
            return None if self.camera.active else self.start_camera("emergency" if app.emergency else "standard")
        if cmd == "wake" and app.mode not in ("listening", "answering", "emergency"):
            return self.wake()

        if app.mode == "answering":
            return self.finish_followup(text)
        if app.mode == "emergency":
            return self.run_symptoms(text, emergency=True)
        if app.mode == "listening" or typed:
            return self.start_symptoms(text)
        if app.mode == "camera":
            app.status = 'Camera is on. Show a thumbs-up, press Capture, or say "Close camera".'
            return
        app.status = 'Say "Wake up Lumi" first, or press Type.'

    # ---------- navigation ----------
    def go_home(self):
        app = self.app
        if app.emergency:
            return
        self.speaker.stop()
        self.camera.stop()
        app.review, app.review_deadline, app.frozen_rgb = None, None, None
        app.show_result = False
        app.mode, app.questions = "idle", []
        app.leaving_t = -10
        app.status = 'Say "Wake up Lumi", or choose a mode below'

    def exit_emergency(self):
        app = self.app
        self.speaker.stop()
        self.camera.stop()
        app.review, app.review_deadline, app.frozen_rgb = None, None, None
        app.mode, app.show_result, app.video_status = "idle", False, None
        app.status = "Glad you're safe. I'm here if you need me."
        app.leaving_t = -10

    def auto_return_home(self, nav):
        """Called only after the audio channel reports that the full response finished playing."""
        app = self.app
        if app.emergency or nav != app.nav_id or not app.show_result:
            return
        app.leaving_t = time.time()  # exit transition (fade) driven by the renderer
        time.sleep(0.6)
        if nav == app.nav_id and not app.emergency:
            self.go_home()
            app.status = "All done. Your result is saved - press V to view it again."
        else:
            app.leaving_t = -10

    # ---------- wake ----------
    def wake(self):
        app = self.app
        if self.camera.active:
            self.camera.stop()
        app.review, app.frozen_rgb, app.show_result = None, None, False
        app.mode, app.wake_t0 = "waking", time.time()
        app.status = "Waking up..."
        self.speaker.chime()
        time.sleep(0.6)
        self.speaker.say("I'm awake. How can I help you?")
        app.mode, app.listen_started, app.heard = "listening", time.time(), ""
        app.status = 'Tell me your symptoms, or say "Camera Lumi" or "Emergency Lumi".'

    # ---------- Diagnosis Lumi ----------
    def start_symptoms(self, text):
        app = self.app
        app.last_symptoms, app.last_symptoms_t = text, time.time()
        if has_red_flag(text):
            return self.run_symptoms(text, emergency=True)
        app.busy, app.status = "thinking", "Checking your symptoms..."
        try:
            t = triage(text)
        except ApiError as e:
            log("Triage failed, analyzing directly:", e)
            t = {"kind": "health", "reply": "", "urgent": False, "questions": []}
        app.busy = None
        if t["kind"] == "other":
            app.status = "Tell me about any symptoms whenever you're ready."
            self.speaker.say(t["reply"] + " If you have a health problem, just tell me your symptoms.")
            app.mode, app.listen_started = "listening", time.time()
            return
        if t["urgent"]:
            return self.run_symptoms(text, emergency=True)
        if t["questions"]:
            app.pending_symptoms, app.questions = text, t["questions"]
            app.mode = "answering"
            app.status = "A quick question first (answer by voice, or press T to type):"
            self.speaker.say("I have a quick question. " + " ".join(t["questions"]))
            app.listen_started = time.time()
            return
        self.run_symptoms(text)

    def finish_followup(self, answer):
        app = self.app
        if answer != "(no answer given)" and not has_red_flag(answer):
            app.busy = "thinking"
            try:
                t = triage(answer)
            except ApiError:
                t = {"kind": "health", "reply": ""}
            app.busy = None
            if t["kind"] == "other":
                self.speaker.say(t["reply"] + " Now, about your symptoms: " + " ".join(app.questions))
                app.listen_started = time.time()
                return
        info = (f"Symptoms: {app.pending_symptoms}\nFollow-up questions: {' '.join(app.questions)}\n"
                f"User's answer: {answer}")
        app.questions = []
        app.mode = "listening"
        app.last_symptoms = f"{app.pending_symptoms}. {answer}"
        self.run_symptoms(info)

    def run_symptoms(self, info, emergency=False):
        app = self.app
        emergency = emergency or has_red_flag(info)
        app.busy = "thinking"
        app.status = "Analyzing with Groq..." if not emergency else "Getting emergency guidance..."
        fut = self.pool.submit(analyze_symptoms, info, emergency)
        if emergency:  # urgent advice right away, the full analysis keeps loading
            self.speaker.say(f"This may be an emergency. If it is life threatening, call {primary_spoken()} now. "
                             "Stay with me, I am getting more guidance.")
            app.busy = "thinking"
        try:
            result = fut.result()
        except ApiError as e:
            app.busy = None
            app.show_error(f"Analysis failed: {e}", 15)
            app.status = "I couldn't get an answer. Please try again."
            self.speaker.say(f"Sorry, I couldn't get an answer. {e}")
            return
        if emergency and not app.emergency:
            self.start_emergency(silent=True)
        self.present(result, {"source": "emergency-voice" if emergency else "voice", "symptoms": info})

    # ---------- Camera Lumi ----------
    def start_camera(self, mode, background=False):
        app = self.app
        emergency = mode == "emergency"
        if self.camera.active and (self.camera.mode == mode or not emergency):
            return True
        if not background:
            self.speaker.stop()
        if not emergency:
            app.mode, app.show_result = "camera", False
        app.review, app.frozen_rgb = None, None
        app.camera_opening = True
        if not background:
            app.status = "Opening the camera..."
        ok, err = self.camera.start(mode)
        app.camera_opening = False
        if not ok:
            app.show_error(err, 20)
            log("camera details:", self.camera.last_open_error)
            app.status = "Camera unavailable. You can Upload a photo instead."
            if not emergency:
                app.mode = "idle"
                self.speaker.say("I could not open the camera. Please check that it is connected and not used by "
                                 "another app, or upload a photo instead.")
            return False
        if self.camera.recognizer is None:
            app.show_error(f"Thumbs-up detection unavailable ({self.camera.recognizer_error}). Use Capture.", 15)
        if not emergency:
            app.status = "Camera on. Hold a thumbs-up, or press Capture."
            if not self.has_queued("photo"):
                self.speaker.say("Camera is on. Show me a thumbs up, or press capture.")
        return True

    def has_queued(self, kind):
        with self.app.events.mutex:
            return any(e[0] == kind for e in self.app.events.queue)

    def close_camera(self):
        app = self.app
        self.camera.stop()
        app.review, app.review_deadline, app.frozen_rgb = None, None, None
        if app.mode == "camera":
            app.mode = "idle"
        app.status = 'Camera closed.' if app.emergency else 'Camera closed. Say "Wake up Lumi" when you need me.'

    def on_photo(self, rgb, display_rgb, source, path=None):
        """A photo from the camera or the gallery: show it for Retake / Analyze."""
        app = self.app
        if app.review and app.review.get("analyzing"):
            return  # never send two photos at once
        os.makedirs(PHOTO_DIR, exist_ok=True)
        jpeg = encode_jpeg(rgb)
        if path is None:
            path = os.path.join(PHOTO_DIR, datetime.now().strftime("lumi_%Y%m%d_%H%M%S.jpg"))
            with open(path, "wb") as f:
                f.write(jpeg)
        app.frozen_rgb = display_rgb
        app.review = {"jpeg": jpeg, "path": path, "source": source, "analyzing": False,
                      "mode": "emergency" if app.emergency else "standard", "t": time.time()}
        app.show_result = False
        # analyze straight away as soon as the photo is taken (no confirm step)
        app.review_deadline = None
        return self.analyze_review()

    def on_upload(self, path):
        app = self.app
        app.nav_id += 1
        try:
            rgb = load_image_rgb(path)
        except ApiError as e:
            app.show_error(str(e))
            return
        if app.mode not in ("emergency",):
            app.mode = "camera"
        if self.camera.active:
            self.camera.armed_at = float("inf")
        log("Photo uploaded:", path)
        self.on_photo(rgb, np.ascontiguousarray(rgb), "upload", path=path)

    def retake(self):
        app = self.app
        source = app.review["source"] if app.review else "camera"
        app.review, app.review_deadline, app.frozen_rgb = None, None, None
        if source == "upload":
            app.request_upload = True
            return
        if self.camera.active:
            self.camera.rearm(1.0)
            app.status = "Retake: hold a thumbs-up, or press Capture."
        else:
            self.start_camera("emergency" if app.emergency else "standard")

    def analyze_review(self):
        app = self.app
        rv = app.review
        if not rv or rv["analyzing"]:
            return
        rv["analyzing"] = True
        self.analysis_window = (time.time(), float("inf"))
        app.review_deadline = None
        emergency = rv["mode"] == "emergency"
        app.busy = "thinking"
        app.status = "Analyzing your photo..."
        recent = app.last_symptoms if time.time() - app.last_symptoms_t < 900 else None
        t0 = time.time()
        log("Photo analysis started")
        try:
            result = analyze_image(rv["jpeg"], emergency, recent)
        except ApiError as e:
            log(f"Photo analysis failed after {time.time() - t0:.1f}s: {e}")
            self.analysis_window = (self.analysis_window[0], time.time())
            app.busy = None
            rv["analyzing"] = False
            app.review_deadline = None
            app.show_error(f"Photo analysis failed: {e}", 15)
            app.status = "Analysis failed. Press Analyze to try again, or Retake."
            self.speaker.say(f"Sorry, I couldn't analyze the photo. {e}")
            return
        log(f"Photo analysis done in {time.time() - t0:.1f}s: {result['kind']} | {result['concern']}")
        self.analysis_window = (self.analysis_window[0], time.time())
        result["photo"] = rv["path"]
        app.review, app.frozen_rgb = None, None
        if emergency:
            if self.camera.active:
                self.camera.rearm(3.0)
        else:
            self.camera.stop()
            if app.mode == "camera":
                app.mode = "idle"
        self.present(result, {"source": "emergency-camera" if emergency else f"camera-{rv['source']}",
                              "photo": rv["path"], "model": VISION_MODEL})
        if emergency:
            self.open_emergency_video()

    # ---------- Emergency Lumi ----------
    def start_emergency(self, symptoms=None, silent=False):
        app = self.app
        if not app.emergency:
            app.result, app.show_result = None, False
            app.video_opened, app.video_status = False, None
        app.mode = "emergency"
        app.review, app.review_deadline = None, None
        opened = self.camera.active and self.camera.mode == "emergency"
        if symptoms or silent:
            if not opened:
                if self.camera.active:
                    self.camera.stop()
                threading.Thread(target=self.start_camera, args=("emergency", True), daemon=True).start()
            if symptoms:
                return self.run_symptoms(symptoms, emergency=True)
            return
        if not opened:
            if self.camera.active:
                self.camera.stop()
            self.start_camera("emergency")
        app.status = "Tell me what happened, or show the injury and give a thumbs-up."
        self.speaker.say(f"Emergency mode. If this is life threatening, call {primary_spoken()} now. Tell me what "
                         "happened, or show me the injury and give a thumbs up.")

    def call(self, number):
        app = self.app
        ok = place_call(number)
        if ok:
            app.show_notice(f"Opening your phone app to call {number}. If nothing happens, dial {number} on your "
                            "phone now.", 20)
        else:
            app.show_notice(f"This device cannot place phone calls. Dial {number} on your phone now.", 30)

    # ---------- shared ----------
    def present(self, result, meta):
        """Show the structured result, read all of it, then return home (never during an emergency)."""
        app = self.app
        result["time"] = time.time()
        app.result = result
        app.result_id += 1
        app.busy = None
        app.show_result = True
        app.questions = []
        if app.mode in ("listening", "answering"):
            app.mode = "idle"
        save_history(dict(meta, result=result))
        nav = app.nav_id
        app.status = "Here's what I found."
        finished = self.speaker.say(compose_speech(result), remember=True)
        if app.emergency:
            app.status = "Emergency guidance stays on screen. Press \"I'm safe\" when you have help."
            return
        if finished:
            self.auto_return_home(nav)
        else:
            app.status = "Press Home when you're done reading, or Repeat to hear it again."

    def repeat(self):
        if self.app.last_spoken:
            if self.app.result and not self.app.emergency:
                self.app.show_result = True
            self.speaker.say(self.app.last_spoken)
        else:
            self.speaker.say("There is nothing to repeat yet.")

    def open_emergency_video(self):
        app = self.app
        if app.video_opened:
            return
        app.video_opened = True
        try:
            ok = webbrowser.open(EMERGENCY_VIDEO, new=2)
        except Exception:
            ok = False
        app.video_status = "opened" if ok else "blocked"
        if ok:
            app.request_iconify = True
        log("Emergency video", "opened" if ok else "could not be opened automatically")


# ============================================================================
# DRAWING
# ============================================================================
BG = (9, 11, 22)
CORE = (14, 18, 40)
CARD = (22, 26, 44)
CARD_HI = (28, 33, 56)
LINE = (44, 50, 78)
TEXT = (236, 238, 246)
MUTED = (150, 158, 184)
GREEN = (52, 199, 89)
RED = (255, 59, 48)
AMBER = (255, 159, 10)
BLUE = (10, 132, 255)
TEAL = (90, 220, 255)
URGENCY = {"Routine": (GREEN, "Routine care"), "Prompt": (AMBER, "Prompt assessment"),
           "Emergency": (RED, "Emergency care")}
KIND_TITLE = {"symptoms": "Diagnosis Lumi", "skin": "Camera Lumi - Skin check", "photo": "Camera Lumi - Photo check",
              "not_health": "Camera Lumi - Photo check", "emergency": "Emergency Lumi"}
COLORS = {
    "sleeping": (110, 140, 210), "awake": TEAL, "listening": (110, 255, 190),
    "thinking": (185, 155, 255), "speaking": (255, 160, 215), "happy": (255, 205, 110),
    "camera": TEAL, "emergency": (255, 75, 90),
}


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def visual_state(app, now):
    if app.busy:
        return app.busy
    if app.mode == "emergency":
        return "emergency"
    if app.mode == "camera":
        return "camera"
    if app.mode in ("listening", "answering") or app.hearing:
        return "listening"
    if app.mode == "sleeping":
        return "sleeping"
    return "happy" if now < app.happy_until else "awake"


def current_screen(app, camera):
    if app.emergency:
        return "emergency"
    if app.review is not None:
        return "review"
    if camera.active or app.camera_opening:
        return "camera"
    if app.show_result and app.result:
        return "result"
    return "home"


class Face:
    def __init__(self):
        self.open = 1.0
        self.blink_t = None
        self.next_blink = time.time() + 2.5
        self.look = [0.0, 0.0]
        self.look_target = [0.0, 0.0]
        self.next_look = time.time() + 2
        self.color = list(COLORS["awake"])
        self.level = 0.0

    def update(self, app, state, dt, now):
        target = COLORS["emergency"] if app.emergency else COLORS.get(state, COLORS["awake"])
        k = min(1.0, dt * 5)
        self.color = [lerp(c, t, k) for c, t in zip(self.color, target)]
        self.level = lerp(self.level, app.mic_level if state == "listening" else 0, min(1, dt * 12))
        if app.mode == "waking":
            self.open = min(1.0, max(0.05, (now - app.wake_t0) / 0.6))
            return
        if self.blink_t is None and now >= self.next_blink and state not in ("sleeping", "happy"):
            self.blink_t = now
        if self.blink_t is not None:
            p = (now - self.blink_t) / 0.16
            if p >= 1:
                self.blink_t, self.open = None, 1.0
                self.next_blink = now + (0.25 if random.random() < 0.2 else random.uniform(2.2, 5.5))
            else:
                self.open = abs(math.cos(p * math.pi))
        if state == "thinking":
            self.look_target = [0.55, -0.6]
        elif state in ("listening", "speaking", "camera", "emergency"):
            self.look_target = [math.sin(now * 0.8) * 0.12, 0.0]
        elif now >= self.next_look:
            self.look_target = [random.uniform(-0.7, 0.7), random.uniform(-0.45, 0.45)]
            if random.random() < 0.35:
                self.look_target = [0.0, 0.0]
            self.next_look = now + random.uniform(1.4, 3.8)
        k = min(1.0, dt * 7)
        self.look = [lerp(self.look[0], self.look_target[0], k), lerp(self.look[1], self.look_target[1], k)]

    @staticmethod
    def mouth_amp(app, now):
        if app.speak_env is not None:
            i = int((now - app.speak_t0) * 30)
            return float(app.speak_env[i]) if 0 <= i < len(app.speak_env) else 0.0
        return 0.0

    def draw(self, surf, app, state, cx, cy, u, now):
        col = tuple(int(c) for c in self.color)
        worried = app.emergency
        ew, eh, gap = 130 * u, 170 * u, 120 * u
        if state == "listening":
            eh *= 1.04 + self.level * 0.14
        elif state == "thinking":
            eh *= 0.72
        lx, ly = self.look[0] * 16 * u, self.look[1] * 14 * u
        for side in (-1, 1):
            ex = cx + side * gap + lx
            if state == "sleeping":
                r = pygame.Rect(0, 0, ew * 0.9, eh * 0.45)
                r.center = (ex, cy + eh * 0.05)
                pygame.draw.arc(surf, col, r, math.pi, 2 * math.pi, max(2, int(12 * u)))
                continue
            if state == "happy":
                r = pygame.Rect(0, 0, ew * 0.95, eh * 0.62)
                r.center = (ex, cy + ly)
                pygame.draw.arc(surf, col, r, 0, math.pi, max(3, int(16 * u)))
                continue
            h = max(eh * self.open, 6 * u)
            rect = pygame.Rect(0, 0, ew, h)
            rect.center = (ex, cy + ly)
            self.glow(surf, rect, col, u)
            pygame.draw.rect(surf, col, rect, border_radius=int(min(ew, h) * 0.45))
            if h > ew * 0.5:
                pygame.draw.circle(surf, (255, 255, 255),
                                   (int(rect.x + ew * 0.32 + lx * 0.6), int(rect.y + h * 0.28 + ly * 0.5)), max(1, int(ew * 0.13)))
                pygame.draw.circle(surf, (255, 255, 255),
                                   (int(rect.x + ew * 0.58 + lx * 0.6), int(rect.y + h * 0.46 + ly * 0.5)), max(1, int(ew * 0.06)))
            if worried:
                g = 30 * u
                drop, slope = h * 0.4, h * 0.4 / rect.w
                L, R, T = rect.left - g, rect.right + g, rect.top - g
                yl, yr = (rect.top + drop + slope * g, rect.top - slope * g) if side < 0 else \
                         (rect.top - slope * g, rect.top + drop + slope * g)
                pygame.draw.polygon(surf, CORE, [(L, T), (R, T), (R, yr), (L, yl)])
        if state in ("awake", "happy", "speaking", "listening", "sleeping") and not worried:
            cheek = pygame.Surface((max(2, int(ew * 0.6)), max(2, int(ew * 0.3))), pygame.SRCALPHA)
            pygame.draw.ellipse(cheek, (255, 110, 160, 85), cheek.get_rect())
            for side in (-1, 1):
                surf.blit(cheek, cheek.get_rect(center=(cx + side * (gap + ew * 0.32), cy + eh * 0.5)))
        my, w4 = cy + 128 * u, max(2, int(4 * u))
        if state == "speaking":
            amp = self.mouth_amp(app, now)
            r = pygame.Rect(0, 0, (40 + amp * 18) * u, (8 + amp * 34) * u)
            r.center = (cx, my)
            pygame.draw.ellipse(surf, (70, 20, 55), r)
            pygame.draw.ellipse(surf, col, r, w4)
        elif state == "listening":
            for i in range(7):
                lv = self.level * (0.5 + 0.5 * math.sin(now * 10 + i * 1.3)) + 0.08
                bh = (6 + lv * 46) * u
                pygame.draw.rect(surf, col, pygame.Rect(cx + (i - 3) * 13 * u - 4 * u, my - bh / 2, max(2, 8 * u), bh),
                                 border_radius=max(1, int(4 * u)))
        elif state == "thinking":
            pygame.draw.line(surf, col, (cx - 6 * u, my), (cx + 26 * u, my - 4 * u), w4)
        elif state in ("emergency", "camera"):
            pygame.draw.circle(surf, col, (int(cx), int(my)), max(2, int(13 * u)), w4)
        elif state == "sleeping":
            pygame.draw.line(surf, col, (cx - 12 * u, my), (cx + 12 * u, my), w4)
        else:
            r = pygame.Rect(0, 0, 58 * u, 34 * u)
            r.center = (cx, my - 8 * u)
            pygame.draw.arc(surf, col, r, math.pi + 0.25, 2 * math.pi - 0.25, w4)

    @staticmethod
    def glow(surf, rect, col, u):
        g = max(3, int(22 * u))
        layer = pygame.Surface((rect.w + 2 * g, rect.h + 2 * g), pygame.SRCALPHA)
        for i, a in enumerate((22, 34, 50)):
            pad = g - i * (g // 3)
            r = pygame.Rect(g - pad, g - pad, rect.w + 2 * pad, rect.h + 2 * pad)
            pygame.draw.rect(layer, (*col, a), r, border_radius=int(min(r.w, r.h) * 0.5))
        surf.blit(layer, (rect.x - g, rect.y - g))


def make_radial_mask(size=256):
    y, x = np.mgrid[0:size, 0:size]
    d = np.sqrt((x - size / 2 + 0.5) ** 2 + (y - size / 2 + 0.5) ** 2) / (size / 2)
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 0))
    pa = pygame.surfarray.pixels_alpha(surf)
    pa[:, :] = (np.clip(1 - d, 0, 1) ** 2.2 * 170).astype(np.uint8).T
    del pa
    return surf


class Orb:
    def __init__(self):
        self.mask = make_radial_mask()
        self.parts = [{"a": random.uniform(0, 2 * math.pi), "d": random.uniform(1.15, 1.9),
                       "s": random.uniform(0.05, 0.3) * random.choice((-1, 1)),
                       "r": random.uniform(1.2, 2.8), "ph": random.uniform(0, 6.28)} for _ in range(40)]
        self.amp = 0.0

    def draw(self, surf, app, face, state, cx, cy, R, now, dt, u, particles=True):
        col = tuple(int(c) for c in face.color)
        target = face.mouth_amp(app, now) if state == "speaking" else app.mic_level if state == "listening" else 0.0
        self.amp = lerp(self.amp, target, min(1, dt * 10))
        breath = 1 + 0.025 * math.sin(now * 1.3)
        wake_p = now - app.wake_t0
        waking = wake_p < 1.8
        boost = 0.3 * max(0.0, 1 - wake_p / 1.4) if waking else 0.0
        hs = max(8, int(R * 3.6 * (breath + 0.2 * self.amp + boost)))
        halo = pygame.transform.smoothscale(self.mask, (hs, hs))
        halo.fill((*col, 255), special_flags=pygame.BLEND_RGBA_MULT)
        surf.blit(halo, halo.get_rect(center=(int(cx), int(cy))))
        if particles:
            speed = 3.5 if state == "thinking" else (2.0 if waking else 1.0)
            for p in self.parts:
                p["a"] += p["s"] * dt * speed
                d = p["d"] + 0.06 * math.sin(now * 0.9 + p["ph"])
                if waking:
                    d *= 1 + 0.7 * math.sin(min(1.0, wake_p / 1.8) * math.pi)
                bright = 0.3 + 0.45 * (0.5 + 0.5 * math.sin(now * 2 + p["ph"]))
                pygame.draw.circle(surf, mix(BG, col, bright),
                                   (int(cx + math.cos(p["a"]) * R * d), int(cy + math.sin(p["a"]) * R * d)),
                                   max(1, int(p["r"] * u)))
        if waking:
            for k in range(3):
                q = (wake_p - k * 0.22) / 1.2
                if 0 <= q <= 1:
                    pygame.draw.circle(surf, mix(col, BG, q), (int(cx), int(cy)), int(R * (1 + 1.4 * q)),
                                       max(1, int(5 * u * (1 - q)) + 1))
        gp = now - app.gesture_t
        if gp < 1.2:
            for k in range(2):
                q = (gp - k * 0.2) / 0.9
                if 0 <= q <= 1:
                    pygame.draw.circle(surf, mix(GREEN, BG, q), (int(cx), int(cy)), int(R * (1 + 1.2 * q)),
                                       max(1, int(6 * u * (1 - q)) + 1))
        Rb = max(4, int(R * breath * (1 + 0.05 * self.amp)))
        pygame.draw.circle(surf, CORE, (int(cx), int(cy)), Rb)
        pygame.draw.circle(surf, mix(col, CORE, 0.78), (int(cx), int(cy)), int(Rb * 0.9), max(1, int(2 * u)))
        pygame.draw.circle(surf, col, (int(cx), int(cy)), Rb, max(2, int((3 + 7 * self.amp) * u)))
        if state == "listening":
            pts = []
            for i in range(96):
                a = i / 96 * 2 * math.pi
                w = (0.55 + 0.45 * math.sin(a * 7 + now * 6)) * (0.6 + 0.4 * math.sin(a * 3 - now * 4))
                r = Rb + (9 + app.mic_level * 42 * w) * u * (R / (120 * u) if R < 120 * u else 1)
                pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
            pygame.draw.lines(surf, col, True, pts, max(1, int(2 * u)))
        if state == "thinking" or app.camera_opening:
            for k, (rr, sp, ln) in enumerate(((1.12, 2.6, 1.4), (1.25, -1.7, 0.9), (1.38, 1.1, 0.5))):
                rect = pygame.Rect(0, 0, 2 * Rb * rr, 2 * Rb * rr)
                rect.center = (cx, cy)
                a0 = now * sp
                for j in (0, math.pi):
                    pygame.draw.arc(surf, mix(col, BG, 0.15 + k * 0.25), rect, a0 + j, a0 + j + ln, max(2, int(3 * u)))
        face.draw(surf, app, state, cx, cy - R * 0.06, R * 0.62 / 150, now)


# ---------------------------------------------------------------- UI helpers
def wrap(text, font, max_w):
    lines, cur = [], ""
    for word in str(text).split():
        t = (cur + " " + word).strip()
        if font.size(t)[0] <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def text_block(surf, text, font, color, x, y, w, max_lines=99, center=False, gap=2):
    for line in wrap(text, font, w)[:max_lines]:
        img = font.render(line, True, color)
        surf.blit(img, img.get_rect(midtop=(int(x + w / 2), int(y))) if center else (int(x), int(y)))
        y += font.get_linesize() + gap
    return y


def pill(surf, font, text, fg, bg, center, border=None, pad=1.0):
    img = font.render(text, True, fg)
    h = img.get_height()
    r = img.get_rect(center=(int(center[0]), int(center[1]))).inflate(int(h * 1.1 * pad), int(h * 0.55 * pad))
    pygame.draw.rect(surf, bg, r, border_radius=r.h // 2)
    if border:
        pygame.draw.rect(surf, border, r, 2, border_radius=r.h // 2)
    surf.blit(img, img.get_rect(center=r.center))
    return r


def shadowed_card(surf, rect, u, color=CARD, border=None, radius=None):
    radius = int(radius or 16 * u)
    sh = pygame.Surface((rect.w + 12, rect.h + 12), pygame.SRCALPHA)
    pygame.draw.rect(sh, (0, 0, 0, 70), (6, 9, rect.w, rect.h), border_radius=radius)
    surf.blit(sh, (rect.x - 6, rect.y - 6))
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, border, rect, 1, border_radius=radius)


class CardBuilder:
    """Builds one card's content onto its own surface (cached per result)."""

    def __init__(self, width, F, u, title, accent, big=False):
        self.w, self.F, self.u, self.accent, self.big = width, F, u, accent, big
        self.pad = int(16 * u)
        self.ops, self.y = [], self.pad
        dot_r = int(4 * u)
        self.ops.append(("circle", accent, (self.pad + dot_r, self.y + F["label"].get_height() // 2), dot_r))
        self.ops.append(("text", F["label"].render(title.upper(), True, accent), (self.pad + dot_r * 3, self.y)))
        self.y += F["label"].get_linesize() + int(8 * u)

    @property
    def inner(self):
        return self.w - 2 * self.pad

    def para(self, text, font="body", color=TEXT, indent=0, gap=3):
        f = self.F[font]
        for ln in wrap(text, f, self.inner - indent):
            self.ops.append(("text", f.render(ln, True, color), (self.pad + indent, self.y)))
            self.y += f.get_linesize() + gap
        return self

    def space(self, px=8):
        self.y += int(px * self.u)
        return self

    def sub(self, text):
        self.space(4)
        return self.para(text, "small_b", MUTED)

    def bullets(self, items, font="body", color=TEXT, mark="-"):
        f = self.F[font]
        for it in items:
            r = max(2, int(3 * self.u))
            self.ops.append(("circle", self.accent, (self.pad + r, self.y + f.get_height() // 2 + 1), r))
            self.para(it, font, color, indent=int(14 * self.u))
            self.space(2)
        return self

    def numbered(self, steps, font="body"):
        f = self.F[font]
        for s in steps:
            rr = int(f.get_height() * 0.62)
            cy = self.y + f.get_height() // 2 + 1
            self.ops.append(("circle", self.accent, (self.pad + rr, cy), rr))
            self.ops.append(("ctext", self.F["small_b"].render(str(s["step"]), True, BG), (self.pad + rr, cy)))
            indent = rr * 2 + int(10 * self.u)
            if s["title"]:
                self.para(s["title"], "body_b" if font == "body" else "big_b", TEXT, indent=indent, gap=1)
            if s["description"]:
                self.para(s["description"], font if s["title"] else font, MUTED if s["title"] else TEXT, indent=indent)
            self.space(6)
        return self

    def chips(self, chips):
        x = self.pad
        f = self.F["small_b"]
        for text, color in chips:
            img = f.render(text, True, (255, 255, 255))
            rr = pygame.Rect(x, self.y, img.get_width() + int(16 * self.u), img.get_height() + int(8 * self.u))
            if rr.right > self.w - self.pad and x > self.pad:
                x = self.pad
                self.y = rr.bottom + int(6 * self.u)
                rr.topleft = (x, self.y)
            self.ops.append(("chip", rr, color, img))
            x = rr.right + int(6 * self.u)
        self.y += f.get_height() + int(16 * self.u)
        return self

    def meter(self, level):
        u = self.u
        seg_w = (self.inner - int(8 * u)) / 3
        for i, name in enumerate(("Routine", "Prompt", "Emergency")):
            color = URGENCY[name][0]
            on = name == level
            rr = pygame.Rect(int(self.pad + i * (seg_w + 4 * u)), self.y, int(seg_w), int(8 * u))
            self.ops.append(("rect", rr, color if on else mix(color, CARD, 0.75), int(4 * u)))
            lbl = self.F["small_b" if on else "small"].render(URGENCY[name][1], True, TEXT if on else MUTED)
            self.ops.append(("text", lbl, (rr.x, rr.bottom + int(5 * u))))
        self.y += int(8 * u) + self.F["small"].get_linesize() + int(12 * u)
        return self

    def finish(self):
        h = self.y + self.pad - int(4 * self.u)
        surf = pygame.Surface((self.w, h), pygame.SRCALPHA)
        for op in self.ops:
            kind = op[0]
            if kind == "text":
                surf.blit(op[1], op[2])
            elif kind == "ctext":
                surf.blit(op[1], op[1].get_rect(center=op[2]))
            elif kind == "circle":
                pygame.draw.circle(surf, op[1], op[2], op[3])
            elif kind == "rect":
                pygame.draw.rect(surf, op[2], op[1], border_radius=op[3])
            elif kind == "chip":
                pygame.draw.rect(surf, op[2], op[1], border_radius=op[1].h // 2)
                surf.blit(op[3], op[3].get_rect(center=op[1].center))
        return surf


def build_result_cards(r, width, F, u, photo_surf=None):
    """Returns a list of (surface, accent) cards in reading order. Irrelevant sections are left out."""
    ucol = URGENCY[r["urgency"]][0]
    emergency = r["kind"] == "emergency"
    cards = []

    c = CardBuilder(width, F, u, "Reported concern" if emergency else "Identified problem", ucol)
    c.para(r["concern"], "title", TEXT)
    c.space(6)
    chips = [(URGENCY[r["urgency"]][1], ucol), (f"Confidence: {r['confidence']}", (70, 78, 110))]
    if r["kind"] == "skin":
        chips.append(("Dermatological check", (88, 86, 214)))
    elif r["kind"] in ("photo", "not_health"):
        chips.append(("From your photo", (40, 100, 160)))
    elif r["kind"] == "symptoms":
        chips.append(("From your symptoms", (40, 100, 160)))
    c.chips(chips)
    if emergency and r["first_instruction"]:
        c.para("Most important right now", "small_b", (255, 190, 185))
        c.para(r["first_instruction"], "big_b", (255, 255, 255))
    cards.append((c.finish(), ucol))

    if photo_surf is not None:
        pc = CardBuilder(width, F, u, "Your photo", BLUE)
        cards.append((pc.finish(), BLUE, photo_surf))

    if r["assessment"] or r["visible_features"] or r["possible_conditions"]:
        a = CardBuilder(width, F, u, "AI assessment", TEAL)
        if r["assessment"]:
            a.para(r["assessment"], "body", TEXT)
        if r["visible_features"]:
            a.sub("What I can see")
            a.bullets(r["visible_features"], "body")
        if r["possible_conditions"]:
            a.sub("Possible conditions - and why")
            for cond in r["possible_conditions"]:
                a.para(cond["name"], "body_b", TEXT, indent=int(2 * u), gap=1)
                if cond["why"]:
                    a.para(cond["why"], "small", MUTED, indent=int(2 * u))
                a.space(4)
        cards.append((a.finish(), TEAL))

    if r["steps"]:
        s = CardBuilder(width, F, u, "Do this now" if emergency else "Recommended next steps", GREEN if not emergency else RED)
        s.numbered(r["steps"], "big" if emergency else "body")
        cards.append((s.finish(), GREEN if not emergency else RED, "Do this now"))

    if r["precautions"]:
        p = CardBuilder(width, F, u, "Do not" if emergency else "Important precautions", AMBER)
        p.bullets(r["precautions"], "big" if emergency else "body")
        cards.append((p.finish(), AMBER, "Do not"))

    if r["specialist"]:
        sp = CardBuilder(width, F, u, "Get help from" if emergency else "Specialist recommendation", (191, 90, 242))
        sp.para(r["specialist"]["name"], "big_b", TEXT)
        if r["specialist"]["reason"]:
            sp.para(r["specialist"]["reason"], "body", MUTED)
        cards.append((sp.finish(), (191, 90, 242), "Get help from"))

    ug = CardBuilder(width, F, u, "Urgency level", ucol)
    ug.meter(r["urgency"])
    if r["urgency_reason"]:
        ug.para(r["urgency_reason"], "body", TEXT)
    if r["call_now_if"]:
        ug.space(4)
        ug.para("Call emergency services immediately if:", "small_b", (255, 190, 185))
        ug.para(r["call_now_if"], "body", TEXT)
    if not emergency and r["urgency"] != "Routine":
        ug.space(4)
        if r["contacts"]:
            ug.para(f"Emergency numbers ({r['country']}): " + "  |  ".join(n for n, _ in r["contacts"]), "body_b", TEXT)
        else:
            ug.para("In an emergency, call your local emergency number.", "body_b", TEXT)
    cards.append((ug.finish(), ucol, "Urgency level"))

    if r["limitations"]:
        lm = CardBuilder(width, F, u, "Limitations", MUTED)
        lm.para(r["limitations"], "small", MUTED)
        cards.append((lm.finish(), MUTED))
    if emergency:  # under stress: what to do comes straight after the concern
        order = {"Do this now": 1, "Do not": 2, "Get help from": 3, "Urgency level": 4}
        titles = ["Reported concern"] + [None] * (len(cards) - 1)
        def rank(i):
            return 0 if i == 0 else order.get(cards[i][-1] if isinstance(cards[i][-1], str) else "", 5)
        cards = [cards[i] for i in sorted(range(len(cards)), key=rank)]
    return [c[:-1] if isinstance(c[-1], str) else c for c in cards]


class Renderer:
    def __init__(self, screen, app, camera):
        self.screen, self.app, self.camera = screen, app, camera
        self.W, self.H = screen.get_size()
        W, H = self.W, self.H
        self.narrow = W < H * 1.05
        self.u = max(min(W / 800, H / 480), min(W / 480, H / 800)) if self.narrow else min(W / 800, H / 480)
        self.u = max(0.75, self.u)
        u = self.u
        names = "segoeui,sfprodisplay,helveticaneue,dejavusans,arial"

        def font(size, bold=False):
            return pygame.font.SysFont(names, max(10, int(size * u)), bold=bold)

        self.F = {"label": font(11, True), "small": font(12.5), "small_b": font(12.5, True),
                  "body": font(14.5), "body_b": font(14.5, True), "big": font(17), "big_b": font(17, True),
                  "title": font(22, True), "hero": font(30, True), "num": font(34, True)}
        self.face, self.orb = Face(), Orb()
        self.flash = pygame.Surface((W, H))
        self.flash.fill((255, 255, 255))
        self.fade = pygame.Surface((W, H))
        self.fade.fill(BG)
        self.frozen_src, self.frozen_surf = None, None
        self.cards_key, self.cards = None, []
        self.scroll, self.content_h, self.scroll_rect = 0, 0, None
        self.buttons = []
        self.preview_rect = None
        self.last_screen, self.screen_t = None, 0
        self.photo_cache = {}

    # ---------- generic pieces ----------
    def button(self, rect, label, action, color, filled=True, font="body_b", sub=None):
        scr, u = self.screen, self.u
        rect = pygame.Rect(rect)
        hover = rect.collidepoint(pygame.mouse.get_pos()) if pygame.mouse.get_focused() else False
        base = color if filled else CARD_HI
        if hover:
            base = mix(base, (255, 255, 255), 0.12)
        pygame.draw.rect(scr, base, rect, border_radius=min(rect.h // 2, int(14 * u)))
        if not filled:
            pygame.draw.rect(scr, mix(color, CARD_HI, 0.3), rect, 1, border_radius=min(rect.h // 2, int(14 * u)))
        img = self.F[font].render(label, True, (255, 255, 255) if filled else TEXT)
        if sub:
            simg = self.F["small"].render(sub, True, (235, 235, 245) if filled else MUTED)
            total = img.get_height() + simg.get_height()
            scr.blit(img, img.get_rect(midtop=(rect.centerx, rect.centery - total // 2)))
            scr.blit(simg, simg.get_rect(midtop=(rect.centerx, rect.centery - total // 2 + img.get_height())))
        else:
            scr.blit(img, img.get_rect(center=rect.center))
        self.buttons.append((rect, action))
        return rect

    def button_row(self, specs, y, h=None):
        """specs: (label, action, color, filled). Centered, wraps on narrow screens."""
        u, W = self.u, self.W
        h = h or int(38 * u)
        f = self.F["body_b"]
        widths = [f.size(s[0])[0] + int(30 * u) for s in specs]
        gap = int(8 * u)
        rows, cur, cw = [], [], 0
        for s, w in zip(specs, widths):
            if cur and cw + w + gap > W - 24 * u:
                rows.append((cur, cw))
                cur, cw = [], 0
            cur.append((s, w))
            cw += w + (gap if cur[:-1] else 0)
        if cur:
            rows.append((cur, cw))
        y0 = y - (len(rows) - 1) * (h + gap)
        for i, (row, cw) in enumerate(rows):
            x = W / 2 - cw / 2
            for (label, action, color, filled), w in row:
                self.button((int(x), int(y0 + i * (h + gap)), w, h), label, action, color, filled)
                x += w + gap
        return y0

    def wave(self, x, y, w, h, color):
        """Real audio-driven waveform: Lumi's voice envelope, or the live mic level."""
        app, scr = self.app, self.screen
        now = time.time()
        n = 28
        bw = max(2, int(w / n * 0.55))
        for i in range(n):
            if app.busy == "speaking" and app.speak_env is not None:
                k = int((now - app.speak_t0) * 30) - (n - 1 - i)
                v = float(app.speak_env[k]) if 0 <= k < len(app.speak_env) else 0.0
            elif app.hearing or app.mode in ("listening", "answering"):
                v = app.mic_level * (0.4 + 0.6 * abs(math.sin(i * 0.7 + now * 7)))
            else:
                v = 0.0
            bh = max(2, int(h * (0.08 + 0.92 * min(1.0, v))))
            bx = int(x + i * w / n)
            pygame.draw.rect(scr, color if v > 0.02 else mix(color, BG, 0.7),
                             (bx, int(y + (h - bh) / 2), bw, bh), border_radius=bw // 2)

    def status_chip(self, center):
        app = self.app
        label, color = {"speaking": ("Lumi is speaking", (255, 160, 215)),
                        "thinking": ("Analyzing...", (185, 155, 255))}.get(app.busy, (None, None))
        if not label:
            if app.hearing:
                label, color = "Hearing you...", (110, 255, 190)
            elif app.mode in ("listening", "answering"):
                label, color = "Listening", (110, 255, 190)
            elif app.camera_opening:
                label, color = "Opening camera...", TEAL
            else:
                label, color = "Ready", MUTED
        return pill(self.screen, self.F["small_b"], label, BG if color != MUTED else TEXT,
                    color if color != MUTED else CARD_HI, center)

    def top_bar(self, title=None, accent=TEAL):
        app, scr, u, W, F = self.app, self.screen, self.u, self.W, self.F
        dot = GREEN if app.mic_ok and not app.mic_silent else RED
        pygame.draw.circle(scr, dot, (int(18 * u), int(18 * u)), int(5 * u))
        scr.blit(F["small"].render("Mic ready" if app.mic_ok and not app.mic_silent else
                                   ("Mic silent" if app.mic_silent else "Mic not ready"), True, MUTED),
                 (int(28 * u), int(10 * u)))
        if title:
            img = F["big_b"].render(title, True, TEXT)
            scr.blit(img, img.get_rect(midtop=(W // 2, int(8 * u))))
        y = int(40 * u)
        msgs = []
        if app.mic_silent:
            msgs.append(("Microphone is silent - turn up input volume in Settings > System > Sound > Input", AMBER, BG))
        if app.error and time.time() < app.error_until:
            msgs.append((app.error, RED, (255, 255, 255)))
        if app.notice and time.time() < app.notice_until:
            msgs.append((app.notice, BLUE, (255, 255, 255)))
        for text, bg, fg in msgs:
            for line in wrap(text, F["small_b"], W * 0.86)[:2]:
                r = pill(scr, F["small_b"], line, fg, bg, (W / 2, y + 10 * u))
                y = r.bottom + int(4 * u)
        return y

    def typing_box(self, y):
        app, scr, u, W, F = self.app, self.screen, self.u, self.W, self.F
        box = pygame.Rect(int(W * 0.06), int(y), int(W * 0.88), int(44 * u))
        shadowed_card(scr, box, u, CARD_HI, border=TEAL, radius=12 * u)
        shown = app.typed_text + ("|" if int(time.time() * 2) % 2 else " ")
        img = F["body"].render(shown, True, TEXT)
        maxw = box.w - int(24 * u)
        if img.get_width() > maxw:
            img = img.subsurface((img.get_width() - maxw, 0, maxw, img.get_height()))
        scr.blit(img, (box.x + int(12 * u), box.centery - img.get_height() // 2))
        if not app.typed_text:
            ph = F["body"].render("Describe your symptoms...", True, MUTED)
            scr.blit(ph, (box.x + int(16 * u), box.centery - ph.get_height() // 2))
        return box.bottom

    def photo_surface(self, rgb, size):
        img = pygame.image.frombuffer(np.ascontiguousarray(rgb).tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        return pygame.transform.smoothscale(img, size)

    def fit(self, w, h, maxw, maxh):
        s = min(maxw / w, maxh / h)
        return int(w * s), int(h * s)

    def scroll_by(self, dy):
        if self.scroll_rect:
            self.scroll = int(min(max(0, self.scroll - dy * 50 * self.u), max(0, self.content_h - self.scroll_rect.h)))

    # ---------- main ----------
    def draw(self, now, dt):
        app, scr = self.app, self.screen
        self.buttons = []
        self.preview_rect = None
        state = visual_state(app, now)
        self.face.update(app, state, dt, now)
        screen = current_screen(app, self.camera)
        if screen != self.last_screen:
            self.last_screen, self.screen_t, self.scroll = screen, now, 0
        scr.fill(BG)
        getattr(self, f"draw_{screen}")(now, dt, state)
        if app.confirm:
            self.draw_confirm()
        fa = 1 - (now - app.flash_t) / 0.35
        if fa > 0:
            self.flash.set_alpha(int(220 * fa))
            scr.blit(self.flash, (0, 0))
        fade_in = 1 - (now - self.screen_t) / 0.3
        fade_out = (now - app.leaving_t) / 0.55 if now - app.leaving_t < 0.8 else 0
        a = max(fade_in, min(1.0, fade_out))
        if a > 0:
            self.fade.set_alpha(int(255 * min(1, a)))
            scr.blit(self.fade, (0, 0))

    # ---------- HOME ----------
    def draw_home(self, now, dt, state):
        app, scr, u, W, H, F = self.app, self.screen, self.u, self.W, self.H, self.F
        top = self.top_bar()
        col = tuple(int(c) for c in self.face.color)
        tiles_h = int(84 * u)
        tiles_total = tiles_h * 3 + int(20 * u) if self.narrow else tiles_h
        foot = int(54 * u)
        text_h = F["hero"].get_linesize() + 2 * F["big"].get_linesize() + int(30 * u)
        avail = H - foot - tiles_total - int(16 * u) - top - text_h
        R = max(34 * u, min(118 * u, avail / 2.8))
        cy = top + R * 1.35
        self.orb.draw(scr, app, self.face, state, W / 2, cy, R, now, dt, u)
        y = cy + R * 1.45
        if app.mode in ("idle", "sleeping") and not app.busy:
            y = text_block(scr, "Hi, I'm Lumi", F["hero"], TEXT, 0, y, W, 1, center=True)
        y = text_block(scr, app.status, F["big"], col if app.mode != "idle" else MUTED, W * 0.08, y + 2 * u,
                       W * 0.84, 2, center=True)
        if app.typing:
            y = self.typing_box(y + 8 * u)
        elif app.busy == "speaking" and app.caption:
            self.wave(W / 2 - 70 * u, y + 4 * u, 140 * u, 18 * u, (255, 160, 215))
            y = text_block(scr, app.caption, F["body"], TEXT, W * 0.1, y + 28 * u, W * 0.8, 2, center=True)
        elif app.mode == "answering" and app.questions:
            for q in app.questions[:2]:
                y = text_block(scr, q, F["body_b"], TEXT, W * 0.1, y + 2 * u, W * 0.8, 2, center=True)
        elif app.heard and app.mode in ("listening", "answering", "idle") and app.busy != "thinking":
            y = text_block(scr, f'"{app.heard}"', F["body"], MUTED, W * 0.1, y + 2 * u, W * 0.8, 1, center=True)

        # three mode tiles
        ty = H - foot - tiles_total - int(8 * u)
        tiles = [("Diagnosis Lumi", "Describe your symptoms", "wake", TEAL),
                 ("Camera Lumi", "Check a photo or skin", "camera", BLUE),
                 ("Emergency Lumi", "Urgent help, right now", "emergency", RED)]
        m = int(16 * u)
        if self.narrow:
            rects = [pygame.Rect(m, ty + i * (tiles_h + int(10 * u)), W - 2 * m, tiles_h) for i in range(3)]
        else:
            tw = (W - 4 * m) // 3
            rects = [pygame.Rect(m + i * (tw + m), ty, tw, tiles_h) for i in range(3)]
        hints = {"wake": "Say \"Wake up Lumi\"  |  W", "camera": "\"Camera Lumi\"  |  C", "emergency": "\"Emergency Lumi\"  |  E"}
        for (title, sub, action, color), r in zip(tiles, rects):
            hover = r.collidepoint(pygame.mouse.get_pos()) if pygame.mouse.get_focused() else False
            shadowed_card(scr, r, u, CARD_HI if hover else CARD, border=mix(color, CARD, 0.55))
            pygame.draw.rect(scr, color, (r.x, r.y + int(14 * u), int(4 * u), r.h - int(28 * u)), border_radius=int(2 * u))
            scr.blit(F["big_b"].render(title, True, TEXT), (r.x + int(18 * u), r.y + int(14 * u)))
            scr.blit(F["body"].render(sub, True, MUTED), (r.x + int(18 * u), r.y + int(14 * u) + F["big_b"].get_linesize()))
            scr.blit(F["small"].render(hints[action], True, mix(color, TEXT, 0.35)),
                     (r.x + int(18 * u), r.bottom - int(10 * u) - F["small"].get_height()))
            self.buttons.append((r, action))
        # footer
        if app.typing:
            specs = [("Send", "send", TEAL, True), ("Cancel", "cancel_type", LINE, False)]
        else:
            specs = [("Type symptoms (T)", "type", LINE, False), ("Upload photo (U)", "upload", LINE, False)]
            if app.busy == "speaking":
                specs.insert(0, ("Stop (S)", "stop", AMBER, True))
            if app.result:
                specs.append(("Last result (V)", "view_result", LINE, False))
        self.button_row(specs, H - foot + int(8 * u), int(36 * u))

    # ---------- CAMERA ----------
    def draw_camera(self, now, dt, state):
        app, scr, u, W, H, F = self.app, self.screen, self.u, self.W, self.H, self.F
        top = self.top_bar("Camera Lumi")
        foot = int(60 * u)
        self.orb.draw(scr, app, self.face, state, int(34 * u), int(64 * u), 22 * u, now, dt, u, particles=False)
        self.status_chip((W - 90 * u, 64 * u))
        area = pygame.Rect(int(16 * u), int(top + 36 * u), W - int(32 * u), H - foot - int(top + 36 * u) - int(56 * u))
        frame = self.camera.get_frame() if self.camera.active else None
        fw, fh = (frame.shape[1], frame.shape[0]) if frame is not None else (640, 480)
        pw, ph = self.fit(fw, fh, area.w, area.h)
        rect = pygame.Rect(0, 0, pw, ph)
        rect.center = area.center
        self.preview_rect = rect
        hold = self.camera.hold
        border = GREEN if (hold > 0 or now - app.gesture_t < 1.5) else TEAL
        shadowed_card(scr, rect.inflate(int(8 * u), int(8 * u)), u, border, radius=18 * u)
        if frame is not None:
            scr.blit(self.photo_surface(frame, rect.size), rect)
            pill(scr, F["small_b"], "LIVE", (255, 255, 255), RED, (rect.x + 34 * u, rect.y + 20 * u))
        else:
            pygame.draw.rect(scr, CORE, rect, border_radius=int(14 * u))
            text_block(scr, "Opening camera..." if app.camera_opening else "Camera unavailable", F["title"], TEAL,
                       rect.x, rect.centery - 16 * u, rect.w, 1, center=True)
        # gesture status
        if self.camera.recognizer is None:
            g = "Gesture AI off - press Capture"
        elif self.camera.armed_at > time.time():
            g = "Gesture AI starting..."
        else:
            name, score = self.camera.gesture
            g = (f"Gesture AI on  |  {name.replace('_', ' ')} {int(score * 100)}%" if name and name != "None"
                 else "Gesture AI on  |  hold a thumbs-up to take the photo")
        gy = rect.bottom + int(12 * u)
        text_block(scr, g, F["body"], MUTED, 0, gy, W, 1, center=True)
        if hold > 0:
            pb = pygame.Rect(int(W / 2 - 80 * u), int(gy + 24 * u), int(160 * u), int(6 * u))
            pygame.draw.rect(scr, LINE, pb, border_radius=3)
            pygame.draw.rect(scr, GREEN, (pb.x, pb.y, pb.w * hold, pb.h), border_radius=3)
        if now - app.gesture_t < 1.6:
            pill(scr, F["title"], "THUMBS UP DETECTED", (5, 40, 15), GREEN, (rect.centerx, rect.y + 56 * u))
        specs = [("Capture (Space)", "capture", GREEN, True), ("Upload (U)", "upload", LINE, False),
                 ("Close (X)", "close", LINE, False)]
        if app.busy == "speaking":
            specs.append(("Stop (S)", "stop", AMBER, True))
        self.button_row(specs, H - foot + int(10 * u))

    # ---------- REVIEW (retake / analyze) ----------
    def draw_review(self, now, dt, state):
        app, scr, u, W, H, F = self.app, self.screen, self.u, self.W, self.H, self.F
        rv = app.review
        top = self.top_bar("Emergency photo" if rv and rv["mode"] == "emergency" else "Camera Lumi")
        self.orb.draw(scr, app, self.face, state, int(34 * u), int(64 * u), 22 * u, now, dt, u, particles=False)
        self.status_chip((W - 90 * u, 64 * u))
        foot = int(60 * u)
        area = pygame.Rect(int(16 * u), int(top + 36 * u), W - int(32 * u), H - foot - int(top + 36 * u) - int(70 * u))
        img = app.frozen_rgb
        if img is None:
            return
        pw, ph = self.fit(img.shape[1], img.shape[0], area.w, area.h)
        rect = pygame.Rect(0, 0, pw, ph)
        rect.center = area.center
        if self.frozen_src is not img or self.frozen_surf.get_size() != rect.size:
            self.frozen_src, self.frozen_surf = img, self.photo_surface(img, rect.size)
        k = min(1.0, (now - app.flash_t) / 0.4)
        s = 1.06 - 0.06 * (1 - (1 - k) ** 3)
        shadowed_card(scr, rect.inflate(int(8 * u), int(8 * u)), u, CARD_HI, radius=18 * u)
        surf = pygame.transform.smoothscale(self.frozen_surf, (int(rect.w * s), int(rect.h * s)))
        scr.set_clip(rect)
        scr.blit(surf, surf.get_rect(center=rect.center))
        scr.set_clip(None)
        analyzing = rv["analyzing"]
        if analyzing:  # loading: dim photo + scanning line + spinner
            dim = pygame.Surface(rect.size, pygame.SRCALPHA)
            dim.fill((9, 11, 22, 120))
            scr.blit(dim, rect)
            sy = rect.y + int((math.sin(now * 2.2) * 0.5 + 0.5) * rect.h)
            pygame.draw.line(scr, TEAL, (rect.x, sy), (rect.right, sy), max(2, int(2 * u)))
            for i in range(8):
                a = now * 5 + i * math.pi / 4
                pygame.draw.circle(scr, mix(TEAL, BG, i / 8), (int(rect.centerx + math.cos(a) * 22 * u),
                                                              int(rect.centery + math.sin(a) * 22 * u)), int(4 * u))
            text_block(scr, "Analyzing your photo with AI...", F["big_b"], TEXT, 0, rect.bottom + 12 * u, W, 1, center=True)
        else:
            left = max(0, app.review_deadline - now) if app.review_deadline else None
            msg = "Check the photo is clear and close." + (f"  Analyzing in {math.ceil(left)}s..." if left else "")
            text_block(scr, msg, F["big"], MUTED, 0, rect.bottom + 12 * u, W, 1, center=True)
            specs = [("Analyze photo (Enter)", "analyze_photo", GREEN, True),
                     ("Retake (Backspace)", "retake", LINE, False)]
            self.button_row(specs, H - foot + int(10 * u))

    # ---------- RESULT ----------
    def layout_cards(self, cards, area, cols):
        """Masonry layout; returns [(surface/None, rect, extra)] and total height."""
        gap = int(14 * self.u)
        colw = (area.w - gap * (cols - 1)) // cols
        heights = [0] * cols
        placed = []
        for item in cards:
            surf = item[0]
            extra = item[2] if len(item) > 2 else None
            h = surf.get_height() + (extra.get_height() + int(14 * self.u) if extra is not None else 0)
            ci = heights.index(min(heights))
            rect = pygame.Rect(area.x + ci * (colw + gap), area.y + heights[ci], colw, h)
            placed.append((surf, rect, extra, item[1]))
            heights[ci] += h + gap
        return placed, max(heights)

    def ensure_cards(self, r, colw):
        key = (self.app.result_id, colw)
        if key != self.cards_key:
            photo = None
            if r.get("photo") and os.path.exists(r["photo"]):
                try:
                    rgb = load_image_rgb(r["photo"])
                    pw, ph = self.fit(rgb.shape[1], rgb.shape[0], colw - int(32 * self.u), 260 * self.u)
                    photo = self.photo_surface(rgb, (pw, ph))
                except Exception:
                    photo = None
            self.cards = build_result_cards(r, colw, self.F, self.u, photo)
            self.cards_key = key

    def draw_cards(self, area, cols):
        scr, u = self.screen, self.u
        gap = int(14 * u)
        colw = (area.w - gap * (cols - 1)) // cols
        self.ensure_cards(self.app.result, colw)
        placed, total = self.layout_cards(self.cards, pygame.Rect(0, 0, area.w, 0), cols)
        self.scroll_rect, self.content_h = area, total
        self.scroll = min(self.scroll, max(0, total - area.h))
        scr.set_clip(area)
        for surf, rect, extra, accent in placed:
            rr = rect.move(area.x, area.y - self.scroll)
            if rr.bottom < area.y or rr.y > area.bottom:
                continue
            shadowed_card(scr, rr, u, CARD, border=LINE)
            pygame.draw.rect(scr, accent, (rr.x + int(16 * u), rr.y, int(36 * u), int(3 * u)), border_radius=int(2 * u))
            scr.blit(surf, rr.topleft)
            if extra is not None:
                scr.blit(extra, extra.get_rect(midtop=(rr.centerx, rr.y + surf.get_height() - int(6 * u))))
        scr.set_clip(None)
        if total > area.h:
            frac = area.h / total
            bh = max(20 * u, area.h * frac)
            by = area.y + (area.h - bh) * (self.scroll / max(1, total - area.h))
            pygame.draw.rect(scr, LINE, (area.right + int(4 * u), by, int(4 * u), bh), border_radius=int(2 * u))

    def draw_result(self, now, dt, state):
        app, scr, u, W, H, F = self.app, self.screen, self.u, self.W, self.H, self.F
        r = app.result
        ucol = URGENCY[r["urgency"]][0]
        top = self.top_bar(KIND_TITLE.get(r["kind"], "Lumi"))
        # header: small orb + live voice waveform + status
        hy = top + int(30 * u)
        self.orb.draw(scr, app, self.face, state, int(40 * u), hy, 24 * u, now, dt, u, particles=False)
        chip = self.status_chip((int(150 * u), hy))
        self.wave(chip.right + 12 * u, hy - 10 * u, min(220 * u, W * 0.3), 20 * u, (255, 160, 215))
        pill(scr, F["small_b"], URGENCY[r["urgency"]][1], BG, ucol, (W - 90 * u, hy))
        foot = int(62 * u)
        cap_h = int(30 * u) if app.busy == "speaking" and app.caption else 0
        area = pygame.Rect(int(16 * u), int(hy + 34 * u), W - int(40 * u), H - foot - int(hy + 34 * u) - cap_h - int(6 * u))
        self.draw_cards(area, 1 if self.narrow or W < 760 * u else 2)
        if cap_h:
            text_block(scr, app.caption, F["body"], TEXT, W * 0.06, area.bottom + 6 * u, W * 0.88, 1, center=True)
        specs = [("Home (H)", "home", TEAL, True)]
        if app.busy == "speaking":
            specs.append(("Stop (S)", "stop", AMBER, True))
        elif app.last_spoken:
            specs.append(("Repeat (R)", "repeat", LINE, False))
        if r["kind"] in ("skin", "photo", "not_health"):
            specs.append(("New photo (C)", "camera", BLUE, False))
        if r["urgency"] != "Routine" and PRIMARY_NUMBER:
            specs.append((f"Call {PRIMARY_NUMBER}", f"call:{PRIMARY_NUMBER}", RED, True))
        self.button_row(specs, H - foot + int(12 * u))

    # ---------- EMERGENCY ----------
    def contacts_lines(self, width):
        F, u = self.F, self.u
        where = f"{COUNTRY} ({COUNTRY_SOURCE})" if COUNTRY else "Location unknown"
        return wrap(f"EMERGENCY CONTACTS  -  {where.upper()}", F["label"], width - 32 * u)

    def contacts_height(self, width):
        F, u = self.F, self.u
        h = int(14 * u) + len(self.contacts_lines(width)) * F["label"].get_linesize() + int(8 * u)
        if CONTACTS:
            for _, purpose in CONTACTS:
                lines = len(wrap(purpose, F["small"], width - 32 * u - 118 * u))
                h += F["num"].get_linesize() + lines * (F["small"].get_linesize() + 2) + int(10 * u)
        else:
            h += int(90 * u)
        return h + int(8 * u)

    def draw_contacts(self, rect):
        """Big, always-visible emergency numbers with Call buttons (each asks for confirmation)."""
        scr, u, F = self.screen, self.u, self.F
        shadowed_card(scr, rect, u, (58, 14, 20), border=RED)
        x, y = rect.x + int(16 * u), rect.y + int(14 * u)
        for line in self.contacts_lines(rect.w):
            scr.blit(F["label"].render(line, True, (255, 190, 185)), (x, y))
            y += F["label"].get_linesize()
        y += int(8 * u)
        if CONTACTS:
            bw = int(104 * u)
            for num, purpose in CONTACTS:
                img = F["num"].render(num, True, (255, 255, 255))
                scr.blit(img, (x, y))
                self.button((rect.right - bw - int(14 * u), y + int(4 * u), bw, int(38 * u)), f"Call {num}",
                            f"call:{num}", RED, True)
                y += F["num"].get_linesize()
                y = text_block(scr, purpose, F["small"], (255, 215, 212), x, y, rect.w - 32 * u - 118 * u) + int(10 * u)
        else:
            y = text_block(scr, "Call your local emergency number now.", F["big_b"], (255, 255, 255), x, y,
                           rect.w - 32 * u)
            y = text_block(scr, "Set LUMI_COUNTRY in lumi.env so Lumi shows the right numbers.", F["small"],
                           (255, 215, 212), x, y + 4 * u, rect.w - 32 * u)
        return y

    def draw_emergency(self, now, dt, state):
        app, scr, u, W, H, F = self.app, self.screen, self.u, self.W, self.H, self.F
        pygame.draw.rect(scr, mix(RED, BG, 0.25 + 0.15 * math.sin(now * 4)), scr.get_rect(), max(3, int(5 * u)))
        # banner
        banner = pygame.Rect(int(12 * u), int(10 * u), W - int(24 * u), int(50 * u))
        pygame.draw.rect(scr, RED, banner, border_radius=int(14 * u))
        scr.blit(F["title"].render("EMERGENCY LUMI", True, (255, 255, 255)), (banner.x + int(16 * u), banner.y + int(6 * u)))
        scr.blit(F["small_b"].render("Emergency mode active", True, (255, 220, 218)),
                 (banner.x + int(16 * u), banner.y + int(6 * u) + F["title"].get_linesize() - int(4 * u)))
        cam = "Camera opening" if app.camera_opening else ("Camera on" if self.camera.active else "Camera off")
        mic = "Mic silent" if app.mic_silent else ("Mic ready" if app.mic_ok else "Mic not ready")
        ai = {"thinking": "AI analyzing", "speaking": "Lumi speaking"}.get(app.busy, "AI ready")
        x = banner.right - int(12 * u)
        for text, ok in ((ai, True), (mic, mic == "Mic ready"), (cam, cam == "Camera on")):
            img = F["small_b"].render(text, True, (255, 255, 255))
            rr = pygame.Rect(0, 0, img.get_width() + int(18 * u), img.get_height() + int(8 * u))
            rr.midright = (x, banner.centery)
            if not self.narrow:
                pygame.draw.rect(scr, (40, 120, 70) if ok else (130, 70, 20), rr, border_radius=rr.h // 2)
                scr.blit(img, img.get_rect(center=rr.center))
            x = rr.x - int(6 * u)
        y = banner.bottom + int(8 * u)
        msgs = []
        if app.mic_silent:
            msgs.append(("Microphone is silent - turn up input volume", AMBER, BG))
        if app.error and time.time() < app.error_until:
            msgs.append((app.error, (150, 30, 40), (255, 255, 255)))
        if app.notice and time.time() < app.notice_until:
            msgs.append((app.notice, BLUE, (255, 255, 255)))
        for text, bg, fg in msgs:
            for line in wrap(text, F["small_b"], W * 0.86)[:2]:
                rr = pill(scr, F["small_b"], line, fg, bg, (W / 2, y + 10 * u))
                y = rr.bottom + int(4 * u)
        foot = int(62 * u) if not self.narrow else int(104 * u)
        m = int(14 * u)
        body = pygame.Rect(m, int(y + 4 * u), W - 2 * m, H - foot - int(y + 4 * u) - int(34 * u))

        if self.narrow:
            contacts = pygame.Rect(body.x, body.y, body.w, self.contacts_height(body.w))
            self.draw_contacts(contacts)
            left = pygame.Rect(body.x, contacts.bottom + m, body.w, body.bottom - contacts.bottom - m)
            right = None
        else:
            rw = int(body.w * 0.40)
            left = pygame.Rect(body.x, body.y, body.w - rw - m, body.h)
            right = pygame.Rect(left.right + m, body.y, rw, body.h)
            contacts = pygame.Rect(right.x, right.y, right.w, self.contacts_height(right.w))
            self.draw_contacts(contacts)
            cam_area = pygame.Rect(right.x, contacts.bottom + m, right.w, right.bottom - contacts.bottom - m)
            self.draw_emergency_camera(cam_area, now)

        if app.result:
            self.draw_cards(left, 1)
        else:
            shadowed_card(scr, left, u, CARD, border=LINE)
            yy = text_block(scr, "Tell me what happened.", F["title"], TEXT, left.x + 18 * u, left.y + 16 * u, left.w - 36 * u)
            yy = text_block(scr, "Speak now, press Type, or show the injury and give a thumbs-up. You do not need a photo.",
                            F["body"], MUTED, left.x + 18 * u, yy + 4 * u, left.w - 36 * u)
            yy += int(14 * u)
            scr.blit(F["label"].render("GENERAL SAFETY GUIDANCE", True, AMBER), (left.x + 18 * u, yy))
            yy += F["label"].get_linesize() + int(6 * u)
            for line in ("If someone is not breathing, unconscious, bleeding heavily or has chest pain, call now.",
                         "Stay with the person and keep them still and warm.",
                         "Press firmly on heavy bleeding with a clean cloth.",
                         "Do not give food or drink to someone who may need surgery."):
                pygame.draw.circle(scr, AMBER, (int(left.x + 24 * u), int(yy + F["big"].get_height() / 2)), int(4 * u))
                yy = text_block(scr, line, F["big"], TEXT, left.x + 36 * u, yy, left.w - 56 * u) + int(6 * u)
            if app.busy == "thinking":
                text_block(scr, "Getting emergency guidance from AI...", F["body_b"], (185, 155, 255),
                           left.x + 18 * u, left.bottom - 34 * u, left.w - 36 * u, 1)
        # caption / typing
        cy = body.bottom + int(6 * u)
        if app.typing:
            self.typing_box(cy - 12 * u)
        elif app.busy == "speaking" and app.caption:
            self.wave(W / 2 - 60 * u, cy, 120 * u, 14 * u, (255, 160, 215))
            text_block(scr, app.caption, F["body"], TEXT, W * 0.06, cy + 16 * u, W * 0.88, 1, center=True)
        else:
            text_block(scr, app.status, F["body_b"], (255, 200, 196), W * 0.06, cy + 4 * u, W * 0.88, 1, center=True)
        # buttons
        if app.typing:
            specs = [("Send", "send", RED, True), ("Cancel", "cancel_type", LINE, False)]
        else:
            specs = [("Type (T)", "type", LINE, False)]
            if self.camera.active and app.review is None:
                specs.insert(0, ("Capture (Space)", "capture", GREEN, True))
            elif not self.camera.active and not app.camera_opening:
                specs.insert(0, ("Camera (C)", "camera", BLUE, False))
            specs.append(("Upload (U)", "upload", LINE, False))
            if app.busy == "speaking":
                specs.append(("Stop (S)", "stop", AMBER, True))
            elif app.last_spoken:
                specs.append(("Repeat (R)", "repeat", LINE, False))
            if app.video_status:
                specs.append(("First-aid video (Y)", "video", RED, False))
            specs.append(("I'm safe - exit (H)", "exit_emergency", (60, 160, 90), True))
        self.button_row(specs, H - int(52 * u))

    def draw_emergency_camera(self, area, now):
        app, scr, u, F = self.app, self.screen, self.u, self.F
        if area.h < 60 * u:
            return
        img = app.frozen_rgb if app.frozen_rgb is not None else (self.camera.get_frame() if self.camera.active else None)
        if img is None:
            shadowed_card(scr, area, u, CARD, border=LINE)
            msg = "Opening camera..." if app.camera_opening else "Camera off - press Camera or Upload"
            text_block(scr, msg, F["body_b"], MUTED, area.x, area.centery - 10 * u, area.w, 1, center=True)
            return
        pw, ph = self.fit(img.shape[1], img.shape[0], area.w, area.h - 26 * u)
        rect = pygame.Rect(0, 0, pw, ph)
        rect.midtop = (area.centerx, area.y)
        self.preview_rect = rect if app.frozen_rgb is None else None
        hold = self.camera.hold
        border = GREEN if hold > 0 or now - app.gesture_t < 1.5 else RED
        pygame.draw.rect(scr, border, rect.inflate(int(6 * u), int(6 * u)), border_radius=int(14 * u))
        scr.blit(self.photo_surface(img, rect.size), rect)
        if app.frozen_rgb is not None:
            pill(scr, F["small_b"], "ANALYZING" if app.busy == "thinking" else "CAPTURED", BG, (255, 255, 255),
                 (rect.x + 50 * u, rect.y + 18 * u))
        else:
            pill(scr, F["small_b"], "LIVE", (255, 255, 255), RED, (rect.x + 30 * u, rect.y + 18 * u))
        name, score = self.camera.gesture
        g = ("Gesture AI off" if self.camera.recognizer is None else
             f"{name.replace('_', ' ')} {int(score * 100)}%" if name and name != "None" else "Thumbs-up to capture")
        text_block(scr, g, F["small"], MUTED, area.x, rect.bottom + 6 * u, area.w, 1, center=True)
        if now - app.gesture_t < 1.6:
            pill(scr, F["body_b"], "THUMBS UP DETECTED", (5, 40, 15), GREEN, (rect.centerx, rect.y + 44 * u))

    def draw_confirm(self):
        app, scr, u, W, H, F = self.app, self.screen, self.u, self.W, self.H, self.F
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 160))
        scr.blit(dim, (0, 0))
        self.buttons = []
        box = pygame.Rect(0, 0, int(min(W * 0.86, 420 * u)), int(170 * u))
        box.center = (W // 2, H // 2)
        shadowed_card(scr, box, u, CARD_HI, border=LINE, radius=20 * u)
        y = text_block(scr, app.confirm["text"], F["title"], TEXT, box.x + 20 * u, box.y + 20 * u, box.w - 40 * u, 2,
                       center=True)
        text_block(scr, app.confirm.get("sub", ""), F["small"], MUTED, box.x + 20 * u, y + 4 * u, box.w - 40 * u, 2,
                   center=True)
        bw = (box.w - int(60 * u)) // 2
        self.button((box.x + int(20 * u), box.bottom - int(56 * u), bw, int(40 * u)), app.confirm["yes_label"],
                    "confirm_yes", RED, True)
        self.button((box.right - int(20 * u) - bw, box.bottom - int(56 * u), bw, int(40 * u)), "Cancel",
                    "confirm_no", LINE, False)


# ============================================================================
# SELF TEST (real devices + real APIs)
# ============================================================================
def run_selftest(mic):
    results = []

    def report(name, ok, detail=""):
        results.append((name, ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    report("API keys", bool(DEEPGRAM_API_KEYS and GROQ_API_KEY),
           f"{len(DEEPGRAM_API_KEYS)} Deepgram key(s), Groq key {'set' if GROQ_API_KEY else 'MISSING'}")
    report("Emergency numbers", True, f"{COUNTRY or 'unknown country'} ({COUNTRY_SOURCE}): "
                                      f"{', '.join(n for n, _ in CONTACTS) or 'generic guidance'}")
    try:
        stream, sr, block, name = open_mic_stream(mic)
        chunks = []
        with stream:
            for _ in range(int(2.5 * sr / block)):
                chunks.append(stream.read(block)[0][:, 0].copy())
        peak = int(np.abs(np.concatenate(chunks).astype(np.int32)).max())
        report("Microphone capture", peak > 40, f"{name}: peak {peak} (0-5 = silent / volume 0%)")
    except Exception as e:
        report("Microphone capture", False, str(e))
    wav, err = text_to_speech("Wake up Lumi.")
    report("Deepgram text-to-speech", wav is not None, err or f"{len(wav)} bytes")
    if wav:
        text, prov, err = speech_to_text(wav)
        report("Speech-to-text + command", detect_command(normalize(text)) == "wake", f"{prov}: {text!r}")
    try:
        r = analyze_symptoms("Headache, fever and dizziness for two days.")
        report("Groq symptom assessment", True, f"{r['concern']} | {len(r['steps'])} steps | "
                                                f"{len(r['precautions'])} precautions | {r['urgency']}")
    except ApiError as e:
        report("Groq symptom assessment", False, str(e))
    app = App()
    cam = CameraWorker(app)
    ok, err = cam.start("standard")
    frame = None
    t = time.time()
    while ok and time.time() - t < 4 and frame is None:
        frame = cam.get_frame()
        time.sleep(0.1)
    report("Camera live feed", frame is not None, err or (f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else cam.last_open_error))
    report("MediaPipe gesture recognizer", cam.recognizer is not None, cam.recognizer_error or "loaded")
    raw = cam.raw
    cam.stop()
    if raw is not None:
        try:
            r = analyze_image(encode_jpeg(raw))
            report("Groq photo assessment", True, f"{r['kind']} | {r['concern']}")
        except ApiError as e:
            report("Groq photo assessment", False, str(e))
    print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed")


# ============================================================================
# MAIN
# ============================================================================
def pick_photo_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="Choose a photo for Lumi",
                                          filetypes=[("Photos", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        root.destroy()
        return path or None, None
    except Exception as e:
        return None, f"Could not open the file picker ({e}). You can drag a photo onto the Lumi window instead."


def main():
    ap = argparse.ArgumentParser(description="Lumi - HealthThread Beacon")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--size", default="1100x700", help="window size, e.g. 800x480 or 480x800")
    ap.add_argument("--mic", default=os.getenv("LUMI_MIC"))
    ap.add_argument("--list-mics", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.list_mics:
        print(sd.query_devices() if sd else f"sounddevice not available: {SD_IMPORT_ERROR}")
        return
    if args.selftest:
        pygame.init()
        run_selftest(args.mic)
        return

    try:
        pygame.mixer.pre_init(24000, -16, 1, 1024)
    except Exception:
        pass
    pygame.init()
    app = App()
    try:
        pygame.mixer.init()
        app.mixer_ok = True
    except Exception as e:
        log("pygame audio output failed, using fallback players:", e)
    if args.windowed:
        w, h = (int(v) for v in args.size.lower().split("x"))
        screen = pygame.display.set_mode((w, h))
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Lumi")
    if not DEEPGRAM_API_KEYS or not GROQ_API_KEY:
        app.show_error("API keys missing: add DEEPGRAM_API_KEY and GROQ_API_KEY to lumi.env", 30)
    log(f"Emergency numbers: {COUNTRY or 'unknown'} ({COUNTRY_SOURCE})")

    speaker = Speaker(app)
    camera = CameraWorker(app)
    Ear(app, speaker, args.mic).start()
    Brain(app, speaker, camera).start()
    renderer = Renderer(screen, app, camera)
    clock = pygame.time.Clock()
    last = time.time()
    skip_text = False

    def open_video():
        try:
            if webbrowser.open(EMERGENCY_VIDEO, new=2):
                app.video_status = "opened"
        except Exception as e:
            app.show_error(f"Could not open the browser: {e}")

    def start_typing():
        nonlocal skip_text
        speaker.stop()
        app.typing, app.typed_text = True, ""
        skip_text = True
        pygame.key.start_text_input()

    def send_typed():
        text = app.typed_text.strip()
        app.typing, app.typed_text = False, ""
        if text:
            app.events.put(("text", text, True))

    def upload():
        path, err = pick_photo_file()
        if err:
            app.show_error(err)
        if path:
            app.events.put(("upload", path, time.time()))

    def do(action):
        app.nav_id += 1  # any user action cancels a pending automatic return
        if app.leaving_t > 0:
            app.leaving_t = -10
        if action.startswith("call:"):
            num = action.split(":", 1)[1]
            app.confirm = {"text": f"Call {num} now?", "yes_label": f"Call {num}", "yes": ("call", num),
                           "sub": "Lumi will open your phone app. Nothing is dialled until you confirm."}
            return
        simple = {"wake": ("wake",), "camera": ("emergency",) if False else ("camera",), "emergency": ("emergency",),
                  "repeat": ("repeat",), "close": ("close_camera",), "home": ("home",),
                  "analyze_photo": ("analyze_photo",), "retake": ("retake",), "exit_emergency": ("exit_emergency",)}
        if action == "confirm_yes":
            if app.confirm:
                app.events.put(app.confirm["yes"])
            app.confirm = None
        elif action == "confirm_no":
            app.confirm = None
        elif action == "type":
            start_typing()
        elif action == "send":
            send_typed()
        elif action == "cancel_type":
            app.typing, app.typed_text = False, ""
        elif action == "stop":
            speaker.stop()
        elif action == "capture":
            camera.capture()
        elif action == "upload":
            upload()
        elif action == "video":
            open_video()
        elif action == "view_result":
            if app.result:
                app.show_result = True
        elif action == "camera":
            if camera.active or app.camera_opening:
                return
            speaker.stop()
            app.events.put(("camera",))
        elif action in simple:
            if action in ("wake", "emergency", "home", "exit_emergency", "retake"):
                speaker.stop()
            app.events.put(simple[action])

    keymap = {pygame.K_w: "wake", pygame.K_t: "type", pygame.K_c: "camera", pygame.K_e: "emergency",
              pygame.K_s: "stop", pygame.K_r: "repeat", pygame.K_x: "close", pygame.K_y: "video",
              pygame.K_u: "upload", pygame.K_h: "home", pygame.K_v: "view_result"}

    while app.running:
        now = time.time()
        dt, last = min(now - last, 0.1), now
        if app.request_upload:
            app.request_upload = False
            upload()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                app.running = False
            elif ev.type == pygame.DROPFILE:
                app.events.put(("upload", ev.file, time.time()))
            elif app.confirm and ev.type == pygame.KEYDOWN:
                do("confirm_yes" if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER) else
                   "confirm_no" if ev.key == pygame.K_ESCAPE else "")
            elif app.typing and ev.type == pygame.TEXTINPUT:
                if skip_text:
                    skip_text = False
                else:
                    app.typed_text += ev.text
            elif app.typing and ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    send_typed()
                elif ev.key == pygame.K_ESCAPE:
                    app.typing, app.typed_text = False, ""
                elif ev.key == pygame.K_BACKSPACE:
                    app.typed_text = app.typed_text[:-1]
                skip_text = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                screen_now = current_screen(app, camera)
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    app.running = False
                elif screen_now == "review" and k in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    do("analyze_photo")
                elif screen_now == "review" and k == pygame.K_BACKSPACE:
                    do("retake")
                elif k == pygame.K_SPACE:
                    do("capture" if camera.active and app.review is None else "wake")
                elif k == pygame.K_h:
                    do("exit_emergency" if app.emergency else "home")
                elif k in keymap:
                    if keymap[k] == "close" and not camera.active:
                        continue
                    do(keymap[k])
            elif ev.type == pygame.MOUSEWHEEL:
                renderer.scroll_by(ev.y)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                hit = next((a for r, a in reversed(renderer.buttons) if r.collidepoint(ev.pos)), None)
                if hit:
                    do(hit)
                elif renderer.preview_rect and renderer.preview_rect.collidepoint(ev.pos) and camera.active:
                    do("capture")
        if app.request_iconify:
            app.request_iconify = False
            if not args.windowed:
                pygame.display.iconify()
        renderer.draw(now, dt)
        pygame.display.flip()
        clock.tick(30)

    speaker.stop()
    camera.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
