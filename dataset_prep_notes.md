# Dataset Construction Notes — Nigerian Speech Audio

Notes from my attempt to build a small Nigerian speech audio dataset from raw recordings. This is a work-in-progress document.

---

## Motivation

The core problem in Nigerian speech NLP is data. Most publicly available ASR training data is American or British English. Adapting models to Nigerian-accented English — let alone Yoruba, Igbo, Pidgin, or code-switched speech — requires domain-specific data that almost doesn't exist in usable form.

This is my attempt to start building one from scratch, even at a small scale.

---

## Data Source

- **Source type**: Long-form audio recordings of Nigerian speakers (monologue, semi-formal speech)
- **Format**: MP3 / WAV, varying quality (phone microphones, built-in laptop mics, room acoustics)
- **Language mix**: Primarily Nigerian English, with natural Yoruba and Pidgin fragments
- **Speaker diversity**: Limited (small number of speakers) — a known weakness of this dataset

---

## Preprocessing Pipeline

### Step 1 — Format Standardization

All audio converted to:
- **Sample rate**: 16,000 Hz (Whisper's native format)
- **Channels**: Mono
- **Bit depth**: 16-bit PCM WAV

Tool used: `librosa` for loading and resampling, `soundfile` for writing.

```python
import librosa
import soundfile as sf

audio, sr = librosa.load('input.mp3', sr=None, mono=True)
audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=16000)
sf.write('output.wav', audio_resampled, 16000, subtype='PCM_16')
```

---

### Step 2 — Chunking

Long recordings (10–60 minutes) chunked into segments of **20–30 seconds**. Rationale:

- Whisper processes audio in 30s windows natively
- Shorter chunks are easier to annotate manually
- Avoids padding issues on very long audio

```python
def chunk_audio(audio, sr, chunk_duration=25):
    chunk_size = int(chunk_duration * sr)
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) > sr * 2:  # skip chunks shorter than 2s
            chunks.append(chunk)
    return chunks
```

---

### Step 3 — Silence Removal

Many raw recordings contain long silences, background noise, and pauses. These degrade model training.

- Used `librosa.effects.split()` to detect non-silent regions
- Applied a 30dB threshold (tunable depending on recording environment)
- Merged nearby non-silent segments to avoid excessive fragmentation

```python
intervals = librosa.effects.split(audio, top_db=30)
cleaned = np.concatenate([audio[start:end] for start, end in intervals])
```

---

### Step 4 — Auto-Transcription (Silver Labels)

Used Whisper `base` to auto-transcribe chunks and generate initial transcripts. These are **silver labels** — not ground truth, but a starting point for manual correction.

```python
import whisper
model = whisper.load_model('base')
result = model.transcribe('chunk_001.wav', language='en')
print(result['text'])
```

Each chunk saved with:
- Audio file: `chunk_001.wav`
- Transcript file: `chunk_001.txt` (auto-generated, pending review)
- Metadata: `chunk_001.json` (duration, source, speaker ID, language tag)

---

### Step 5 — Manual Review (Partial)

Manual correction of auto-transcripts is the bottleneck. Issues found:

| Issue | Frequency | Example |
|-------|-----------|---------|
| Pidgin words dropped | High | "e don finish" → "it's done finished" |
| Yoruba fragments hallucinated | Medium | Yoruba word replaced with phonetically similar English |
| Proper nouns wrong | Medium | "Ile-Ife" → "early fee", "Abeokuta" → "Abbey Okuta" |
| Tonal stress misread | Low | Subtle — affects punctuation placement, not word identity |

---

## Dataset Structure (Target Format)

```
/dataset
  /audio
    chunk_001.wav
    chunk_002.wav
    ...
  /transcripts
    chunk_001.txt
    chunk_002.txt
    ...
  /metadata
    chunk_001.json
    chunk_002.json
    ...
  manifest.csv   ← master index: path, duration, speaker_id, language_tag, reviewed (bool)
```

### manifest.csv columns

| Column | Description |
|--------|-------------|
| `audio_path` | Relative path to .wav file |
| `transcript_path` | Relative path to .txt transcript |
| `duration_s` | Clip duration in seconds |
| `speaker_id` | Anonymous speaker identifier |
| `language_tag` | `en-NG`, `yo`, `pcm` (Nigerian Pidgin), or `mixed` |
| `reviewed` | Whether transcript has been manually verified |
| `quality` | Subjective audio quality: `clean`, `noisy`, `very_noisy` |

---

## Current Status

| Stage | Status |
|-------|--------|
| Format standardization | ✅ Done |
| Chunking pipeline | ✅ Done |
| Silence removal | ✅ Done |
| Auto-transcription | ✅ Done (Whisper base) |
| Manual transcript review | 🔄 Partial (~20% reviewed) |
| Speaker diversity expansion | ❌ Not started |
| Public release | ❌ Pending quality threshold |

---

## Known Limitations

- Small speaker pool — not representative of Nigerian speech diversity across regions
- Yoruba tonal diacritics not yet captured in transcripts (requires Yoruba-specific annotation tooling)
- No Igbo or Hausa speakers yet — dataset currently skews Yoruba/South-West Nigerian
- Manual annotation is slow without a team

---

## Resources & References

- [Mozilla Common Voice](https://commonvoice.mozilla.org/) — crowdsourced multilingual speech, limited Yoruba data available
- [Menyo-20k](https://arxiv.org/abs/2103.08647) — Yoruba-English machine translation dataset
- [FLEURS](https://huggingface.co/datasets/google/fleurs) — Google's multilingual speech benchmark, includes Yoruba
- [AfriSpeech-200](https://arxiv.org/abs/2104.02010) — accented African English ASR dataset (closest to this goal)
- [openai/whisper](https://github.com/openai/whisper) — base ASR model used for silver labelling
