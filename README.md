# Nigerian Speech & NLP Experiments

Personal research and experimentation around NLP and automatic speech recognition (ASR) for Nigerian English and Nigerian languages — covering accents, code-switching, tonal interference, and low-resource language challenges.

## Motivation

Standard ASR and NLP models are trained overwhelmingly on American and British English. When deployed on Nigerian speech, they struggle — not because Nigerian speakers are unclear, but because:

- **Nigerian English** carries phonological influence from Yoruba, Igbo, Hausa, and other L1 languages
- **Tonal languages** like Yoruba use pitch to distinguish meaning — this creates accent patterns that confuse models trained on non-tonal L1 speakers
- **Code-switching** between English and indigenous languages (e.g. Yoruba-English, Pidgin-English) is natural and frequent in real Nigerian speech
- **Low-resource reality**: very little labelled Nigerian speech data exists for fine-tuning

This repo documents my personal exploration of these problems — model evaluations, preprocessing experiments, and dataset construction notes.

---

## Contents

| File | Description |
|------|-------------|
| `whisper_asr_evaluation.ipynb` | Evaluating OpenAI Whisper across model sizes on Nigerian-accented English audio |
| `text_classification_distilbert.ipynb` | HuggingFace transformer pipeline for domain-specific text classification |
| `dataset_prep_notes.md` | Notes and approach for building a Nigerian speech audio dataset from scratch |

---

## Key Observations (so far)

- Whisper `base` and `small` handle Nigerian-accented English reasonably well for clear speech, but degrade noticeably on fast speech, code-switched segments, and Yoruba-inflected pronunciation
- DistilBERT works out of the box for general English classification but needs domain adaptation for Nigerian text (slang, pidgin fragments, local proper nouns)
- Audio dataset construction is the real bottleneck — most publicly available Nigerian speech data is either too small, too clean, or not representative of natural conversational speech

---

## Models Used

- [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
- [openai/whisper-base](https://huggingface.co/openai/whisper-base)
- [openai/whisper-small](https://huggingface.co/openai/whisper-small)
- [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)
- [Deepgram Nova](https://deepgram.com/) (API-based, evaluated externally)
- Google Coloud trancription service also being considered

---

## Background

I'm a Mechanical Engineering student at Obafemi Awolowo University (OAU), Ile-Ife — but I've been working on applied ML and NLP projects on the side, particularly around speech recognition for Nigerian contexts. My interest in this space comes partly from working on a real-time speech-to-text system for live events where Nigerian-accented English and language mixing were constant challenges. Overall experience is minimal, but I learn fast, and I'm still learning.

---

## Status

Active but informal — this is a portfolio/learning repo, not a production project. Notebooks are documented for readability.
At the moment, project is on pause due to personal more pressing matters, and minimal resources.

---

## Author

**Olaoluwa Ogunsola** — [GitHub: Olaolu22](https://github.com/Olaolu22)
