# ASR Shootout — Project Plan & Implementation Guide
### Vahan AI Intern Assessment | Gopi | May 2026

---

## Table of Contents

1. [What This Project Is About](#1-what-this-project-is-about)
2. [Problem Statement](#2-problem-statement)
3. [What We Are Building](#3-what-we-are-building)
4. [Our Approach & Philosophy](#4-our-approach--philosophy)
5. [Models We Are Benchmarking](#5-models-we-are-benchmarking)
6. [Datasets We Are Using](#6-datasets-we-are-using)
7. [Metrics We Are Measuring](#7-metrics-we-are-measuring)
8. [Tech Stack & Tools](#8-tech-stack--tools)
9. [End-to-End Workflow](#9-end-to-end-workflow)
10. [Phase-by-Phase Implementation Plan](#10-phase-by-phase-implementation-plan)
11. [Failure Analysis Strategy](#11-failure-analysis-strategy)
12. [Report Structure](#12-report-structure)
13. [Submission Checklist](#13-submission-checklist)
14. [Timeline](#14-timeline)
15. [Risks & Mitigations](#15-risks--mitigations)

---

## 1. What This Project Is About

Vahan is a blue-collar hiring platform operating across India. Candidates interact with the platform through phone calls and WhatsApp voice notes. These candidates speak in Hindi, Hinglish (Hindi-English mix), and regional languages. They call from noisy environments — busy streets, factories, public transport. Their audio comes in over low-bandwidth mobile connections, often at 8kHz telephony quality.

The platform needs to understand what candidates are saying. Specifically, it needs to extract structured information from speech — things like where a candidate lives, what kind of work they are looking for, and their availability. A candidate saying "Haan bhai, main Koramangala mein rehta hoon" should result in the system correctly extracting "Koramangala" as their locality.

This is an Automatic Speech Recognition (ASR) problem. And not all ASR systems are built equal for this specific use case. Some are built for clean English studio audio. Some are built for Indian languages. Some run on servers with high latency. Some can run in real-time.

This project evaluates which ASR system best serves Vahan's use case — and why.

---

## 2. Problem Statement

**Core question:** Which ASR system most accurately transcribes conversational Hindi and Hinglish speech, specifically in noisy telephony conditions, with correct detection of Indian proper nouns such as Bangalore locality names?

**Why this is hard:**

- Indian conversational speech is highly code-switched. A sentence like "Bhai, mujhe HSR Layout jaana hai, koi cab milegi kya?" mixes Hindi grammar with English words and Kannada place names. No single language model handles this cleanly.
- Locality names like Byatarayanapura, Kadugondanahalli, and Rajarajeshwarinagar are long, Kannada-origin words. They are rarely in the training data of general-purpose ASR systems. They are exactly the kind of named entity that breaks standard models.
- Phone-quality audio (8kHz, compressed, noisy) is fundamentally different from the clean 16kHz studio audio that most open-source models are trained on.
- Standard ASR metrics like Word Error Rate (WER) measure word-level accuracy but do not specifically penalise getting a proper noun wrong. A model that transcribes "Silk Road" instead of "Silk Board" might have a low overall WER but completely fail at the one thing that matters — extracting the locality.

---

## 3. What We Are Building

We are building a reproducible ASR benchmarking pipeline that:

- Takes a set of audio files as input
- Sends each file through multiple ASR systems
- Receives transcriptions back from each system
- Compares each transcription against a known ground-truth reference
- Computes multiple evaluation metrics per model per file
- Aggregates results into comparison tables and charts
- Identifies where each model fails and why
- Produces a written recommendation for which model Vahan should use

The final deliverables are:
1. 20 self-recorded audio files of Bangalore locality sentences in Hinglish
2. A clean Python codebase (Colab notebook) that runs the full pipeline
3. A 3-page written report with findings, failure analysis, and recommendation
4. Readiness for a 10-minute technical walkthrough presentation

---

## 4. Our Approach & Philosophy

### 4.1 Primary vs Supplementary Data

Our primary evaluation dataset is the 20 self-recorded audio files. These are the most valuable because they simulate exactly what Vahan receives — a single speaker, phone mic, natural conversational Hindi/Hinglish, messy conditions, locality names embedded in sentences.

The supplementary dataset is the GramVaani Hindi evaluation set (3 hours, 62MB) — real telephony speech from Hindi speakers across India. This broadens our evaluation beyond a single speaker and validates whether patterns we observe in our 20 recordings hold at scale.

### 4.2 Entity-First Evaluation

Standard ASR benchmarks measure overall Word Error Rate. For Vahan's use case, we care about something more specific — did the model correctly capture the locality name? A model can get 90% of the sentence right but still be useless if it mishears "Koramangala" as "Karamanagara."

We therefore introduce a custom metric called Entity Accuracy: did the locality name appear correctly in the transcription? This is the single most important metric for this use case, and we will report it prominently alongside WER.

### 4.3 Condition-Based Slicing

We do not just report aggregate scores. We record our 20 files under four distinct acoustic conditions and report performance per condition. This tells a more honest story — a model might do well on quiet audio but collapse on noisy telephony, which is exactly the condition Vahan operates in.

### 4.4 Cost-Aware Recommendation

API-based models charge per minute of audio. At Vahan's scale — potentially tens of thousands of calls per day — cost is a real factor. We will calculate cost-per-1000-minutes for each API model and factor it into the final recommendation alongside accuracy.

### 4.5 Honest Limitations

We will explicitly acknowledge what we cannot measure — streaming latency at scale, concurrent request throughput, real-time performance, model drift over time. Acknowledging limitations honestly is part of good engineering judgement.

---

## 5. Models We Are Benchmarking

### 5.1 Deepgram Nova-2 (Baseline — Required)

Deepgram is the required baseline for this assignment. Nova-2 is their most capable model as of 2025. It is a cloud API — we send audio, it returns a transcription. It supports Hindi (language code: `hi`). Deepgram is known for low latency and strong performance on telephony audio, which is why Vahan likely already uses or evaluates it.

**Why it is the baseline:** It represents the incumbent commercial option. Everything else we benchmark will be compared against it.

**Cost:** Free tier with $200 in credits — more than sufficient for 20 audio files.

### 5.2 Sarvam AI Saaras v3 (API — India-Specific)

Sarvam AI is an Indian AI company that has built speech models specifically for Indian languages. Their Saaras v3 model supports 22 Indian languages including Hindi and explicitly handles Hinglish code-switching. It is designed for call center audio — 8kHz, background noise, multiple speakers. This is the closest off-the-shelf model to Vahan's exact use case.

**Why we chose it:** It is the most domain-aligned API model. If any API beats Deepgram on Indian conversational speech, it will be Sarvam.

**Cost:** ₹1000 free credits on signup — sufficient for this evaluation.

### 5.3 OpenAI Whisper large-v3 (Open-Source — General)

Whisper is OpenAI's open-source speech recognition model, trained on 680,000 hours of multilingual audio. The large-v3 variant is the most accurate. It supports Hindi natively. It is the most widely used open-source ASR model in the world and serves as the open-source general-purpose baseline.

**Why we chose it:** It represents what you get for free with no India-specific tuning. Comparing it against Sarvam and Deepgram tells us how much India-specific tuning matters.

**Cost:** Free. Runs on Google Colab T4 GPU.

### 5.4 IndicWhisper (Open-Source — India-Tuned)

IndicWhisper is Whisper fine-tuned on Indian speech data by AI4Bharat, the AI research lab at IIT Madras. It has been evaluated on the Vistaar benchmark — a comprehensive Indian ASR benchmark — and achieves the lowest WER in 39 out of 59 benchmarks, with an average 4.1 point improvement over standard Whisper.

**Why we chose it:** It is the most directly relevant open-source model. It is Whisper but trained on Indian data. Comparing it to vanilla Whisper isolates the impact of India-specific fine-tuning.

**Cost:** Free. Available on HuggingFace. Runs on Google Colab T4 GPU.

### 5.5 Model Selection Rationale (Summary)

| Model | Type | India-Specific | Cost | Key Tradeoff |
|---|---|---|---|---|
| Deepgram Nova-2 | API | Partial | $200 free | Low latency, paid at scale |
| Sarvam Saaras v3 | API | Yes | ₹1000 free | Best India fit, newer player |
| Whisper large-v3 | Open-source | No | Free | Accurate but no India tuning |
| IndicWhisper | Open-source | Yes | Free | Best open-source for India |

This combination gives us coverage across all four important axes: API vs open-source, general vs India-specific.

---

## 6. Datasets We Are Using

### 6.1 Primary Dataset — Self-Recorded (Required)

**What it is:** 20 audio files recorded on a phone microphone, each containing a natural Hinglish sentence with a Bangalore locality name embedded in it.

**Language:** Hindi/Hinglish — sentences like "Haan bhai, main Koramangala mein rehta hoon."

**Conditions:** Four distinct acoustic conditions across the 20 files.
- Files 01–05: Quiet indoor room, normal speaking pace
- Files 06–10: Outdoor with street or traffic noise in the background
- Files 11–15: Rushed speech, faster pace, slight impatience
- Files 16–20: Whispered or low-energy speech, like talking on a crowded bus

**Why four conditions matter:** Vahan's candidates call from all these environments. If a model breaks on noisy audio but not quiet audio, that is critical information. Four conditions with five samples each gives us enough data to make meaningful comparisons.

**File naming convention:** `01_koramangala_quiet.wav`, `06_jayanagar_noisy.wav`, etc.

**Ground truth:** For each file, we write a reference transcript — the exact sentence spoken, lowercased and normalized. This is the target we compare model outputs against.

### 6.2 Supplementary Dataset — GramVaani Evaluation Set

**What it is:** 3 hours (62MB) of telephony-quality Hindi speech collected by Gram Vaani from Mobile Vaani users across India. It includes regional and dialectal Hindi, background noise, and spontaneous natural speech. It comes with reference transcripts.

**Why we use it:** It simulates real phone calls from working-class Indians across India — almost exactly Vahan's candidate profile. It validates our findings beyond a single speaker.

**How we use it:** We run a random sample of 50–100 utterances from this set through each model and report WER. This gives us a larger, multi-speaker, multi-dialect benchmark to support our findings from the 20 personal recordings.

**Download:** Freely available at openslr.org/118 — no login required.

---

## 7. Metrics We Are Measuring

### 7.1 Entity Accuracy (Most Important)

**Definition:** For each audio file, did the ASR system correctly produce the locality name in its output? This is a binary metric per file — 1 if the locality name appears in the transcript, 0 if it does not.

**Why it is most important:** This is Vahan's actual use case. The platform needs to know where a candidate lives. A system that correctly transcribes every other word but mangles "Marathahalli" is useless for this task. Entity Accuracy directly measures the thing that matters.

**How we compute it:** Check whether the expected locality name (lowercased) is a substring of the model's output (lowercased). For locality names with multiple words like "HSR Layout" or "BTM Layout", both words must appear.

### 7.2 Word Error Rate (WER)

**Definition:** The percentage of words in the reference transcript that were incorrectly transcribed by the model, accounting for insertions, deletions, and substitutions.

**Why we measure it:** It is the standard ASR metric and allows comparison with published benchmarks on datasets like GramVaani and Kathbath. Reviewers will expect to see it.

**Limitation for this task:** WER treats all words equally. Getting "Koramangala" wrong counts the same as getting "bhai" wrong. This is why Entity Accuracy is more important for our specific use case.

### 7.3 Character Error Rate (CER)

**Definition:** Same as WER but at the character level rather than the word level.

**Why we measure it:** For long, complex proper nouns like "Byatarayanapura" or "Rajarajeshwarinagar", a model might get the first half right and the second half wrong. WER counts this as a fully wrong word. CER shows how close the model got character by character. It is more informative for evaluating named entity recognition quality.

### 7.4 Latency

**Definition:** The wall-clock time from when we send an audio file to an API (or start model inference) to when we receive the full transcription back.

**Why we measure it:** For a real-time phone call application, latency determines whether the system can keep up with a live conversation. A model that takes 8 seconds to transcribe a 3-second utterance cannot work in production. We measure this for all models, though we note this is batch latency (not streaming) and serves as a proxy.

**How we measure it:** Python `time.time()` around each API call or inference call. We take the average across all 20 files per model.

### 7.5 Cost Per Hour of Audio (API Models Only)

**Definition:** Based on each API's published pricing, what would it cost to transcribe one hour of audio? What would it cost at Vahan's scale — say, 10,000 minutes per day?

**Why we measure it:** Accuracy differences of 5% between two models are meaningless if one costs 10x more at scale. Cost is a first-class product constraint.

---

## 8. Tech Stack & Tools

### 8.1 Recording

- **Tool:** Default Voice Recorder app on any Android or iPhone
- **Format output:** M4A or MP3 (we convert this later)
- **Cost:** Free

### 8.2 Audio Processing

- **ffmpeg:** Command-line tool for converting audio formats and resampling. We use it to convert all recordings from M4A to 16kHz mono WAV, which is the standard input format for all ASR models.
- **pydub:** Python library for basic audio operations — splitting, padding, normalizing volume.
- **Cost:** Both are free and open-source.

### 8.3 Compute Environment

- **Google Colab:** Free cloud Jupyter notebook environment with access to an NVIDIA T4 GPU (15 hours/day on free tier). We use this to run Whisper large-v3 and IndicWhisper, both of which require a GPU to run in reasonable time.
- **Cost:** Free

### 8.4 ASR APIs

- **Deepgram SDK (Python):** Official Python client for Deepgram's API. Install via pip.
- **Sarvam AI SDK (Python):** Official Python client for Sarvam's API. Install via pip.
- **Authentication:** API keys stored as Colab secrets (not hardcoded in notebook).
- **Cost:** Free tier — $200 Deepgram credit, ₹1000 Sarvam credit.

### 8.5 Open-Source Models

- **openai-whisper:** OpenAI's official Python package for Whisper. The large-v3 model (~3GB) downloads automatically from their CDN on first use.
- **transformers (HuggingFace):** The standard Python library for loading and running HuggingFace models. Used to load IndicWhisper.
- **datasets (HuggingFace):** Used to load the GramVaani supplementary dataset.
- **torch:** PyTorch, required by both Whisper and HuggingFace transformers.
- **Cost:** All free and open-source.

### 8.6 Metrics & Analysis

- **jiwer:** The standard Python library for computing WER and CER. Used in academic ASR papers and industry benchmarks.
- **pandas:** For storing and manipulating all results in DataFrames. Every result — model name, file name, condition, reference, hypothesis, WER, CER, entity hit, latency — goes into a structured table.
- **Cost:** Both free and open-source.

### 8.7 Visualization

- **matplotlib:** For generating bar charts, heatmaps, and comparison plots.
- **seaborn:** Higher-level chart styling on top of matplotlib. Makes publication-quality charts with less code.
- **Cost:** Both free and open-source.

### 8.8 Report Writing & Submission

- **VS Code or any text editor:** For writing the Markdown report.
- **GitHub:** For hosting the code repository. The submission form likely asks for a repo link.
- **Cost:** Both free.

---

## 9. End-to-End Workflow

The pipeline flows in a single direction, from raw audio to final report. Here is the complete path a single audio file takes through the system:

**Step 1 — Record**
A sentence is spoken into a phone mic under a specific acoustic condition and saved as an M4A file.

**Step 2 — Convert**
ffmpeg converts the M4A to a 16kHz mono WAV file. This is a standard preprocessing step that ensures all models receive audio in their expected format.

**Step 3 — Ground truth labeling**
The reference transcript for the file is written manually — the exact sentence spoken, normalized to lowercase, punctuation removed. This is stored in a CSV alongside the file name, locality name, and condition label.

**Step 4 — API inference (Deepgram, Sarvam)**
Each WAV file is sent to the Deepgram API and the Sarvam API. The API returns a transcription string. We record the transcription and the wall-clock time taken.

**Step 5 — Local model inference (Whisper, IndicWhisper)**
Each WAV file is passed through the locally loaded Whisper large-v3 and IndicWhisper models on the Colab GPU. Each model returns a transcription string. We record the transcription and inference time.

**Step 6 — Metric computation**
For each (audio file, model) pair, we now have a reference transcript and a hypothesis transcript. We compute WER, CER, and Entity Accuracy. We also record the latency captured in steps 4 and 5.

**Step 7 — Aggregation**
All results are merged into a single pandas DataFrame — one row per (file, model) pair. This gives us 80 rows total (20 files × 4 models).

**Step 8 — Analysis**
We group and aggregate by model, by condition, and by locality. We look for patterns — which models fail on noisy audio, which locality names break all models, which model has the best entity accuracy.

**Step 9 — Visualization**
We generate comparison charts — WER by model, Entity Accuracy by model, WER by condition heatmap, latency comparison.

**Step 10 — Report**
We write a 3-page Markdown report covering approach, findings, failure analysis, and recommendation. Charts are embedded.

**Step 11 — GramVaani validation (supplementary)**
We run the same inference pipeline (steps 4–8) on 50–100 utterances from the GramVaani evaluation set and report WER scores. This broadens the evaluation and adds credibility.

---

## 10. Phase-by-Phase Implementation Plan

### Phase 1 — Record 20 Audio Samples

**Goal:** Produce 20 natural-sounding Hinglish audio files with Bangalore locality names, varying acoustic conditions.

**What to do:**
- Choose 20 sentences from the prepared sentence list, one per locality
- Record files 01–05 in a quiet indoor room at normal speaking pace
- Record files 06–10 outdoors near a road or busy area
- Record files 11–15 indoors but speaking quickly and with urgency in the voice
- Record files 16–20 in a whispered or low-energy style
- Rename all files consistently: `01_koramangala_quiet.m4a`, etc.
- Transfer files to your laptop for upload to Google Drive

**Common mistakes to avoid:**
- Do not read the sentence off a screen in a flat robotic voice. Speak naturally, with the intonation of someone actually in that situation.
- Do not record all files in the same room at the same pace. The whole point is variation.
- Do not re-record until it sounds "perfect." Imperfection is the point. Natural hesitations, slight mispronunciations, and regional accent are valuable.

**Time estimate:** 1.5 to 2 hours including renaming and transfer.

### Phase 2 — Environment Setup & Audio Preparation

**Goal:** Set up Google Colab with all required libraries, convert all audio to the correct format, and create the ground truth CSV.

**What to do:**
- Open Google Colab and create a new notebook
- Upload the 20 audio files to Google Drive and mount Drive in Colab
- Install all required Python packages: deepgram-sdk, openai-whisper, transformers, datasets, jiwer, pandas, matplotlib, seaborn, pydub
- Install ffmpeg in Colab (one shell command)
- Write a loop that converts every M4A file to a 16kHz mono WAV file using ffmpeg
- Manually create the ground truth CSV — one row per recording, with columns: file name, reference transcript (what was said, lowercased), locality name, and condition

**Important note on ground truth:** The reference transcript must be written carefully. It should reflect exactly what was said — including filler words, false starts, or informal contractions. If you said "haan bhai main koramangala mein rehta hoon" then write exactly that, lowercased. Do not write the "correct" grammatical version if that is not what you said.

**Time estimate:** 1 hour.

### Phase 3 — Run API Models

**Goal:** Get transcriptions from Deepgram and Sarvam for all 20 audio files, with latency recorded.

**What to do:**
- Sign up for Deepgram at deepgram.com — get API key from the dashboard
- Sign up for Sarvam AI at sarvam.ai — get API subscription key
- Store both keys as Colab secrets (not hardcoded)
- Write a function that takes a WAV file path and a model choice, calls the appropriate API, and returns the transcript and latency
- Loop through all 20 files for Deepgram, saving results to a list
- Loop through all 20 files for Sarvam, saving results to a list
- Convert each list to a pandas DataFrame and save as CSV

**Parameters to use:**
- Deepgram: model `nova-2`, language `hi`, smart formatting off (raw transcription is easier to compare)
- Sarvam: model `saaras:v3`, language code `hi-IN`

**Time estimate:** 1 to 2 hours including API signup.

### Phase 4 — Run Open-Source Models

**Goal:** Get transcriptions from Whisper large-v3 and IndicWhisper for all 20 audio files on Colab GPU.

**What to do:**
- In Colab, switch runtime to GPU (Runtime → Change runtime type → T4 GPU)
- Load Whisper large-v3 using the openai-whisper package
- Loop through all 20 WAV files and run inference, recording transcript and inference time
- Save results to CSV
- Load IndicWhisper from HuggingFace using the transformers pipeline API
- Set the forced decoder language to Hindi before inference
- Loop through all 20 WAV files and run inference, recording transcript and inference time
- Save results to CSV

**Notes:**
- Whisper large-v3 takes about 2–4 seconds per short audio file on a T4 GPU
- IndicWhisper may take slightly longer depending on model size
- If a model runs out of GPU memory, reduce batch size or use a smaller model variant. Note this in the report as a valid observation.

**Time estimate:** 2 to 3 hours including model download time (models are large and take time to download on first run).

### Phase 5 — Compute Metrics and Analyse

**Goal:** Calculate WER, CER, entity accuracy, and latency for every (file, model) pair. Identify patterns and failure modes.

**What to do:**
- Load all four results CSVs and the ground truth CSV into pandas DataFrames
- Merge them into a single combined DataFrame with one row per (file, model) pair — 80 rows total
- Apply jiwer.wer() and jiwer.cer() to each row's reference-hypothesis pair
- Apply the entity accuracy check: is the locality name present in the hypothesis?
- Compute a summary table grouping by model: average WER, average CER, entity accuracy percentage, average latency
- Compute a condition breakdown table grouping by model and condition
- Find the hardest localities: which locality names have the most failures across all models?
- Find specific failure examples: pick 3–5 interesting cases where a model made a notable or surprising error

**What to look for in failure analysis:**
- Models that substitute a known English word for an Indian locality ("Silk Road" for "Silk Board")
- Models that completely skip a locality name and produce silence or a filler word
- Models that partially get a locality name right (e.g., "Korama" instead of "Koramangala")
- Models that perform well on quiet audio but collapse on noisy audio
- Locality names that all models consistently fail on (likely the long Kannada-origin ones)

**Time estimate:** 2 to 3 hours.

### Phase 6 — GramVaani Supplementary Analysis

**Goal:** Validate findings on a broader, multi-speaker telephony Hindi dataset.

**What to do:**
- Download the GramVaani evaluation set (62MB) from openslr.org/118
- Upload to Google Drive and mount in Colab
- Pick a random sample of 50 to 100 utterances from the evaluation set
- Run all four models on this sample using the same pipeline from phases 3 and 4
- Compute WER for each model on this sample
- Compare: do the rankings match what we saw in our 20 recordings?

**Why this matters for the report:** It moves the evaluation from one speaker to many speakers, and from our constructed sentences to real spontaneous speech. If Sarvam beats Deepgram on our recordings and also on GramVaani, that is a strong finding. If the rankings flip, that is an equally interesting finding that deserves explanation.

**Time estimate:** 1 to 2 hours (most of the code is reusable from phases 3 and 4).

### Phase 7 — Visualizations

**Goal:** Produce 3–4 clean charts to include in the report.

**Charts to make:**

1. **Bar chart — Average WER by model:** Four bars, one per model. Deepgram bar gets a label noting it is the baseline. Lower is better.

2. **Bar chart — Entity Accuracy by model:** Four bars showing what percentage of locality names each model got right. This is the headline chart.

3. **Heatmap — WER by model × condition:** Rows are models, columns are the four conditions (quiet, noisy, rushed, whispered). Cell color encodes WER. This immediately shows which model degrades most in noisy conditions.

4. **Bar chart — Average latency by model:** Only meaningful for comparison purposes. Note in the chart that this is batch latency, not streaming.

**Time estimate:** 1 hour.

### Phase 8 — Write the Report

**Goal:** Produce a concise, opinionated 3-page Markdown document.

**Guidelines:**
- Do not pad. Every sentence must earn its place.
- Lead with the headline finding, not with methodology.
- Use tables for numbers, not prose. Do not write "Deepgram achieved a WER of 32% while Sarvam achieved 28%..." — just put it in a table.
- Name specific failure examples. "On file 14 (Silk Board, noisy condition), Deepgram returned 'Silk Road' — the model substituted a known English term for the Indian locality" is worth more than any aggregate number.
- End with a clear recommendation. Name one model. Say what condition you recommend it under. Say what the tradeoffs are.
- Keep the limitations section honest and brief.

**Time estimate:** 2 to 3 hours.

---

## 11. Failure Analysis Strategy

Failure analysis is explicitly called out as a criterion in the assignment. Here is how we approach it systematically.

### 11.1 Failure Categories

We classify every wrong transcription into one of these categories:

- **Entity miss:** The locality name is completely absent from the output
- **Entity substitution:** A different word appears where the locality should be (e.g., "Silk Road" for "Silk Board")
- **Entity truncation:** Part of the locality name is correct but the rest is missing (e.g., "Hebba" for "Hebbal")
- **Entity transliteration error:** The locality name appears but in a different script, language, or phonetic variant (e.g., "Marathhali" for "Marathahalli")
- **Noise collapse:** The entire transcription is wrong or empty — the model failed to process the audio
- **Code-switch failure:** The model correctly transcribed the Hindi parts but got confused at the point where a Kannada-origin place name appeared

### 11.2 Hardest Localities

We expect these locality names to be hardest for all models, in order of predicted difficulty:

1. Byatarayanapura — very long, uncommon, Kannada origin
2. Kadugondanahalli — long, Kannada, rare in training data
3. Rajarajeshwarinagar — very long, compound Kannada-Sanskrit
4. Kengeri Upanagara — multi-word, uncommon suffix
5. Thalaghattapura — less common, Kannada origin

We will verify this prediction against actual results and discuss it in the report.

### 11.3 Model-Specific Failure Patterns

We will look for patterns that reveal something about each model's training data or architecture:

- Does Deepgram show a bias toward common English words when it cannot recognize an Indian name?
- Does Whisper hallucinate — produce plausible-sounding but wrong text rather than silence?
- Does IndicWhisper handle long Kannada-origin words better than the others, since it was trained on more Indian data?
- Does Sarvam struggle with any specific class of locality name despite its India focus?

---

## 12. Report Structure

The report is a maximum of 3 pages in Markdown format. Here is the exact structure:

### Section 1 — Approach (0.5 pages)
What did we evaluate, why, and how. Model selection rationale. Dataset description. Metric choices with justification. One sentence on what surprised us (teaser).

### Section 2 — Results (1 page)
Summary table: WER, CER, entity accuracy, latency per model. Condition breakdown table: WER by model by condition. The headline chart (entity accuracy bar chart). One key observation per table.

### Section 3 — Failure Analysis (0.5 pages)
A table of the top 5 most-failed locality names per model. Three specific transcription examples with reference, hypothesis, and a one-line diagnosis of what went wrong. A brief categorization of failure modes (entity miss vs substitution vs truncation).

### Section 4 — Recommendation (0.5 pages)
A clear, direct recommendation. Name one model as primary recommendation. State the conditions under which it wins. State the tradeoff being made (e.g., accuracy vs cost, or open-source control vs API convenience). Note any secondary recommendation for a different constraint (e.g., "if self-hosted is required, use IndicWhisper").

### Section 5 — Limitations (brief)
What we could not measure: streaming latency, concurrent throughput, real-time performance, multi-speaker diarization, dialectal variation beyond single speaker. Two or three sentences maximum.

---

## 13. Submission Checklist

Before submitting, verify every item:

**Audio files**
- [ ] 20 WAV files present
- [ ] Named consistently: `NN_locality_condition.wav`
- [ ] Cover all four conditions (at least 4–5 files per condition)
- [ ] Each file is 3–8 seconds long
- [ ] Each file contains a full Hinglish sentence, not just the locality name

**Code**
- [ ] Single Colab notebook that runs end-to-end
- [ ] API keys loaded from environment/secrets, not hardcoded
- [ ] All CSV outputs (results per model, combined results) are saved
- [ ] Charts are generated and saved as PNG
- [ ] A requirements.txt or pip install block is present at the top
- [ ] The notebook runs without errors from top to bottom on a fresh Colab session

**Report**
- [ ] Maximum 3 pages
- [ ] Contains all four sections: approach, results, failure analysis, recommendation
- [ ] Has at least one table and at least one chart
- [ ] Names a specific recommended model with justification
- [ ] Acknowledges limitations
- [ ] No filler sentences

**Repository**
- [ ] Code is pushed to a public GitHub repository
- [ ] README explains how to run the notebook
- [ ] Audio files are either in the repo or linked from Google Drive with public access

---

## 14. Timeline

| Day | Phase | Output |
|---|---|---|
| Day 1 morning | Record 20 audio files | 20 named M4A/WAV files |
| Day 1 afternoon | Colab setup + audio conversion + ground truth | 20 WAV files at 16kHz, ground_truth.csv |
| Day 2 morning | Run Deepgram and Sarvam APIs | results_deepgram.csv, results_sarvam.csv |
| Day 2 afternoon | Run Whisper and IndicWhisper on Colab GPU | results_whisper.csv, results_indicwhisper.csv |
| Day 3 morning | Metrics computation + GramVaani supplementary | all_results.csv, charts.png |
| Day 3 afternoon | Write report and push to GitHub | report.md, final repo |

Total estimated time: 14 to 18 hours of active work across 3 days.

---

## 15. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Colab GPU quota exhausted | Medium | Run Whisper first (heaviest model). If quota runs out, Whisper small is a valid fallback — document the tradeoff. |
| Deepgram API key issues | Low | Sign up in advance. Free tier activates immediately. |
| Sarvam API key issues | Low | Sign up in advance. Test with one file before running all 20. |
| IndicWhisper model download slow | Medium | Download in background while running API models. The HuggingFace model cache persists across Colab sessions if Drive is mounted. |
| GramVaani download slow | Low | 62MB file — takes a few minutes on average broadband. |
| Own recordings sound unnatural | Medium | Do not overthink it. Record quickly without re-listening. The first take is usually more natural than the fifth. |
| All models perform similarly | Low | Even small differences are interesting. Focus the analysis on the condition breakdown and failure cases — patterns will differ even if aggregate WER is close. |

---

*Document version 1.0 — Prepared for Vahan AI Intern Assessment, May 2026*
