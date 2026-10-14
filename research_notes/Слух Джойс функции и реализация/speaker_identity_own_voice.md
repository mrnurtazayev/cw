# Speaker identity ("who is talking") and own-voice handling for Joyce

Scope: speaker embeddings, enrollment, combining with Soniox diarization, streaming diarization, own-voice and echo handling, anti-spoofing, privacy and legal issues, and addressee detection, for a home assistant with about 5 Russian-speaking household members. Notes are as of 2026-10-14.

Method caveats:
- Some fetches failed because soniox.com and picovoice.ai did not resolve in the sandbox. Facts from those sites come from search-result snippets of the official pages, not from full reads.
- Many vendor numbers are self-reported. They are flagged where relevant.

## Q1. Which speaker-embedding model should we use (accuracy, latency, size, license, short utterances, Russian)?

### Takeaway
On VoxCeleb, open models sit in a narrow band of about 0.4–1.1% EER. The two that best fit fast, cheap and commercial use are:
- **CAM++ / ERes2NetV2** (3D-Speaker, Alibaba): small, built for short segments.
- **WeSpeaker ResNet34 / SimAM-ResNet (VoxBlink2, multilingual)**: shipped as ONNX.

Two other options:
- **Picovoice Eagle** is a commercial, on-device, turnkey option. Its vendor benchmark claims the best EER, but it is closed and paid.
- **The real risks are elsewhere.** Short utterances (2–3 s roughly doubles EER), children, and whisper or other vocal-effort changes matter more than the choice of model. Language mismatch is also a risk if anyone enrolls in one language and speaks another.

### Cited Findings
**WeSpeaker (wenet-e2e)**
- The published pretrained models are:
  - VoxCeleb (EN): ResNet34, ResNet34_LM, ResNet152_LM, ResNet221_LM, ResNet293_LM, CAM++, CAM++_LM, ECAPA512/1024 (+_LM), and Gemini_DFResnet114_LM.
  - CNCeleb: ResNet34.
  - **Multilingual (VoxBlink2):** SimAMResNet34 and SimAMResNet100, plus "_ft" versions fine-tuned on VoxCeleb2.
  - VoxCeleb+VoxBlink2: W2V-BERT2.0 variants.
  - Formats: `.pt` checkpoints and `.onnx` runtime models.
  — [WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- **Licensing:** "The pretrained model in WeNet follows the license of it's corresponding dataset", for example VoxCeleb under CC BY 4.0. Check this for commercial use, especially for VoxBlink2-trained models. — [WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- **"_LM" models:** large-margin fine-tuned, which "could perform better on long audios, e.g. >3s". For 1–3 s home commands the non-LM model may be preferable; this is an inference from that wording. — [WeSpeaker pretrained.md](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)
- **ResNet34:** 0.723% EER / 0.069 minDCF on Vox1-O-clean after LM fine-tuning and AS-Norm. — [WeSpeaker README](https://github.com/wenet-e2e/wespeaker/)
- **ResNet293:** 0.447% EER / 0.043 minDCF on Vox1-O-clean. It is unclear whether this is the exact released checkpoint. — [WeSpeaker README](https://github.com/wenet-e2e/wespeaker/); [WeSpeaker VoxSRC2023 baselines](https://arxiv.org/pdf/2306.15161)

**3D-Speaker (CAM++, ERes2NetV2)**
- **CAM++ vs ECAPA-TDNN:** 51% fewer parameters and 18% lower EER on VoxCeleb-O, in the authors' controlled comparison. — [CAM++ paper](https://arxiv.org/pdf/2303.00332)
- **ERes2NetV2, by trial duration on VoxCeleb1-O:** **0.61% EER on full-length trials, 0.98% on 3 s, 1.48% on 2 s.** The model was designed for short-duration verification and was pruned to cut parameters and FLOPs. — [ERes2NetV2 paper](https://arxiv.org/abs/2406.02167)
- **3D-Speaker toolkit:** ERes2NetV2 performs best of its six models, and CAM++ is "competitive with lower computational complexity". Pretrained ERes2NetV2 and CAM++ checkpoints trained on **200k labeled speakers** are on ModelScope. — [3D-Speaker toolkit paper](https://arxiv.org/pdf/2403.19971); [3D-Speaker GitHub](https://github.com/alibaba-damo-academy/3D-Speaker)

**TitaNet (NeMo) and ECAPA (SpeechBrain)**
- **TitaNet paper:** 0.68% EER on VoxCeleb1; TitaNet-S has only 6M parameters. — [TitaNet paper](https://arxiv.org/abs/2110.04410v1)
- **TitaNet-S on NGC:** the page lists 1.08% EER but its text attributes that figure inconsistently. — [NGC TitaNet-small](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small)
- **TitaNet-Large:** the speakerverification_en model is about 23M parameters. — [HF nvidia/speakerverification_en_titanet_large](https://huggingface.co/nvidia/speakerverification_en_titanet_large/raw/main/README.md)
- **TitaNet speed:** one comparison found TitaNet has the best VoxCeleb1 EER but is slower at producing embeddings than the other models compared. — [Zero-shot multilingual SV in clinical trials](https://arxiv.org/pdf/2404.01981)
- **ECAPA-TDNN:** pretrained on VoxCeleb, it reaches 0.8% EER on VoxCeleb1-O ([ChildAugment](https://arxiv.org/pdf/2402.15214)) and 0.87% ([CryCeleb](https://arxiv.org/pdf/2305.00969)).

**Picovoice Eagle (commercial, on-device)**
- **Vendor-published EER:** Eagle **0.18%**, SpeechBrain 0.49%, pyannote 0.70%. — [Picovoice Eagle product page](https://picovoice.ai/products/voice/speaker-recognition/)
- **Benchmark setup:** VoxConverse, 96 ms frames, offline enrollment. WeSpeaker is also compared, and the benchmark code is Apache 2.0. — [Picovoice benchmark docs](https://picovoice.ai/docs/benchmark/speaker-recognition)
- **Picovoice's own caveat:** EER alone can mislead, so compare full DET/ROC curves. — [Picovoice blog](https://picovoice.ai/blog/evaluating-speaker-recognition-performance/)
- Picovoice publishes a "speaker-aware voice assistant" cookbook pattern. — [Picovoice cookbook](https://picovoice.ai/cookbook/ai-voice-assistant-speaker-recognition/)

**Cross-language (relevant for Russian)**
- **TidyVoice 2026 challenge** (Common Voice, about 40 languages): the SimAM-ResNet34 baseline gets about 3.07% EER on seen languages but **about 9.8–11.6% on unseen languages**. Its pretraining is VoxBlink2+VoxCeleb2 with TidyVoice fine-tuning. — [TidyVoice papers](https://arxiv.org/pdf/2601.21960); [TidyVoice dataset](https://arxiv.org/pdf/2601.16358)
- **Language normalization helps:** projecting embeddings off a language subspace (Nuisance Attribute Projection) before cosine scoring cut EER from 2.97% (cosine) and 2.70% (AS-Norm) to 2.18%. — [Simple Language Normalization Wins](https://arxiv.org/pdf/2607.22923)
- **Older English-to-Chinese result:** domain adaptation cut x-vector EER from about 9.33% to 7.65%. — [arXiv 1908.01447](https://arxiv.org/pdf/1908.01447)

### Inferences
**Recommended open-source stack (5 users, Russian)**
- Primary candidates:
  - **3D-Speaker ERes2NetV2 or CAM++** (200k-speaker training; strong on 2–3 s).
  - **WeSpeaker SimAM-ResNet34 trained on VoxBlink2** (multilingual).
  - Run either on ONNX Runtime on CPU.
- On a laptop or Raspberry Pi 5 class CPU, a few-million-parameter model typically embeds 1–3 s of audio in tens of milliseconds. **This number was not found in sources and must be measured.**
- Closed-set identification among 5 enrolled people is much easier than open-set verification against impostors. Expected top-1 accuracy is above 95% on clean ≥2 s adult speech.
- On 1 s snippets, children's speech, or far-field audio with music, expect noticeably worse results. The 2 s EER (1.48%) is already about 2.4× the full-length EER.

**Practical rules**
- Do not decide identity on segments shorter than about 1.5–2 s. Instead, accumulate audio per Soniox speaker label and re-score.
- Benchmark 2–3 candidate models on your own household recordings in Russian. Public EERs come from English celebrity data (VoxCeleb).
- Use Eagle if you want a supported SDK and accept per-device licensing. Prefer CAM++/ERes2NetV2 (Apache-2.0 code; check the weight license) if cost and openness matter.

**Russian language**
- If everyone enrolls and speaks in Russian (same language), the cross-language penalty mostly does not apply.
- It does apply for code-switching households (Russian/Kazakh/English). Enroll in every language the person actually uses.

### Gaps
**Model figures not found**
- No independent CPU latency or RTF numbers for each model.
- Model-card EERs for CAM++/ERes2NetV2 checkpoints (the toolkit table was garbled in the fetch).

**Licenses not verified**
- ECAPA (SpeechBrain): the Apache-2.0 code license was not verified in this session.
- pyannote/embedding: the model's conditions were not verified in this session.
- 3D-Speaker weights: the license was not verified.

**Data and vendors not covered**
- No Russian-specific SV benchmark found (no Russian breakdown from TidyVoice was retrieved).
- Azure Speaker Recognition: its current availability and status were not checked. It was announced for retirement in earlier years; verify.
- Picovoice pricing was not retrieved; the site was unreachable.
- Singing: no source found on speaker-embedding performance for singing.

## Q2. How should enrollment, profile updates, normalization, thresholds, unknown speakers, kids and vocal effort be handled?

### Takeaway
Enroll several samples per person under different conditions, average or keep multiple centroids, and score with cosine plus AS-Norm. Use two thresholds (accept / unknown / ask) and update profiles only from high-confidence segments.

Expect large degradation in three cases:
- **Kids:** worse for younger children, so plan for periodic re-enrollment.
- **Whisper:** a strong mismatch with normal-voice enrollment; compensation can recover up to about 50% relative.
- **Shouting:** similar effect to whispering.

### Cited Findings
**Score normalization**
- AS-Norm is used in reported SOTA results: ResNet34 reaches 0.723% EER "after LM fine-tuning and AS-Norm". — [WeSpeaker README](https://github.com/wenet-e2e/wespeaker/)
- In the TidyVoice work, AS-Norm improves on raw cosine (2.97%→2.70% EER). — [arXiv 2607.22923](https://arxiv.org/pdf/2607.22923)

**Children**
- Adult-trained extractors show a **40–45% relative EER increase** on children. — [ChildAugment](https://arxiv.org/pdf/2402.15214)
- VoxCeleb-pretrained ECAPA on the OGI kids corpus: EER goes from **30.65% (kindergarten)** to **6.15% (grade 10)**, versus 0.8% on VoxCeleb1-O. EER falls as children age, with a sharp change around puberty for boys (grades 7–8). — [ChildAugment](https://arxiv.org/pdf/2402.15214)
- Domain adaptation to children causes "catastrophic forgetting" for adults. An age-agnostic approach with a domain classifier aims to cover both groups. — [Age-agnostic SV](https://arxiv.org/pdf/2508.01637)
- Age mismatch between enrollment and test degrades deep speaker embeddings. — [SV Across Ages](https://arxiv.org/pdf/2306.07501)
- Infant cries reach about 38% EER with a naive baseline (about 28% after fine-tuning), so they are not usable for identification. — [CryCeleb](https://arxiv.org/pdf/2305.00969)

**Whisper and shouting**
- Non-normal modes "severely hurt" verification. Vocal-effort detection plus calibration and embedding compensation gave **up to 52% relative EER improvement** on whispered–normal trials. — [Prieto et al. 2022](https://zaguan.unizar.es/record/148274)
- Compensation methods for shouted and whispered speech exist: MMSE in a PCA domain and vocal-effort transfer vectors. — [López-Espejo MLSP 2023](https://esperanto.univ-lemans.fr/en/publications/conferences/lopez-espejo_-ieee-mlsp2023.html); [arXiv 2305.02147](https://arxiv.org/html/2305.02147v3)
- Whisper-to-neutral feature mapping cut whispered-trial EER by about 24% relative. — [Naini et al. Interspeech 2019](https://eprints.iisc.ac.in/id/eprint/78251)
- An April 2026 paper reports 1.88% EER for whispered-vs-whispered trials, and says noise hurts whispered speech more than normal speech. — [arXiv 2604.20229 summary](https://www.emergentmind.com/papers/2604.20229)

### Inferences
**Enrollment**
- Collect about 5–10 utterances of 3–5 s each (about 30–60 s total) per person, across different conditions:
  - near and far from the mic,
  - with the TV on and off,
  - morning and evening voice,
  - every language the person speaks.
- Store either the mean embedding or a small set of centroids (k=2–4) per person.
- Score against the **max** of the centroids, which suits multi-condition enrollment. This is standard practice, not directly sourced here.

**Thresholds**
- Use cosine with AS-Norm. The cohort can be 200–1000 embeddings from public data; the household's other members can be added too.
- Define two thresholds from your own household-recorded calibration:
  - **Accept:** score ≥ T_hi and best−second-best ≥ margin.
  - **Unknown:** score < T_lo.
  - **Between the two:** label "uncertain", then accumulate more audio or ask "Это Аня?" ("Is this Anya?").

**Incremental profile updates**
- Add only segments with high score, sufficient length (≥3 s) and no overlap.
- Cap the store with FIFO/reservoir, and keep the original enrollment as an anchor to prevent drift or poisoning.
- Re-enroll kids every 6–12 months.

**Auto-enrolling new household members**
- Cluster persistent "unknown" embeddings across days.
- Once a cluster has N≥5 consistent segments, ask the user to name the person. This also gives a consent moment.

**Whisper and singing**
- Treat whisper and singing as low-confidence states: don't let them update profiles, and don't use them for privileged actions.
- No data was found for singing. **Assume embeddings degrade** and test it.

### Gaps
- No primary source found specifically for the "number and length of enrollment samples" recommendation, or for incremental-update algorithms.
- No source found on illness or voice aging over months for adults.
- No source found on singing.

## Q3. How do we combine Soniox real-time diarization with embedding-based identification, and what are the streaming diarization alternatives?

### Takeaway
Soniox gives each token an anonymous `speaker` label that is consistent within one stream but is not an identity. Labels can briefly flip before settling.

The simple, robust pattern:
- Buffer audio per (stream, Soniox-speaker) using token timestamps.
- Embed once at least 1.5–3 s has accumulated, then map label→person with a sticky, majority-vote mapping.
- Re-check periodically.

If Soniox labels prove unreliable, the alternatives are NVIDIA Streaming Sortformer (≤4 speakers, frame-level, NeMo) or diart (pyannote-based, 0.5–5 s latency).

### Cited Findings
**Soniox**
- Enable diarization with `enable_speaker_diarization: true`. It works in real-time and async modes, and each token carries a `speaker` field. — [Soniox docs: speaker diarization](https://soniox.com/docs/stt/concepts/speaker-diarization); [Soniox real-time transcription](https://soniox.com/docs/stt/rt/real-time-transcription)
- Token fields: `text`, timing, `confidence`, `is_final`, and `speaker` (a string present when diarization is on). — [Soniox llms.txt](https://soniox.com/docs/llms.txt)
- Capacity: up to **15 speakers**, 60+ languages, a label on every token. Labels **can briefly switch before settling**, and async gives the highest accuracy. — [Soniox diarization feature page](https://soniox.com/speech-to-text/features/diarization)
- "Speaker diarization recognizes different speakers, but it does not identify the speakers." — [Soniox docs](https://soniox.com/docs/speech-to-text/core-concepts/speaker-diarization)
- LiveKit's Soniox plugin reports `capabilities.diarization=True`. — [LiveKit Soniox plugin](https://docs.livekit.io/agents/models/stt/soniox.md)

**NVIDIA Streaming Sortformer**
- Frame-level labels (spk_0…), up to **4 speakers**. Labels can lag a few hundred ms after a speaker change. — [NVIDIA blog](https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/); [NVIDIA FAQ](https://perspectives.nvidia.com/nemotron-speech/task/faq/which-speaker-diarization-models-support-real-time-streaming-output-for-live-con/)
- The default checkpoint is `nvidia/diar_streaming_sortformer_4spk-v2.1`. — [NVIDIA Voice Agent docs](https://docs.nvidia.com/nemo/labs-voice-agent/about/core-concepts/speech-pipeline/speaker-diarization)
- Quality degrades beyond 4 speakers. For 5+ speakers, pyannote offline is recommended. — [ONNX port](https://huggingface.co/soniqo/Sortformer-Diarization-4spk-ONNX); [FluidInference docs](https://docs.fluidinference.com/diarization/sortformer)

**Nemotron 3 Diarization**
- Released under the OpenMDW 1.1 license for commercial use. Buffers range from 0.32 s for live agents to 30 s. — [Baseten library](https://baseten.co/library/nemotron-3-diarization). This is a secondary source; verify on the NVIDIA model card.

**diart**
- Overlap-aware online diarization: incremental clustering plus a rolling buffer updated every 500 ms, with latency tunable from 500 ms to 5 s. — [diart PyPI](https://pypi.org/project/diart/0.7.0/); [JOSS paper](https://theoj.org/joss-papers/joss.05266/10.21105.joss.05266.pdf)
- It relies on gated pyannote/segmentation and pyannote/embedding models, whose user conditions must be accepted. — [diart GitHub](https://github.com/hbredin/diart)

### Inferences
**Recommended pipeline**
1. Keep the raw 16 kHz mic audio in a ring buffer, aligned to Soniox token `start_ms`/`end_ms`.
2. For each Soniox `speaker` label, concatenate the audio of final tokens, dropping tokens that are near speaker changes or short.
3. Embed at 2 s, then every additional about 3 s, and update a running mean embedding for that label.
4. Assign the label to the best-matching enrolled person under the two-threshold rule. Keep the assignment sticky unless the evidence flips consistently.
5. Before the first confident ID (the first 1–2 s), treat the speaker as "likely X" from context, for example the last speaker or the wake-word speaker.

**Per-utterance check**
- For short single commands, embedding the whole utterance (the VAD segment) may beat diarization labels, because Soniox labels can flip early.
- If Soniox labels two people the same, a per-utterance embedding check catches it.

**Overlapping speech**
- Don't run identification on overlap. Sortformer and diart are overlap-aware; Soniox's per-token single label cannot express overlap.
- Mark low-confidence segments (rapid label flips, low token confidence) as "overlap/unknown".

**Cost**
- Soniox diarization is part of the STT stream, so there is no extra model cost.
- Adding a local embedding model costs a few ms to tens of ms of CPU per 2–3 s segment (not measured).

### Gaps
- Not found: published Soniox diarization accuracy (DER) for real-time Russian, or label stability across reconnects.
- Not found: DER for Streaming Sortformer on Russian.

## Q4. How do we handle Joyce's own voice (talking and singing): echo cancellation, barge-in, filtering her TTS?

### Takeaway
The main defense is **reference-based AEC** (WebRTC AEC3 in the browser, app, or native layer) fed with the exact audio Joyce plays. On top of it, add simple logic:
- Gate or flag STT tokens whose timestamps fall inside Joyce's playback window when they match her TTS text.
- Treat a speaker-embedding match to "Joyce" as a filter.
- Stop TTS fast on confirmed barge-in.

Singing and music with a beat are harder for AEC than speech. Expect residual echo and rely on the text and embedding filters.

### Cited Findings
**The failure mode**
- The mic picks up TTS playback, STT transcribes it as user input, and the LLM answers itself. — [Coval blog](https://www.coval.ai/blog/voice-ai-echo-cancellation) (vendor)

**How WebRTC AEC works**
- It models the reference (far-end) signal and subtracts the predicted echo. AEC3 is the current WebRTC canceller, inside the Audio Processing Module. — [Fora Soft article](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/echo-cancellation-aec-ai-hybrid-webrtc); [Simba Voice](https://simbavoice.ai/resources/echo-cancellation-in-real-time-voice-ai)
- An AEC3 delay hint `stream_delay_ms=0` lets AEC3 auto-estimate the delay. Set it manually only on hardware with large or fixed latency, for example Bluetooth. — [Strands Agents bidi audio docs](https://strandsagents.com/docs/api/python/strands.experimental.bidi.audio/)
- A Rust port, `aec3`, exists. — [docs.rs aec3](https://docs.rs/crate/aec3/0.4.0)

**Warm-up and double-talk**
- AEC needs a warm-up; one vendor cites a 3–4 s window at call start, with no measurement given. — [Coval](https://www.coval.dev/blog/voice-ai-echo-cancellation)
- Double-talk (Joyce talking while the user talks) is "imperfectly solved". Recommended practice is to stop TTS quickly on barge-in, because AEC works better after playback stops. — [Coval](https://www.coval.ai/blog/voice-ai-echo-cancellation)

**Where to cancel**
- Server-side AEC with the known TTS reference is possible: subtract server-side, then send cleaned audio to STT. — [Simba Voice](https://simbavoice.ai/resources/echo-cancellation-in-real-time-voice-ai)
- Patents describe combined reference signals for AEC during barge-in. — [US10013995B1](https://patents.google.com/patent/US10013995)

### Inferences
**Layered own-voice defense (cheap to expensive)**
1. **Hardware and OS AEC with a correct reference.**
   - Browser: `getUserMedia({audio:{echoCancellation:true}})` works only when the playback goes through the same browser/WebRTC audio path. TTS played by a separate app or device is not cancelled.
   - Android: `VOICE_COMMUNICATION` source + AcousticEchoCanceler.
   - iOS: VoiceProcessingIO.
   - Linux: PulseAudio/PipeWire module-echo-cancel (WebRTC).
   - Smart speaker hardware: a mic array with a loopback reference.
2. **Half-duplex gating.** While Joyce speaks, raise the VAD/barge-in threshold. Accept barge-in only when energy is well above the expected echo level, or a wake word or keyword is heard, or STT text differs from Joyce's own TTS text.
3. **Text echo filter.** Fuzzy-match Soniox tokens during playback (plus about 300 ms tail) against the TTS text or lyrics being played, and drop matches.
4. **Embedding filter.** Enroll Joyce's TTS voice as a "speaker" (it is fixed, so enrolling is trivial). Drop Soniox labels whose embedding matches Joyce.
5. **Neural residual-echo or noise suppression** (DeepFilterNet, Krisp, NVIDIA Maxine): optional, after AEC. These are noise suppressors and do not replace a linear AEC with a reference signal. This last point is not sourced here.

**Singing**
- Music has wide bandwidth and sustained tones, so AEC usually leaves more residual. Lyrics are known, so the text filter (3) is very effective.
- Consider ducking Joyce's volume while VAD detects user speech during singing.

### Gaps
- No primary, measured benchmark found for: WebRTC AEC3 ERLE, SpeexDSP vs AEC3, Krisp/Maxine AEC, or DeepFilterNet as an AEC.
- No sourced details found on browser, Android or iOS AEC limits.
- Most sources here are vendor blogs.

## Q5. Is anti-spoofing (replay and deepfake detection) needed, and what does it cost?

### Takeaway
For a home assistant, voice ID is a **personalization signal, not authentication**. Spoof detection (AASIST-class) adds cost, false rejects, and brittleness under codecs and compression.

So skip it for personalization. For sensitive actions (purchases, door locks), use a second factor (PIN, phone confirmation) instead of trusting voice. Even ASVspoof 5 winners degrade under adversarial attacks and codecs.

### Cited Findings
- **ASVspoof 5** evaluated 53 teams. Many solutions perform well, but performance **degrades under adversarial attacks and neural codec/compression**. — [ASVspoof 5 overview](https://arxiv.org/html/2601.03944v1); [ASVspoof 5 dataset](https://arxiv.org/html/2408.08739v1)
- In ASVspoof 5, attacks significantly compromise the baseline systems, and submissions improve on them substantially. — [ASVspoof 5 overview](https://arxiv.org/html/2601.03944v1)
- **Scalable AASIST** (2025) reports **7.6% EER** with a modified AASIST, versus a weaker re-implemented AASIST baseline. — [arXiv 2507.11777](https://arxiv.org/html/2507.11777v1)
- In XMUspeech's ASVspoof 5 system, input audio length strongly affects detection performance. — [XMUspeech ASVspoof 5](https://arxiv.org/html/2509.18102v1)
- ASVspoof 5 metrics now include minDCF and Cllr in addition to EER and t-DCF. — [Survey of AI-generated voice detection](https://arxiv.org/pdf/2608.15411)

### Inferences
- A detector with around 7% EER would wrongly reject roughly 1 in 14 genuine commands at the EER operating point. That is unacceptable for everyday UX.
- Also, Joyce's own TTS would be flagged as "synthetic" (useful only as a self-voice filter).
- **Cost/benefit:** not worth it in v1. Revisit if voice ever gates payments or locks.

### Gaps
- Official AASIST baseline EERs on ASVspoof 5 were not retrieved.
- CPU cost of AASIST was not retrieved. The original AASIST-L is known to be small, but this was not verified in this session.

## Q6. What privacy and legal rules apply to storing voice biometrics (GDPR, Kazakhstan, Russia)?

### Takeaway
Voiceprints used to identify a person are **biometric data** in all three regimes, so explicit consent is required.

Keep embeddings on-device, never store raw enrollment audio longer than needed, and provide delete/re-enroll. Each household member, including kids via their parents, should consent individually.

In Russia, 152-FZ Art. 11 requires **written** consent for biometric processing (with exceptions). Kazakhstan's law defines biometric data similarly and requires prior consent.

### Cited Findings
**Kazakhstan**
- Law No. 94-V defines biometric data as personal data characterizing physiological and biological features "on the basis of which may establish his (her) identity". — [Adilet: Law on Personal Data](https://adilet.zan.kz/eng/docs/Z1300000094/compare)
- Consent is required prior to collection (with law-enforcement/statistics exceptions). Cross-border transfer is allowed to countries with adequate protection. — [Hunton summary](https://hunton.com/privacy-and-information-security-law/kazakhstan-enacts-data-privacy-law); [Mondaq](https://www.mondaq.com/privacy-protection/283254/kazakhstan-introduces-new-privacy-law)
- A 2024/2025 amendment is said to classify biometrics as sensitive. This is unverified, from a secondary source. — [Clym summary](https://regulations.clym.io/law-no-94-v-kazakhstan)
- Academics note gaps in Kazakhstan's biometric protection framework. — [ENU law bulletin](https://bullaw.enu.kz/index.php/main/article/download/24/5/37)

**Russia**
- 152-FZ Art. 11 defines biometric personal data as physiological and biological features used to establish identity. Processing requires the **written consent** of the person, with exceptions in Part 2. — [Kontur](https://kontur.ru/articles/1124); [ppt.ru](https://ppt.ru/art/personal-data/kak-sostavit-soglasie-na-obrabotku-biometricheskikh-personalnykh-dannykh)
- Voice is treated as biometric in regulator guidance. — [ppt.ru](https://ppt.ru/art/personal-data/kak-sostavit-soglasie-na-obrabotku-biometricheskikh-personalnykh-dannykh)
- The EBS (Unified Biometric System) has a separate electronic consent regime under Government Decree 1067 (2022). — [v2b.ru](https://www.v2b.ru/articles/chto-takoe-biometricheskie-dannye-i-kak-s-nimi-rabotat/)
- These are secondary sources; check against consultant.ru or garant.ru and the current 572-FZ/152-FZ wording.

### Inferences
**Status of embeddings and household use**
- Under GDPR, Art. 9 treats biometric data "for the purpose of uniquely identifying a natural person" as a special category, and voiceprint embeddings fall in it. **This was not fetched in this session; cite the GDPR text.**
- A purely household use may fall under GDPR's household exemption for the family itself. The vendor (Joyce's developer), however, is a controller or processor if data touches its servers.
- **Embeddings are not anonymous.** They are a biometric template and can be linked back to the person.

**Design recommendations**
- Use explicit in-app consent per person.
- For Russian-jurisdiction users, provide a written or electronically signed consent form.
- Store embeddings on-device or encrypted; keep no raw audio after enrollment unless opted in.
- Provide one-tap deletion, and handle consent for minors via a parent.
- Data localization: Russia's 242-FZ requires storing Russian citizens' personal data in Russia. This is not fetched in this session; verify.

### Gaps
- Primary texts of 152-FZ Art. 11, GDPR Art. 9, and the Kazakhstan 2024–2025 amendments were not fetched.
- No legal analysis found specifically on household voice assistants storing local voiceprints.

## Q7. How can Joyce tell whether speech is addressed to her, and reject TV or radio voices?

### Takeaway
State-of-the-art device-directed speech detection (Apple) fuses acoustic features, ASR text and ASR confidence in a small LLM-based classifier. Fusion gives 20–40% fewer errors than any single signal.

A cheap practical version for Joyce:
- Wake word or name detection, plus a follow-up window.
- LLM text-intent check on the Soniox transcript ("is this addressed to Joyce?").
- Speaker-ID filter: unknown voice during TV-playing means likely TV.
- Optionally, TV audio reference/AEC if the TV is played through Joyce.

### Cited Findings
**Multi-signal LLM (Apple)**
- Fuses an audio-encoder prefix, ASR text, and ASR confidence/lattice features. EER is 38.9% better than text-only and 20.5% better than audio-only. — [Apple ML Research](https://machinelearning.apple.com/research/llm-device-directed-speech-detection); [arXiv 2312.03632](https://arxiv.org/pdf/2312.03632)

**Follow-up queries without a wake word (Apple)**
- Prompting an LLM with the first query's text plus ASR uncertainty gives **20–40% fewer false alarms at 10% fixed false rejects**. — [Apple research: device-directed follow-ups](https://machinelearning.apple.com/research/device-directed)

**FLoRA (Apple)**
- Low-rank adapters add audio to an LLM. 22% relative EER reduction over text-only, close to full fine-tuning. — [arXiv 2406.09617](https://arxiv.org/pdf/2406.09617)

### Inferences
**TV/radio rejection**
- TV voices are mostly "unknown speakers" to the embedding model, and TV audio is often far-field with a constant direction and music beds.
- **Rule:** an unknown speaker with no wake word, while media is playing, means ignore.
- Joyce's own playback should already be cancelled by AEC (Q4). An external TV is not cancelled unless its audio is routed through Joyce (or an HDMI/loopback reference is available).

**Phrasing and cost**
- Russian addressing cues are useful text features: the name "Джойс" (Joyce), imperative mood, and second-person "ты" (informal "you") with assistant verbs.
- A small LLM call on the final transcript (already in the pipeline) is the cheapest high-accuracy addressee classifier.

### Gaps
- Amazon Science papers on device-directed detection or false-wake suppression were not retrieved; there was one search and no results.
- Google's work on this was not retrieved.
- No measured numbers found for TV-speech rejection.

---

## Bottom line for the key questions

### Best fast, cheap, commercially licensable stack (about 5 members)
| Layer | Choice |
|---|---|
| STT and diarization | Soniox real-time, `speaker` labels |
| Embeddings | Local ONNX model: 3D-Speaker CAM++ or ERes2NetV2, or WeSpeaker ResNet34 / SimAM-ResNet34 (VoxBlink2, multilingual) |
| Scoring | Cosine + AS-Norm |
| Decision | Two thresholds, accumulating ≥2 s per Soniox label |
| Own voice | WebRTC AEC3 with Joyce's TTS reference, plus a text echo filter and a "Joyce voice" embedding filter |
| Addressee detection | Wake word / name + LLM directedness check |
| Anti-spoofing | None; use a second factor for sensitive actions |

Commercial turnkey alternative: Picovoice Eagle, with a vendor-claimed 0.18% EER and pricing not retrieved.

Licenses still to verify:
- 3D-Speaker and WeSpeaker weights: dataset-derived licenses (VoxCeleb CC BY 4.0, VoxBlink2 terms).
- pyannote: gated models.

### Expected accuracy and latency
**Accuracy**
- Public EERs: about 0.4–1.1% on VoxCeleb full utterances, about 1% at 3 s, about 1.5% at 2 s (ERes2NetV2).
- Worse for:
  - kids: +40–45% relative, and much worse for young children;
  - whisper and shouting;
  - cross-language: about 3× EER on unseen languages in TidyVoice.
- For 5 known adults in a quiet home, closed-set ID on ≥2 s is likely above 95%. This is an inference and must be measured.

**Latency**
- Identity is available about 2 s after a person starts speaking (the accumulation window), plus embedding compute (not measured; likely tens of ms on CPU for small models).
- Soniox label flips can delay a stable mapping.
