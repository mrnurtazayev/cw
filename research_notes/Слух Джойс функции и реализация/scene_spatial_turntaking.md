# Joyce's hearing: auditory scene, space and conversational timing (models, tools, hardware)

Scope: fast, cheap real-time perception for a home voice assistant ("Joyce") that already uses Soniox real-time STT, with Russian-speaking users. Current as of 2026-10-14. Sources are marked as follows:
- Unmarked: verified in this session by search or fetch.
- **[PK]**: from prior knowledge, with a canonical URL I did not re-fetch this session. The report writer should treat these as likely but unverified.
- **[DATED]**: the source is old enough that the details may have changed.

Several vendor domains (docs.livekit.io, livekit.com, daily.co, arxiv.org) could not be resolved by the fetch tool during this session, so some vendor claims come from search snippets only.

---

## Q1. What is the cheapest real-time stack (CPU, on-device or small server) per capability, with realistic latency numbers?

### Takeaway
One small x86 CPU or a Raspberry Pi 5-class device can run nearly everything in real time:
- **VAD:** Silero or TEN VAD.
- **Audio tagging:** an EfficientAT MobileNet (mn04 to mn10) or CED-Tiny/Mini, run on 1 s windows.
- **Turn detection:** Smart Turn v3.x (audio-only, ~8 MB, 10–100 ms on CPU). Soniox's built-in semantic endpointing comes at no extra cost.
- **Music:** AudD at about $5 per 1,000 song-ID requests when cloud ID is needed, and beat_this/BeatNet/Essentia locally.

Full-duplex speech-to-speech models such as Moshi (~200 ms) have the best timing but need a GPU, and they do not fit a Russian-language, Soniox-centred pipeline. The practical choice is a cascaded pipeline plus turn-taking heuristics.

### Cited Findings

#### 1. Sound event detection and audio tagging (AudioSet, 527 classes)

| Model | AudioSet mAP | Params | CPU / edge notes | License |
|---|---|---|---|---|
| EfficientAT mn04_as | 43.2 | 0.98 M (0.11 GMACs) | MobileNetV3 built for "resource constrained platforms"; no official latency figures | MIT |
| EfficientAT mn05_as / mn10_as | 44.3 / 47.1 | 1.43 M / 4.88 M (0.16 / 0.54 GMACs) | Compute depends mainly on spectrogram resolution (default 128 mels, 10 ms hop). Larger hops cut MACs: mn10 at hop 25 runs 0.22 GMACs at 44.7 mAP | MIT |
| EfficientAT dymn04 / dymn10 / dymn20 | 45.0 / 47.7 / 49.1 | 1.97 / 10.57 / 40.02 M | Dynamic MobileNets | MIT |
| EfficientAT mn40_as_ext | 48.7 | 68.4 M (8.03 GMACs) | — | MIT |
| CED-Tiny / Mini / Small / Base | 48.1 / 49.0 / 49.6 / 50.0 (AS-2M) | 5.5 / 9.6 / 22 / 86 M | 16 kHz input, 64 mels. ONNX export provided, with prebuilt Android APKs via sherpa-onnx. README says CED-Tiny "should be faster than MobileNets on a single x86 CPU" | Repo is GPL-3.0; model-weight license not separately stated |
| Whisper-AT (tiny / base / small / large-v1) | 36.5 / 37.6 / 39.8 / 42.1 | Whisper backbone plus a light head | Claims <1% extra compute on top of Whisper ASR (with a footnote saying not for all models). Tagging resolution is a multiple of 0.4 s, default 10 s | BSD-2-Clause |
| BEATs (iter3) | Reports "SOTA mAP 50.6% on AudioSet-2M" | 90 M | Too heavy for always-on CPU use | MIT [PK] |
| EAT-B / EAT-L | Claims SOTA on AS-20K and AS-2M (exact mAP not retrieved) | 88 M / 309 M | Teacher or server-class | — |
| Dasheng Base / 0.6B / 1.2B | AS mAP not retrieved | 86 M / 0.6 B / 1.2 B | On X-ARES, Dasheng 1.2B beats BEATs on FSD50k (0.455 vs 0.217) | — |
| PANNs CNN14 | 43.1 **[PK]** | ~81 M **[PK]** | Has a framewise "DecisionLevel" SED variant **[PK]** | MIT **[PK]** |
| AST | 45.9 single, 48.5 ensemble **[PK]** | ~87 M **[PK]** | Heavy for CPU streaming | BSD-3 **[PK]** |
| YAMNet | 30.6 **[PK]** | ~3.7 M **[PK]** | 0.96 s frames with 0.48 s hop; easy TFLite deployment | Apache-2.0 **[PK]** |

Sources:
- EfficientAT rows: [EfficientAT GitHub](https://github.com/fschmid56/EfficientAT). The EfficientAT paper says its models use about 10× fewer parameters and about 100× fewer MACs than the Transformer teachers: [arXiv 2211.04772](https://arxiv.org/pdf/2211.04772).
- CED: [CED GitHub](https://github.com/RicherMans/CED) and [arXiv 2308.11957](https://arxiv.org/abs/2308.11957).
- Whisper-AT: [Whisper-AT GitHub](https://github.com/YuanGongND/whisper-at). Its README notes that Whisper's representation is "NOT noise-invariant" and instead encodes the background sound type. That is why a tagging head comes almost free on top of Whisper.
- BEATs: [arXiv 2212.09058](https://arxiv.org/pdf/2212.09058).
- EAT and Dasheng parameter counts: [OpenBEATs comparison, arXiv 2507.14129](https://arxiv.org/pdf/2507.14129). EAT SOTA claim: [EAT repo mirror](https://github.com/standardgalactic/EAT). Dasheng X-ARES result: [CMU-AIST ICME 2025 report](https://arxiv.org/html/2601.16273v1). The OpenBEATs linear-probe sound average ranks Dasheng-1.2B 0.57, Dasheng-0.6B 0.56, BEATs 0.51 and EAT-B 0.43.
- PANNs: [arXiv 1912.10211](https://arxiv.org/abs/1912.10211), [repo](https://github.com/qiuqiangkong/audioset_tagging_cnn).
- AST: [arXiv 2104.01778](https://arxiv.org/abs/2104.01778).
- YAMNet: [TF models/yamnet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet).

Zero-shot and CLAP:
- LAION-CLAP and MS-CLAP allow text-prompted, open-vocabulary detection, such as "kettle whistling".
- I could not verify their zero-shot ESC-50 numbers this session. The only figure found was an aggregator's "CLAP (LAION) 96.7% ESC-50, 86M params", which is probably not zero-shot ([codesota](https://codesota.com/audio/classification)). See Gaps.

Home-relevant AudioSet classes **[PK]**:
- The ontology includes: Doorbell, Ding-dong, Knock, Smoke detector/smoke alarm, Fire alarm, Alarm clock, Siren, Baby cry/infant cry, Dog, Bark, Cat, Glass, Shatter, Water tap/faucet, Sink (filling or washing), Television, Music, Applause, Speech, Whispering, Shout, Yell, Screaming, Singing, Laughter, Baby laughter, Cough, Snoring, Microwave oven, Vacuum cleaner, Telephone bell ringing and Ringtone.
- It also has scene-like classes: "Inside, small room", "Inside, large room or hall", "Outside, urban or manmade", "Vehicle" and "Car".
- Source: [AudioSet ontology](https://research.google.com/audioset/ontology/index.html).

Commercial reference points:
- **Apple Sound Recognition** (on-device since iOS 14) covers fire/smoke alarms, sirens, cats, dogs, appliances, car horns, doorbells, door knocks, running water and baby crying. Some lists add shouting and coughing.
  - iOS 18 adds user-trained custom alarms, appliances and doorbells.
  - Apple warns not to rely on it in emergencies.
  - Turning it on disables "Hey Siri".
  - Users report it is oversensitive to doorbell-like sounds in noise.
  - Sources: [Tom's Guide](https://tomsguide.com/how-to/how-to-enable-sound-recognition-on-iphone), [AbilityNet iOS 18](https://mcmw.abilitynet.org.uk/how-to-detect-sounds-using-your-iphone-or-ipad-with-ios-18), [iMore](https://imore.com/sound-recognition-ios-14-will-alert-deaf-users-sirens-knocking-more).
- **Alexa Guard** is discontinued **[DATED, ~2023]**. It was free and listened for smoke alarms and glass breaking, but only in "away" mode. It was replaced by Alexa Emergency Assist at $5.99/month, which listens continuously: [PCWorld](https://www.pcworld.com/article/2082386/alexa-guard-discountinued.html).
- **Audio Analytic** (ai3 sound recognition for smoke alarms, glass break and baby cry) was acquired by Meta in November 2022 and folded into Reality Labs Research. It no longer exists as an independent licensable SDK vendor: [tech.eu](https://tech.eu/2022/11/09/meta-acquires-audio-analytic/).
- **Google Nest:** I did not verify its current sound-event features this session (see Gaps).

#### 2. Acoustic environment: noise/SNR, scene, reverb, distance
- **DCASE 2025 Task 1** (low-complexity acoustic scene classification) had these rules:
  - At most 128 kB of parameters and 30 MMACs per inference.
  - Training on 25% of TAU Urban Acoustic Scenes 2022 Mobile, which has 10 urban scenes and about 64 h of audio.
  - Device IDs available at inference.
  - The baseline scored 50.72% → 51.89% with device-specific fine-tuning. Distillation systems reached 57.9–59.3% on the dev set.
  - Sources: [arXiv 2507.19557](https://arxiv.org/html/2507.19557v1), [arXiv 2509.09262](https://arxiv.org/html/2509.09262v1).
  - These scenes are *urban* (airport, metro, park and so on), not home/car/indoor-outdoor, so models would need retraining or relabelling.
- **Brouhaha** **[PK]** is an open pyannote-based model that jointly predicts VAD, frame-level SNR and C50 (clarity, a reverberation measure) from a single mic. It is the most direct "noise level + reverb" estimator for this pipeline: [brouhaha-vad GitHub](https://github.com/marianne-m/brouhaha-vad).
- WADA-SNR and DNSMOS are lighter or more established alternatives **[PK]**.

#### 3. Conversation timing: VAD, end-of-turn, barge-in, backchannels, full-duplex

VAD:
- **TEN VAD:**
  - Real-time factor (RTF) by device:

    | Device | TEN VAD RTF | Silero RTF |
    |---|---|---|
    | Xeon Gold 6348 | 0.0086 | 0.0127 |
    | Ryzen 9 5900X | 0.0150 | not listed |
    | M1 | 0.0160 | not listed |
    | iPhone 8 | 0.0050 | not listed |
    | Low-end Android | 0.049–0.057 | not listed |
  - Library size is about 277–731 KB, against Silero's 2.16–2.22 MB.
  - It uses 16 kHz input with a 10 or 16 ms hop.
  - License is "Apache 2.0 with additional conditions" (worth checking).
  - Vendor claim: it detects speech-to-silence transitions faster, while Silero has "a delay of several hundred milliseconds". This is not independently verified.
  - Sources: [TEN VAD GitHub](https://github.com/TEN-framework/ten-vad), [podcast transcript with inconsistent latency numbers](https://podcast.convoai.world/episodes/ep-007-convo-ai-world-podcast-transcript.txt).
- **Silero VAD v6.x:**
  - v6.2 has a fully reworked training pipeline and better stability on edge cases, with modest user-visible changes: [Habr (author)](https://habr.com/en/users/snakers4/posts).
  - whisper.cpp ships silero-v6.2.0: [whisper.cpp PR](https://github.com/ggml-org/whisper.cpp/pull/3524/files).
  - MIT license, under 1 ms per 30 ms chunk on one CPU thread **[PK]**: [silero-vad](https://github.com/snakers4/silero-vad).
- **Picovoice Cobra** needs an AccessKey. A "Forever-Free" plan covers commercial use within limits, but Cobra's exact quota was not visible: [Picovoice pricing](https://picovoice.ai/pricing), [pvcobra PyPI](https://pypi.org/project/pvcobra).
  - The 2021 free tier was "up to 3 users" **[DATED]**: [CNX](https://www.cnx-software.com/2021/12/08/picovoice-offline-voice-ai-engine-gets-free-tier-for-up-to-3-users/).
  - A third-party aggregator claims the free tier ends 2026-06-30, which contradicts the official page: [stork.ai](https://www.stork.ai/en/picovoice).
- **WebRTC VAD** is a GMM-based model: extremely cheap but the least accurate **[PK]**.

Semantic end-of-turn detection (key point for Russian):
- **Pipecat Smart Turn v3.x (v3.2 is latest):**
  - Audio-only, based on Whisper Tiny plus a linear classifier, about 8 M parameters.
  - Builds: int8 CPU at 8 MB and fp32 GPU at 32 MB. The GPU build is about 1% more accurate.
  - Input is up to 8 s of 16 kHz PCM.
  - CPU latency is "as little as 10 ms on some CPUs, under 100 ms on most cloud instances", and about 65 ms on Pipecat Cloud.
  - Supports 23 languages **including Russian and Ukrainian**. License is BSD-2-Clause.
  - It publishes no absolute or per-language accuracy figures.
  - Source: [smart-turn GitHub](https://github.com/pipecat-ai/smart-turn).
  - Daily's launch post gives 12 ms on modern CPUs and 60 ms on a low-cost AWS instance: [Daily blog](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/).
  - A third-party ONNX re-export of v3.2 measured 36.3 ms per 8 s window (int8, 11.1 MB) on an Apple M5 Pro CPU with 2 threads: [soniqo/Smart-Turn-v3.2-ONNX](https://huggingface.co/soniqo/Smart-Turn-v3.2-ONNX).
  - The only Russian figure is from older v2: 93.0% on a synthetic set: [Smart Turn v2 card](https://huggingface.co/pipecat-ai/smart-turn-v2).
- **LiveKit turn detector (MultilingualModel):**
  - Text-based, using the STT transcript; built on Qwen2.5-0.5B.
  - About 14 languages including Russian, about 400 MB RAM, about 25 ms inference: [package listing](https://simple-repository.app.cern.ch/project/livekit-plugins-turn-detector), [LiveKit docs](https://docs.livekit.io/agents/logic/turns/turn-detector).
  - A mirrored model card reports Russian TPR 99.3% / TNR 88.0%. The mirror is not an official domain: [mirror](https://huggingface-gw-kz.just-ai.com/livekit/turn-detector).
  - LiveKit says an improved end-of-turn model cut interruptions by 39%; I could not fetch the details: [LiveKit blog](https://livekit.com/blog/improved-end-of-turn-model-cuts-voice-ai-interruptions-39).
  - The weights are under the "LiveKit Model License", which ties use to LiveKit Agents **[PK, verify]**.
- **TEN Turn Detection:**
  - Text-based, built on **Qwen2.5-7B**, with three classes: finished, wait and unfinished.
  - Self-reported accuracy: English finished 90.64% / unfinished 98.44% / wait 91%; Chinese 98.90% / 92.74% / 92%.
  - **Only English and Chinese** are documented, so it is not suitable for Russian without fine-tuning. At 7B parameters it needs a GPU.
  - Sources: [HF TEN_Turn_Detection](https://huggingface.co/TEN-framework/TEN_Turn_Detection), [GitHub](https://github.com/TEN-framework/ten-turn-detection).
- **Soniox endpoint detection (already in Joyce's stack):**
  - Enabled with `enable_endpoint_detection`. It emits an `<end>` token and finalizes tokens.
  - It uses the speech model itself, listening to intonation, pauses and context.
  - Latency profile, sensitivity and maximum delay are tunable.
  - Included in real-time pricing at $0.12/h.
  - Semantic endpointing arrived with v4 real-time (Feb 2026), and v5 has faster, tunable semantic endpointing. All claims are from the vendor, with no independent benchmarks.
  - Sources: [Soniox turn-detection page](https://soniox.com/speech-to-text/features/turn-detection), [Soniox docs](https://soniox.com/docs/stt/rt/endpoint-detection), [Soniox v4 blog](https://soniox.com/blog/2026-02-05-soniox-v4-real-time), [v5 article](https://www.voiceaispace.com/articles/soniox-v5-real-time-follows-a-live-conversation-not-just-the-words), [Bolna docs (stt-rt-v5)](https://www.bolna.ai/docs/providers/transcriber/soniox).

Backchannels ("ага", "угу"):
- Inoue et al. (NAACL 2025) fine-tuned a Voice Activity Projection (VAP) model to predict *when* and *which* backchannel to produce, continuously and in real time. Zero-shot VAP was sensitive to the threshold: [arXiv 2410.15929](https://arxiv.org/abs/2410.15929).
- Paierl et al. (Interspeech 2025) used acoustic-only features with LightGBM to reach about 130 ms MAE on backchannel timing in real time. The key cues were speaking duration, intensity and F0 in the window 275–875 ms before the backchannel: [ISCA](https://www.isca-archive.org/interspeech_2025/paierl25_interspeech.html).
- No commercial voice-agent framework was found that ships backchannel generation as a feature.

Full-duplex:
- **Moshi** (Kyutai) runs the user's and its own speech as parallel streams. Latency is 160 ms theoretical and about 200 ms in practice: [arXiv 2410.00037](https://arxiv.org/html/2410.00037v1).
  - It is a 7B model that needs a GPU and is essentially English-only **[PK]**.
- Sesame's open release, CSM-1B, is a TTS/speech-generation model, not an open full-duplex conversational agent **[PK]**.
- I found nothing reliable on other 2026 open full-duplex models (see Gaps).

#### 4. Target speaker / personalised enhancement (cocktail party, single mic)
- **Personalized PercepNet** (Amazon, Interspeech 2021) is real-time and as cheap as PercepNet. The base PercepNet uses under 5% of one CPU core at fullband: [arXiv 2106.04129](https://arxiv.org/pdf/2106.04129).
- **Personalized DeepFilterNet2** (2024 paper) adds a speaker embedding with "minimal computational overhead". The paper's CPU timings were not retrieved: [arXiv 2404.08022](https://arxiv.org/html/2404.08022v1).
- No turnkey open-source pDeepFilterNet release was confirmed this session.

#### 5. Music perception
- **AudD:** $5 per 1,000 file-recognition requests, $45 per stream per month for 24/7 monitoring, 300 free requests. An Indie plan is reported at $5/month for 1,000 recognitions. Sources: [AudD (via APIs listing)](https://providers.apievangelist.com/providers/audd/), [Raycast extension](https://www.raycast.com/erenalparslan/music-recognizer).
- **ACRCloud:** prices are behind a login, with a 14-day free trial and up to 20% off for following its branding guidelines: [ACRCloud docs](https://docs.acrcloud.com/get-started/tutorials/recognize-music), [apis.io plans](https://apis.io/plans/acrcloud/acrcloud-plans-pricing/).
- **ShazamKit:** no extra cost for Apple Developer Program members ($99/yr individual, $299/yr organization). It has an Android SDK and custom on-device catalogues. No published rate limits. Sources: [Apple ShazamKit](https://developer.apple.com/shazamkit), [apis.io](https://apis.io/plans/shazam/shazam-plans-pricing/), [Apple forum](https://developer.apple.com/forums/thread/728654).
  - The unofficial RapidAPI "Shazam" API costs $24.99/month for 2,000 requests.
- **Beat and tempo:**
  - BeatNet supports streaming from a mic and causal online/real-time modes at 22.05 kHz: [BeatNet](https://github.com/davies-w/BeatNet).
  - madmom listed streaming as future work in its paper: [arXiv 1605.07008](https://arxiv.org/pdf/1605.07008).
  - beat_this (CPJKU, ISMIR 2024) is offline, with optional madmom DBN post-processing: [Replicate](https://replicate.com/xavriley/beat_this/versions/26142842c6dc94673820f0a9762214fa7109015d966031d8e6137bef4fd14323/api).
  - Essentia (C++ with Python bindings, AGPL-3.0 **[PK]**) has RhythmExtractor2013, KeyExtractor and TempoCNN **[PK]**.
  - madmom's pretrained models are CC BY-NC-SA **[PK]**, so they cannot be used commercially.

### Inferences
- **Cheapest per capability:**

| Capability | Cheapest recommended choice | Cost / latency |
|---|---|---|
| Sound events | EfficientAT mn05/mn10 (MIT) or CED-Mini via ONNX, on 1 s windows with 0.5 s hop. Add class-specific thresholds and N-of-M temporal smoothing | Under 0.6 GMACs per window, roughly 5–20 ms on a modern x86 core (estimate). Detection latency ~0.5–1 s, dominated by the window. YAMNet is the simplest TFLite fallback but is much less accurate (30.6 vs 47 mAP) |
| Open-vocabulary sounds | CLAP on a small server, run only when a cheap tagger fires "unknown/other" | GPU or heavy CPU |
| Noise / reverb | Brouhaha (SNR + C50 + VAD) every few seconds, or an RMS dB meter plus VAD-gated noise-floor tracking | Nearly free |
| Indoor / outdoor / car | AudioSet "Inside, small room", "Vehicle" and "Outside" logits from the same tagger | No extra model |
| VAD | Silero v6 (MIT) or TEN VAD (smaller, claims faster offset detection) | Free, sub-millisecond per frame |
| End of turn | (1) Soniox semantic endpoint as primary (no extra model, cost or language risk). (2) Smart Turn v3.2 on the last 8 s of audio after a VAD pause of about 200 ms, as an audio-prosody second opinion (BSD, Russian-listed). (3) Optionally a text EOU check on the Soniox transcript | (1) included at $0.12/h. (2) 10–60 ms CPU |
| Music | ACRCloud/AudD only when the tagger's "Music" score is high, rate-limited (e.g. one ID per 30–60 s per music episode) | Song ID ≈ $0.005 per request at AudD. Local beat/tempo free |

- **End-of-turn notes:**
  - Do not use TEN Turn Detection: it is English/Chinese only and needs a 7B model.
  - LiveKit's detector supports Russian but its license probably ties it to the LiveKit framework.
- **Singing-along detection:** combine the "Singing" class with "Music" plus Soniox transcript similarity to lyrics, or a simple detector for the user's voice over music. This is an inference with no model found.
- **Barge-in:** reliability depends mainly on acoustic echo cancellation (AEC) of Joyce's own TTS. Without AEC, the VAD fires on Joyce's own voice. Recommended logic:
  - Pause the TTS on VAD onset combined with at least 2–3 words from Soniox.
  - Ignore backchannel-only words ("ага", "угу", "да-да") so they do not count as interruptions.
- **Backchannels:** the cheapest route is a rule or LightGBM model on prosody (pause above about 300 ms inside a long user turn, falling F0, Smart Turn "incomplete" probability) that emits short "угу"/"ага" clips at low volume. Training a VAP model is research-grade.
- **Full-duplex:** Moshi-class systems give the most human-like timing but are GPU-bound, English-centred and would replace Soniox. They are not cheap for a Russian home assistant in 2026.

### Gaps
- Per-class AP for home classes (doorbell, smoke alarm, glass shatter, baby cry, water tap and others) from EfficientAT, CED, PANNs or YAMNet was not retrieved. These are published in per-class result files and papers. Expect strong results for speech, music, dog, baby cry, siren and smoke alarm, and weaker results for knock, glass shatter and water tap (inference, not verified).
- Measured CPU latency (ms per window) for EfficientAT and CED on Raspberry Pi or x86 is not published in the repos. It needs benchmarking.
- Zero-shot ESC-50 accuracy for LAION-CLAP and MS-CLAP was not verified this session.
- Exact AudioSet mAP for EAT and Dasheng was not retrieved.
- Google Nest's current sound-detection features (smoke/CO alarm, glass break with Nest Aware) and their status in 2026 were not verified.
- No independent benchmark of TEN VAD vs Silero end-of-speech latency was found; the claims are vendor-only.
- No independent benchmark of Soniox v4/v5 semantic endpointing on Russian was found, and Smart Turn v3 has no official Russian accuracy figure.
- Current open full-duplex alternatives with Russian support (for example any 2026 releases) and Sesame's current status were not verified.
- ACRCloud's actual per-request price is not public.
- No source covers blind room T60 or speaker-distance estimation as a packaged real-time tool. Research methods exist (C50/DRR estimation), but there is no turnkey library confirmed beyond Brouhaha (C50).

---

## Q2. Which hardware (mic array) choices unlock spatial abilities and at what cost?

### Takeaway
A 2-mic pair, as on a phone or laptop, gives only an ambiguous left/right angle because front and back produce the same time delay. That is enough for coarse "which side" or light beamforming, not 360° localisation. A 4-mic circular array with on-board DSP fixes this: the Seeed ReSpeaker XVF3800 costs about £38–48 / A$113–125 and has hardware DoA, multi-beam forming, AEC and noise suppression up to about 5 m. Combined with open-source ODAS or pyroomacoustics (SRP-PHAT/MUSIC), it gives 360° DoA, speaker tracking and focus on one talker.

### Cited Findings
- With two microphones, mirror-image source angles around the mic axis give identical TDOA (front/back ambiguity). Each delay maps to two possible angles except at end-fire, and angular resolution is quantized by spacing and sample rate (about 4° in one example). Sources: [DSP Concepts GCC-DOA docs](https://documentation.dspconcepts.com/awe-designer/8.D.2.6/generalized-cross-correlation-direction-of-arrival), [Acta Acustica 2021](https://acta-acustica.edpsciences.org/articles/aacus/pdf/2021/01/aacus210004.pdf).
- A thesis on 2-mic smartphone localisation found that resolving front/back with orientation sensors was unsatisfactory and recommended a third microphone for 360°: [UCT thesis](https://open.uct.ac.za/handle/11427/24295).
- GCC-PHAT is the standard TDOA estimator and is more robust to noise than plain cross-correlation: [IPSJ](https://ipsj.ixsq.nii.ac.jp/record/108431/files/IPSJ-Z73-4C-1.pdf).
- Google's **SpeechCompass** (2025) used multi-microphone localisation on a phone-form-factor device for diarisation and direction cues in mobile captioning. Details were not fetched this session: [arXiv 2502.08848](https://arxiv.org/pdf/2502.08848).
- **ReSpeaker XMOS XVF3800** (4-mic circular, USB or with XIAO ESP32S3):
  - Advertised features: 360° DoA, multi-beamforming, AEC, AGC, noise suppression, about 5 m pickup.
  - Prices: Pi Hut £38.32 on sale (£47.90 regular, often out of stock); Pakronics A$112.88 for USB and A$125.47 for the cased ESP32S3 version; Switch Science about ¥10.4–11.4k.
  - The **reSpeaker Flex XVF3800** comes in linear or circular split-board versions at about £49–58.50.
  - Sources: [The Pi Hut](https://thepihut.com/products/respeaker-xmos-xvf3800-ai-powered-4-mic-array-for-clear-voice-even-in-noise), [Pakronics](https://pakronics.com.au/products/respeaker-xvf3800-usb-4-mic-array-with-aec-agc-beamforming-doa-noise-suppression-360-far-field-voice-processing-for-xiao-esp32s3-raspberry-pi-jetson-device-ss101991441), [Switch Science](https://www.switch-science.com/collections/all/products/10749), [Flex linear](https://thepihut.com/products/respeaker-flex-xvf3800-linear-with-xiao-esp32s3).
- **ODAS** (IntRoLab) **[PK]** is an open-source C library for localisation (SRP-PHAT), tracking and separation of multiple sources. It ships configs for ReSpeaker and Matrix arrays (MIT license): [ODAS GitHub](https://github.com/introlab/odas).
- **pyroomacoustics** **[PK]** is a Python library (MIT) with DoA algorithms (SRP-PHAT, MUSIC, NormMUSIC, TOPS, CSSM, WAVES, FRIDA), beamformers (delay-and-sum, MVDR) and room simulation for synthetic training data: [pyroomacoustics GitHub](https://github.com/LCAV/pyroomacoustics).
- Older ReSpeaker boards include the USB Mic Array v2.0 (XVF-3000, 4 mics), the 4-Mic and 6-Mic Pi HATs (raw mics, no DSP) and the 2-Mic Pi HAT **[PK, prices and availability not verified, likely superseded by XVF3800]**.
- Fewer microphones (and head or body diffraction) degrade localisation performance: [Osaka SIG-Challenge B202](https://www.osaka-kyoiku.ac.jp/~challeng/SIG-Challenge-B202/B202-12.pdf).
- With one mic and no array, target-speaker extraction falls back to speaker-embedding methods: Personalized PercepNet at under 5% of one core, or personalised DeepFilterNet2 ([arXiv 2106.04129](https://arxiv.org/pdf/2106.04129), [arXiv 2404.08022](https://arxiv.org/html/2404.08022v1)).

### Inferences
- **Tiered recommendation:**

| Tier | Hardware | Approx. cost | What it unlocks |
|---|---|---|---|
| 0 | Phone or laptop with 1–2 mics | $0 | Tagging, VAD, turn-taking, SNR. Coarse left/right only. Use OS AEC. Cocktail-party handled only by speaker-embedding enhancement plus diarisation |
| 1 (recommended) | ReSpeaker XVF3800 USB on a Pi 5 or mini-PC | ≈ $50–80 board plus host | Hardware AEC, which is critical for barge-in while Joyce speaks; 360° DoA readable over USB control; fixed beams toward the talker; about 5 m far-field pickup. It also offloads DSP from the host CPU |
| 2 | Raw 4/6-mic array (ReSpeaker 6-mic HAT or a UMA-8-style array) plus ODAS or pyroomacoustics | Cheap hardware, more DIY | Multi-source tracking (two people talking plus TV), MVDR beamforming per tracked source, DoA on non-speech events ("the knock came from the hallway"). More engineering and CPU |

- **Fusing direction with events:** DoA becomes useful when fused with the event tagger and diarisation. Examples:
  - Tag "TV" from a fixed direction and learn that as the TV's position, then suppress it.
  - Answer the user in the direction they are speaking from.
  - Ignore speech from the TV direction for wake and turn-taking.
- **Distance to the mic:** can be approximated from level plus direct-to-reverberant cues (C50 via Brouhaha). An array improves this via coherence between mics, but no off-the-shelf "distance" estimator was found.

### Gaps
- No official Seeed price or spec sheet was fetched; only retailer listings. Real DoA accuracy and update rate of the XVF3800 in a reverberant living room are not independently measured.
- ODAS and pyroomacoustics CPU cost on a Pi 5, and their maintenance status in 2026, were not verified.
- SpeechCompass accuracy numbers were not retrieved (fetch failed).
- iPhone/Android APIs for multi-mic raw capture (needed for phone-based DoA) were not researched. In practice, mobile OSes often expose only processed mono or stereo.
