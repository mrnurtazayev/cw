# Human auditory capabilities → use cases for the voice companion "Joyce" (catalog, as of Oct 2026)

Context: Joyce is a home/personal voice companion that talks with emotion and can sing. Current stack: Soniox real-time STT → fast LLM → Google TTS. Users speak Russian, and possibly Kazakh and English.

Rating legend used in the catalog below:
- **Wow**: 1 (barely noticeable) to 5 (users will tell friends).
- **Feasibility in 2026**: Easy (off-the-shelf or a simple DSP/LLM heuristic), Medium (needs an extra model or tuning), Hard (unreliable or needs custom data), Research (not production-ready).
- **Lang**: language-dependent (L), language-independent (I), or partly dependent (P).

---

## Q1. What are the distinct perceptual abilities of human hearing that matter for conversation and home life, and how does each map to Joyce?

### Takeaway
Human hearing goes well beyond "what words were said". It covers roughly 10 families: speech content, paralinguistics, non-verbal vocalizations, music, speaker identity, conversational dynamics, auditory scene, spatial hearing, attention/habituation, and voice memory. Joyce's current pipeline (text-only STT → LLM) captures only the first family, plus language ID and diarization if Soniox flags are enabled. Most paralinguistic, vocal-burst, and scene abilities need either (a) a native-audio model, as in Gemini Live, gpt-realtime, or Hume EVI, or (b) parallel side-channel classifiers that feed tags to the LLM.

### Cited Findings
- **Computational paralinguistics as a task taxonomy.** The Interspeech ComParE challenge (since 2009) has benchmarked these tasks, grouped by year:
  - Social signals (laughter, fillers), conflict, emotion, autism (2013).
  - Degree of nativeness, Parkinson's, eating condition (2015).
  - Deception, sincerity, native language (2016).
  - Atypical affect, self-assessed affect, crying, heart beats (2018).
  - Styrian dialects, continuous sleepiness, baby sounds, orca activity (2019).
  - Elderly emotion, breathing, mask (2020).

  Sources: [auditory.org 2013 CfP](https://www.auditory.org/mhonarc/2013/msg00000.html); [auditory.org 2015](https://auditory.org/mhonarc/2015/msg00038.html); [auditory.org 2016](https://auditory.org/mhonarc/2016/msg00062.html); [Interspeech 2018 special sessions](https://www.interspeech2018.org/program-special-sessions.html); [auditory.org 2019 ComParE](https://www.auditory.org/postings/2019/137.html). The 2014, 2017 and 2021+ task lists were not retrieved; see Gaps.
- **Prosody-based emotion.** Hume's speech prosody model measures 48 dimensions of emotional expression from non-linguistic qualities of speech (pitch, pace, intensity). It uses the same 48 labels as its vocal-burst model, and output can be per word, sentence, utterance or turn. Hume's docs also announce a sunset of the standalone Expression Measurement API: Playground job creation ended 14 May 2026 and API use ended 14 June 2026. — [Hume prosody docs](https://dev.hume.ai/docs/expression-measurement/models/prosody); [Hume models](https://dev.hume.ai/docs/expression-measurement/models)
- **Melody matching.** Melody can be recognized independently of timbre and instruments. Google's Hum to Search builds an embedding of a melody directly from a spectrogram, matches hums, whistles or singing to polyphonic recordings, and does not need a perfect pitch from the user (Google Research, Nov 2020 — older info). — [Google Research blog](https://research.google/blog/the-machine-learning-behind-hum-to-search/?hl=en)
- **Addressee detection.** Detecting whether speech is directed at the device without a wake word is a recognized research problem. Apple researchers called it "significantly more challenging than voice trigger detection" and explored LLM-based multimodal models that combine ASR hypotheses and decoder signals (2023). No shipped wake-word-free Siri mode was found. — [arXiv 2312.03632](https://arxiv.org/html/2312.03632v1); [Apple ML research](https://machinelearning.apple.com/research/device-directed-speech)
- **What Soniox already gives Joyce.** Soniox real-time offers:
  - Speaker diarization: up to 15 speakers, 60+ languages, a speaker label on every token, off by default.
  - Automatic language ID that handles mid-conversation language switches.
  - Endpoint detection: `max_endpoint_delay_ms` 500–3000; `endpoint_sensitivity` requires stt-rt-v5.

  Sources: [Soniox diarization](https://soniox.com/speech-to-text/features/diarization); [LiveKit Soniox plugin](https://docs.livekit.io/agents/models/stt/plugins/soniox/); [Pipecat Soniox](https://reference-server.pipecat.ai/en/latest/api/pipecat.services.soniox.stt.html); [Soniox vs Speechmatics](https://soniox.com/compare-stt/soniox-vs-speechmatics)
- **Semantic end-of-turn.** OpenAI's semantic VAD uses a semantic classifier to estimate whether the user has finished, based on the words uttered. A trailing-off, hesitant utterance gets a longer wait. — [OpenAI Realtime VAD docs](https://developers.openai.com/api/docs/guides/realtime-vad)

### Inferences — THE CATALOG (synthesis)
This is my synthesis. Product facts are cited in Q2. Feasibility and wow ratings are my judgment for a Soniox + LLM + TTS stack in Oct 2026, assuming extra models may be added.

#### A. Speech content (what is said)
| # | Human ability | Joyce use case | Wow | Feasibility | Lang |
|---|---|---|---|---|---|
| A1 | Word recognition in noise; filling in masked words ("phonemic restoration") | Robust STT in a kitchen or with the TV on. Ask a targeted clarification ("ты сказал 'в семь' или 'в восемь'?") instead of a generic "не поняла". | 2 | Easy (Soniox) | L |
| A2 | Language/code-switch recognition (RU↔KZ↔EN mid-sentence) | Answer in the language the user used. Mirror Russian–Kazakh mixing naturally. Do not "correct" the user. | 4 | Easy–Medium (Soniox LID; Kazakh quality unverified) | L |
| A3 | Understanding disfluent and self-corrected speech ("в пять… нет, в шесть") | Use the corrected value. Do not echo the error. | 3 | Easy (LLM) | L |
| A4 | Understanding semantics, pragmatics and indirect requests ("что-то холодно" = turn up the heating) | Act on indirect hints. | 4 | Easy–Medium (LLM) | L |
| A5 | Prosodic disambiguation: stress, focus, question vs statement intonation (Russian "Ты пришёл." vs "Ты пришёл?" differ only in intonation) | Detect questions without a question word. Detect contrastive focus. | 3 | Medium. Text STT loses this; Soniox punctuation partly recovers it; native-audio models do better. | P |
| A6 | Recognition of names, rare words, the user's own vocabulary | Learn family names and pet names; custom vocabulary or context. | 3 | Easy (Soniox context and custom vocabulary) | L |

#### B. Paralinguistics (how it is said)
| # | Human ability | Joyce use case | Wow | Feasibility | Lang |
|---|---|---|---|---|---|
| B1 | Loudness / vocal effort | Match volume: the user speaks softly late at night, so Joyce lowers volume. A loud room makes Joyce louder (cf. Alexa Adaptive Volume). | 3 | Easy (RMS/LUFS of the user's speech vs noise floor; TTS gain) | I |
| B2 | Whisper detection | User whispers → Joyce whispers back and stays in "night mode" until normal voice returns (cf. Alexa Whisper Mode). Whispered speech has no F0 (unvoiced), so it is detectable acoustically. | 5 | Easy to detect; Medium on the output side. Google TTS whisper style must be checked (see Gaps). Fallback: lower volume and a slower, softer voice. | I |
| B3 | Shout / raised voice / anger in voice | Calm, short, de-escalating answers. If the shout is aimed at someone else (family argument), stay silent. | 3 | Medium | P |
| B4 | Pitch level and contour (F0), including monotone vs lively | Feeds mood estimation. Lets Joyce mirror energy level. | 2 | Easy (pitch tracker) | I (but interpretation is P) |
| B5 | Speech rate / tempo | Fast, rushed user → Joyce answers briefly and fast. Slow, elderly user → Joyce slows down and pauses more. | 4 | Easy (syllables/sec from STT timestamps) | P |
| B6 | Emotion: valence/arousal and categories (joy, sadness, anger, fear, tenderness, boredom) | "Ты сегодня какая-то грустная — всё нормально?" Adapt tone. Celebrate along with good news. | 5 | Medium (native-audio models or an SER side model). Reliability is limited; see Q4. | P (culture-dependent) |
| B7 | Sarcasm/irony (mismatch between prosody and words, e.g. a flat "ну просто супер") | Do not answer sarcasm literally ("рада, что вам понравилось!" = fail). | 4 | Hard. Needs prosody plus text plus context. Native-audio LLMs are partly capable. | L+P |
| B8 | Confidence/uncertainty (rising intonation, hedges, "наверное…") | Offer options or confirm before acting on an uncertain command. | 3 | Medium | P |
| B9 | Hesitation / thinking pauses / filled pauses ("э-э", "ммм") | Do not interrupt while the user thinks (longer endpoint). Help find a word they are searching for. | 4 | Easy–Medium (Soniox endpoint sensitivity or semantic VAD; fillers in transcript) | P |
| B10 | Fatigue / sleepiness (ComParE "sleepiness" task) | "Ты звучишь уставшим — может, отложим и ляжешь пораньше?" | 4 | Hard (weak signal, high false-positive risk) | P |
| B11 | Illness: hoarseness, nasal (cold) voice, coughing, breathing | "Голос хриплый — простыл? Сделать напоминание про чай/лекарства?" | 5 | Medium for cough/hoarseness; anything diagnostic is research/regulated. | I |
| B12 | Stress / anxiety (tremor, faster rate, higher F0) | Slow down, offer breathing, be supportive. | 4 | Hard | P |
| B13 | Intoxication / slurred speech | Probably ignore; at most avoid risky actions. Creepy if verbalized. | 1 | Hard | P |
| B14 | Speaking while eating/chewing (ComParE "eating condition") or with a mask | Tolerate it; a joke at most. | 1 | Research | I |
| B15 | Sincerity/deception (ComParE 2016) | Do NOT productize. Creepy and unreliable. | — | Research | — |
| B16 | Politeness/register (ты vs вы, formal vs familiar) | Mirror the user's register. Grandma gets "вы" and a softer style. | 3 | Easy (LLM on text) | L |
| B17 | Emphasis and urgency ("СРОЧНО", stressed words) | Prioritize the action. Shorten the answer. | 3 | Medium | P |

#### C. Non-verbal vocalizations (vocal bursts)
| # | Human ability | Joyce use case | Wow | Feasibility | Lang |
|---|---|---|---|---|---|
| C1 | Laughter (and the type: amused, polite, nervous) | Laugh along ("хаха, да!"). Recognize that a joke landed and remember it. | 5 | Medium (gpt-realtime and native models claim laughter cue detection; also a side model) | I (mostly) |
| C2 | Sigh (tired, relieved, sad) | "Тяжёлый день?" — gently, not every time. | 5 | Medium–Hard | I |
| C3 | Crying / sobbing (adult) | Switch to a support mode: slow, warm, no jokes, offer to just be there. Safety escalation if crisis words appear. | 5 | Medium (ComParE 2018 crying; Hume vocal bursts) | I |
| C4 | Gasp / scream / "ой!" | Check in: "Всё в порядке?" Fall-detection context. | 4 | Medium | I |
| C5 | Cough, sneeze | "Будь здоров!" after a sneeze (classic delight moment). Track a cough over days, opt-in. | 5 | Medium (Apple Sound Recognition lists coughing) | I |
| C6 | Yawn | "Похоже, пора спать" / shorten the story. | 4 | Hard (few datasets) | I |
| C7 | Interjections and affect bursts ("ух ты", "фу", "ммм вкусно", "ага") | Mirror the reaction. Treat "ага/угу" as a backchannel, not a turn (see F3). | 3 | Easy–Medium (text) | L |
| C8 | Breathing (heavy, out of breath after sport; ComParE 2020 breathing) | "Ты бегал? Отдышись." | 3 | Hard | I |
| C9 | Humming to oneself | Offer to identify the song or sing along (see D). | 4 | Medium | I |
| C10 | Clicks, tongue-tsks ("ц-ц"), lip smacks | Disapproval/thinking cue; mostly ignore. | 1 | Research | P |

#### D. Singing, humming, whistling, musical perception
| # | Human ability | Joyce use case | Wow | Feasibility | Lang |
|---|---|---|---|---|---|
| D1 | Detect that someone is singing (vs speaking) | Do not treat sung lyrics as commands. Join in with a harmony or backing; applaud at the end. | 5 | Medium (singing/speech classifier, F0 stability) | I |
| D2 | Song recognition from a recording in the room (Shazam-like) | "Что это играет?" → title and artist; "добавь в плейлист". | 4 | Easy (fingerprint APIs: ACRCloud/Shazam-type; vendor not verified here) | I |
| D3 | Recognize a hummed or whistled melody (query-by-humming) | "Joyce, что за песня — ля-ля-ля…" | 5 | Medium. Google Hum to Search proves it; there is no public Google API (gap). | I |
| D4 | Pitch accuracy (in tune or not), intonation feedback | Karaoke or vocal coach: "вторая фраза чуть ниже ноты". Kids' singing games. | 5 | Medium (pitch tracking vs reference melody) | I |
| D5 | Rhythm/tempo/beat (tapping, clapping, metronome) | Clap-along games. "Ты ускоряешься". Detect claps as commands. | 4 | Medium | I |
| D6 | Key/tonality, chords | Joyce sings in the user's key and range: match the user's comfortable pitch before singing together. | 5 | Medium–Hard | I |
| D7 | Lyrics recognition in singing (Russian/Kazakh songs) | Continue the song when the user stops: "а дальше?". Lyric quiz. | 5 | Hard (STT on singing degrades) | L |
| D8 | Instrument recognition, music genre/mood | "Красиво играешь на гитаре!" Learn music preferences. | 3 | Medium | I |
| D9 | Musical emotion | Choose a song for the mood. | 3 | Medium | I |

#### E. Speaker identity and traits
| # | Human ability | Joyce use case | Wow | Feasibility | Lang |
|---|---|---|---|---|---|
| E1 | Who is speaking (known household member) | Personal reminders and calendar per person. "Привет, Аня!" without being told. | 5 | Medium (speaker embeddings, enrollment; cf. Alexa Voice ID, Google Voice Match) | I |
| E2 | Familiar vs stranger / guest | Do not reveal private info to unknown voices. Introduce itself to guests. | 4 | Medium | I |
| E3 | Child vs adult (age group) | Simpler words, safe content, playful tone, no purchases. | 5 | Medium (age-group classifiers; child ASR is weaker) | I/P |
| E4 | Elderly speaker | Slower, louder, clearer speech; patience; more confirmations. | 4 | Medium (ComParE elderly emotion) | I/P |
| E5 | Gender / voice type | Correct Russian grammatical gender ("ты устал" vs "ты устала") — very important in Russian. | 4 | Medium. Use the enrolled profile, not acoustic guessing; ask once. | L (critical for RU) |
| E6 | Accent / dialect / native language (ComParE nativeness, native language, dialects) | Adapt vocabulary for non-native speakers. Do not comment on the accent. | 2 | Medium; ethically risky | L |
| E7 | Number of people present | Group mode (games, quiz). Lower intimacy and privacy. | 3 | Easy–Medium (Soniox diarization speaker count) | I |
| E8 | Voice change over time (the same person sounds unwell or upset) | Compare with the person's baseline. Far more reliable than absolute emotion. | 4 | Hard | I |
| E9 | Recognizing a voice from TV/radio vs a live person | Ignore the TV's "Joyce" mentions and commands spoken by TV characters. | 4 | Hard (replay/media detection; AEC reference helps only for Joyce's own audio) | I |

#### F. Conversational dynamics
| # | Human ability | Joyce use case | Wow | Feasibility | Lang |
|---|---|---|---|---|---|
| F1 | End-of-turn prediction (prosody + syntax + semantics) | Fast response without cutting the user off. A longer wait when the user trails off ("и ещё…"). | 5 (felt as "natural") | Medium (Soniox endpointing; semantic VAD; turn-detector models) | P |
| F2 | Barge-in / interruption handling | User says "стоп/подожди" or starts talking → Joyce stops mid-sentence within ~200 ms and remembers where it stopped. | 5 | Easy–Medium (needs AEC so Joyce does not hear itself) | P |
| F3 | Backchannels ("ага", "угу", "да-да", "ну") | Do NOT stop talking on "угу" during a story. Joyce produces backchannels while the user tells a long story ("угу", "ого"). | 5 | Medium (classify a short burst as backchannel vs interruption) | L |
| F4 | Addressee detection (talking to Joyce vs to a family member or the phone) | Follow-up questions without the wake word. Stay silent when the user turns to a child ("Маша, иди есть"). | 5 | Hard (Apple research; Gemini proactive audio is the closest product) | P |
| F5 | Overlapping speech separation | Two people talk at once → handle both or ask one by one. | 3 | Hard | I |
| F6 | Cocktail-party attention (follow one voice in babble) | Stay locked onto the active user at a party. | 4 | Hard (target-speaker extraction is research to early product) | I |
| F7 | Rhythm/timing of a conversation (gap length, synchrony, speech-rate convergence) | Match the user's pace and pause lengths (entrainment). | 4 | Medium | P |
| F8 | Repair detection ("что?", "а?", "не поняла") | Repeat more slowly and louder, rephrase; do not repeat verbatim. | 3 | Easy | L |
| F9 | Topic/attention cues (the user lost interest: short answers, "ну ладно") | Wrap up the story; offer to stop. | 4 | Medium (LLM on text + prosody) | L |
| F10 | Side remarks / talking to self ("так, где мои ключи…") | Optionally, proactive help ("ключи — ты говорил, они в куртке?"). Risky; opt-in. | 4 | Hard | L |

#### G. Auditory scene (environment sounds, Bregman's "auditory scene analysis")
| # | Human ability | Joyce use case | Wow | Feasibility | Lang |
|---|---|---|---|---|---|
| G1 | Doorbell / knock | "Кто-то звонит в дверь" (useful for people with hearing loss or wearing headphones). Pause the audio. | 4 | Easy–Medium (YAMNet/PANNs-class classifiers; Apple and Nest ship it) | I |
| G2 | Smoke / CO / fire alarm, siren | Loud alert, notify family, ask "всё в порядке?" | 5 (safety) | Medium (shipped by Apple, Nest, Amazon) | I |
| G3 | Baby crying | Notify parents; play a lullaby (Joyce sings!). | 5 | Medium (Apple Sound Recognition) | I |
| G4 | Dog bark, cat meow | "Рекс лает уже 10 минут" while the owner is away. | 3 | Easy–Medium | I |
| G5 | Glass breaking / loud bang / fall thud | Check in. Security mode while away. | 4 | Medium (high false-positive risk) | I |
| G6 | Running water, kettle whistle, microwave/oven beep, washing machine finished | "Чайник вскипел", "вода льётся уже 15 минут — забыли кран?" | 4 | Medium (Apple lists running water and appliances; per-home custom sounds) | I |
| G7 | TV / music / radio playing in the background | Duck own volume. Do not treat TV speech as a command. Offer "сделать тише?" | 3 | Medium | I |
| G8 | Background noise level (quiet / noisy) | Adapt volume and verbosity (cf. Alexa Adaptive Volume). | 3 | Easy | I |
| G9 | Indoor vs outdoor, car, kitchen (acoustic scene classification) | Context: in the car → short answers, no visuals. | 3 | Medium (only if Joyce runs on mobile) | I |
| G10 | Room acoustics (reverb, small vs large room, distance to mic) | Estimate the user's distance and raise volume when the user is far. | 2 | Medium | I |
| G11 | Household activity sounds (cooking, vacuum, shower) | Wait until the vacuum stops before speaking. Recipe step-by-step timing. | 3 | Medium | I |
| G12 | Snoring, sleep sounds (Nest Hub had sleep sensing with cough/snore; not verified here) | Opt-in sleep summary. | 3 | Medium; privacy-heavy | I |
| G13 | Car horn, traffic | Mobile only. | 1 | Easy | I |
| G14 | Silence / absence of expected sounds (no activity all morning) | Elderly check-in: "Доброе утро, как спалось?" if nobody has spoken by 11:00. | 4 | Easy (activity logging); ethically sensitive | I |

#### H. Spatial hearing
| # | Human ability | Joyce use case | Wow | Feasibility | Lang |
|---|---|---|---|---|---|
| H1 | Direction of sound (azimuth) | Turn the LED ring or avatar "gaze" toward the speaker. Choose the beam toward the user. | 3 | Medium (needs a mic array: DOA/beamforming; impossible with a single mic) | I |
| H2 | Distance | Far user → speak louder. Very close whisper → intimate quiet mode. | 3 | Medium–Hard | I |
| H3 | Movement (approaching / leaving) | User walks away mid-answer → pause or summarize: "договорю, когда вернёшься". | 4 | Hard | I |
| H4 | Which room (multi-device arbitration) | The nearest device answers. | 3 | Medium (multi-device energy comparison; Alexa/Google ESP-like) | I |
| H5 | Spatial unmasking (hearing a target among sources in different directions) | Better ASR in noise via beamforming. | 2 | Medium (hardware) | I |

#### I. Attention, habituation, memory for voices and sounds
| # | Human ability | Joyce use case | Wow | Feasibility | Lang |
|---|---|---|---|---|---|
| I1 | Habituation (stop noticing the fridge hum or a ticking clock) | Do not alert on recurring benign sounds. Learn the household baseline. | 3 (felt as an absence of annoyance) | Medium | I |
| I2 | Orienting to novel or salient sounds (own name, a sudden bang) | Wake on its own name in any intonation, including spoken to someone else ("спроси у Джойс"). | 4 | Medium | L |
| I3 | Voice memory (recognize someone after a long time; learn new voices fast) | "Это же Дина! Давно не слышала." Few-shot enrollment of a guest. | 5 | Medium (embeddings plus consent) | I |
| I4 | Echoic memory ("what did you just say?" — replay the last seconds) | "Что он сказал?" about a TV line. Joyce recalls the last 10 s (privacy!). | 4 | Medium (ring buffer); privacy-sensitive | L |
| I5 | Selective attention by instruction ("listen for the kettle") | "Скажи, когда закипит" / "когда малыш проснётся" → a one-off targeted sound watch. | 5 | Medium | I |
| I6 | Personal baseline of each voice (knowing "normal" for a person) | Detect deviation (tired, ill, sad) relative to the baseline instead of absolute. | 4 | Hard | I |
| I7 | Recognize recurring household sounds (own doorbell melody, own car) | Custom sound enrollment, as in Apple iOS 16 custom sounds. | 4 | Medium | I |

### Gaps
- The full ComParE task list for 2014, 2017 and 2021–2025 was not retrieved (the official repository, compare.openaudio.eu, was not fetched).
- Kazakh support and accuracy in Soniox (language ID, diarization, endpointing) was not verified. Neither was quality on RU↔KZ code-switching.
- It was not verified whether Google Cloud TTS (Chirp 3 HD / Gemini-TTS) supports a true whisper style, singing, laughter tokens or emotion prompts in Russian. This is critical for the output side of B2, C1 and D.
- No public API for query-by-humming (Google Hum to Search) was found.
- No quantitative accuracy figures for Russian-language speech emotion recognition, sarcasm, fatigue or child-age detection were found in this pass.

---

## Q2. Which capabilities have existing voice assistants and voice AI productized, and how?

### Takeaway
Products cover a few clusters:
- **Loudness adaptation and whisper:** Amazon.
- **Speaker ID and child voice profiles:** Amazon, Google.
- **Home sound events:** Apple, Google Nest, Amazon. Amazon moved these behind a paywall.
- **Melody recognition:** Google, Shazam.
- **Emotion-adaptive tone, proactive "should I respond" behavior, and semantic turn-taking:** native-audio LLMs (Gemini Live native audio, OpenAI gpt-realtime, Hume EVI, Sesame).

Addressee detection without a wake word is still mostly research (Apple), with Gemini "proactive audio" as the closest shipped API.

### Cited Findings
- **Amazon Alexa Whisper Mode.** When the user whispers, Alexa responds in a whisper as long as the user keeps their voice low. It is enabled under Settings → Voice Responses (coverage from around 2021; older info). — [Engadget](https://www.engadget.com/alexa-adaptive-volume-074739210.html); [Reviewed](https://reviewed.com/smarthome/features/how-activate-adaptive-volume-on-amazon-echo-devices)
- **Amazon Alexa Adaptive Volume (2021, US).**
  - Temporarily raises output volume for one interaction when background noise is detected.
  - Off by default.
  - Apparently works only in the louder direction.
  - A reviewer's test (Echo Dot next to a PC fan) found it inconsistent.

  Sources: [AFTVnews](https://www.aftvnews.com/new-adaptive-volume-feature-for-amazon-alexa-makes-it-speak-louder-when-its-noisy/); [TechRadar](https://www.techradar.com/news/amazon-echo-gets-new-adaptive-volume-controls-so-alexa-can-cut-through-the-noise); [What Hi-Fi](https://www.whathifi.com/news/alexa-can-raise-its-voice-to-be-heard-with-new-adaptive-volume-feature)
- **Amazon Alexa Guard → Emergency Assist.** Amazon retired Guard from late 2023. Smoke/CO alarm detection and glass-break detection moved into the paid Alexa Emergency Assist (about $5.99/month or $59/year). Away Lighting and Home/Away modes stayed free. — [PCWorld](https://www.pcworld.com/article/2082386/alexa-guard-discountinued.html); [CE Pro](https://www.cepro.com/news/amazon-shutters-alexa-guard-service-in-favor-of-new-paid-service/127469); [Cord Cutters News](https://cordcuttersnews.com/once-free-alexa-guard-features-will-now-cost-you-6-a-month/)
- **Amazon Alexa Voice ID for kids.** Parents can teach Alexa a child's voice via Amazon Kids (reported: Kids+ subscription, up to 4 children; older article). — [Amazon help](https://digprjsurvey.amazon.com/csad/help/node/G3UFSS8YLN4P59KD); [GearBrain](https://www.gearbrain.com/teach-alexa-learn-childs-voice-2653595341.html)
- **Google Voice Match.** Supports up to 6 people per home device. Adding a child's voice requires parental permission. Face Match exists only on Nest Hub Max. — [Google Nest Help](https://support.google.com/googlenest/answer/9071584?hl=en-AU)
- **Google Nest sound detection.**
  - Nest cams recognize categories such as dog bark and smoke/CO alarm, and distinguish them from other loud noises.
  - Nest speakers and displays (with Nest Aware, US only) detect smoke/CO alarms and breaking glass.
  - Google's own pages are inconsistent about which devices support what.

  Sources: [How Nest cameras detect sound](https://support.google.com/googlenest/answer/16594324); [Nest Aware store page](https://store.google.com/ca/product/nest_aware?hl=en-GB)
- **Apple Sound Recognition (iOS 14+, on-device accessibility feature).**
  - Core set: fire alarm, siren, smoke alarm, cat, dog, appliances, car horn, doorbell, door knock, running water, baby crying. Some lists also include shouting.
  - Later additions include glass breaking, kettle and coughing.
  - Since iOS 16, users can train custom alarms, appliances and doorbells.
  - Sources are from the iOS 14–16 era; the current list was not verified.

  Sources: [How-To Geek](https://www.howtogeek.com/693029/how-to-use-sound-recognition-on-iphone/); [Gadget Hacks](https://ios.gadgethacks.com/how-to/if-you-wear-headphones-with-your-iphone-you-need-know-about-0385321/); [Cult of Mac](https://www.cultofmac.com/how-to/sound-recognition-on-iphone)
- **Google Hum to Search (2020).** Hum, whistle or sing a melody to find a song; available on Android and iOS. The ML system matches a melody embedding to the original recordings. — [Google Research](https://research.google/blog/the-machine-learning-behind-hum-to-search/?hl=en); [Classic FM](https://www.classicfm.com/music-news/google-hum-to-search-whistle-sing-melody-tool/)
- **Gemini Live API (native audio).**
  - The GA model `gemini-live-2.5-flash-native-audio` lists "affective dialog" and "proactive audio" as key features.
  - Affective dialog: adapts response style and tone to the user's expression. A developer write-up describes it as detecting frustration, confusion or joy and matching tone.
  - Proactive audio: lets the model decide when to respond and in what contexts. The developer write-up says it distinguishes the speaker from background noise and can stay silent.
  - Proactive audio appears as "Preview" in some doc versions.
  - Audio formats: input 16 kHz PCM, output 24 kHz.

  Sources: [Google Cloud Live API docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api); [Google dev forum write-up](https://discuss.google.dev/t/gemini-live-part-1-building-a-low-latency-telephone-voice-agent-with-freeswitch-and-adk-agents-powered-by-gemini-live/332641)
- **OpenAI gpt-realtime / Realtime API.**
  - Press coverage says it picks up non-verbal cues such as laughter, switches languages mid-sentence, and follows tone instructions ("friendly with a French accent", "fast and professional").
  - Semantic VAD (`turn_detection.type = semantic_vad`) estimates whether the user has finished speaking.
  - The OpenAI primary announcement page could not be fetched (DNS error), so the laughter claim rests on secondary coverage.

  Sources: [The Decoder](https://the-decoder.com/openais-real-time-api-picks-up-laughter-accents-and-switches-languages-in-real-time/); [OpenAI VAD docs](https://developers.openai.com/api/docs/guides/realtime-vad)
- **Hume (EVI and Expression Measurement).**
  - Prosody and vocal-burst models score 48 expressions.
  - Third-party coverage says EVI uses prosody (tune, rhythm, timbre) to shape its responses.
  - The standalone Expression Measurement API was sunset in June 2026.

  Sources: [Hume prosody docs](https://dev.hume.ai/docs/expression-measurement/models/prosody); [Contrary Research](https://research.contrary.com/company/hume-ai)
- **Sesame ("Crossing the uncanny valley of conversational voice", Feb 27 2025).**
  - Goal stated as "voice presence": emotional intelligence, conversational dynamics, contextual awareness, consistent personality.
  - Sesame admits gaps in personality, memory, expressivity and appropriateness.
  - Press describes deliberate breaths, chuckles, self-corrections and natural interruptions in the "Maya"/"Miles" demos.

  Sources: [Sesame post (mirror)](https://martinsoler.10minhotel.com/?p=5747); [CO/AI](https://getcoai.com/news/sesames-ai-voice-presence-creates-emotional-bonds-with-human-like-imperfections)
- **Apple device-directed speech detection.** Research on false-trigger mitigation and on wake-word-free follow-ups (2022 and 2023 papers). No shipped wake-word-free Siri mode was found. — [Apple ML](https://machinelearning.apple.com/research/device-directed-speech); [arXiv 2203.15975](https://arxiv.org/pdf/2203.15975); [arXiv 2312.03632](https://arxiv.org/html/2312.03632v1)

### Inferences
- **Coverage map, productized vs not:**
  - Productized: B1 loudness, B2 whisper, E1/E3 speaker and child ID, G1–G6 home sound events, D2/D3 song and hum recognition, F1/F2 turn-taking and barge-in, and B6/C1 emotion and laughter (in native-audio LLMs).
  - Not productized in mainstream assistants: sarcasm, fatigue, illness, sigh/yawn, backchannel generation, pitch-accuracy coaching in an assistant, addressee detection without a wake word (only Gemini proactive audio comes close), baseline-relative mood.

  Joyce can differentiate in that second group.
- **Architecture implication.** Joyce's cascaded stack (Soniox → text LLM → Google TTS) cannot "hear" prosody, laughter, sighs or whispers by default, because text strips them. Two options:
  - (a) Add a parallel audio-analysis lane. It runs on the same mic stream and emits tags such as `[whisper] [laugh] [sigh] [speaker=Аня] [child] [rate=fast] [noise=high] [doorbell]`, which are injected into the LLM prompt. It needs VAD, an F0/energy tracker, an AudioSet-class event model (YAMNet/PANNs/BEATs), speaker embeddings (ECAPA-type) and an optional SER model.
  - (b) Switch or augment the brain with a native-audio model (Gemini Live native audio or gpt-realtime) for the "emotional" lane.

  Option (a) keeps Soniox's Russian accuracy and is language-independent for most acoustic cues.
- **Turn-taking can be improved within the current stack.** Use Soniox endpoint sensitivity (stt-rt-v5) plus an LLM or semantic end-of-turn check.
- **Amazon's paywalling of safety sounds** suggests vendors see sound-event detection as a premium feature. Joyce could offer it as a "home guardian" feature.

### Gaps
- The current (2026) state of Alexa+ and Gemini for Home features (e.g. whether Alexa+ keeps Whisper Mode, and Gemini for Home's conversational follow-up and addressee behaviour) was not verified in this pass.
- The OpenAI primary source could not be fetched, so the exact gpt-realtime claims (laughter detection, benchmarks, release date of Aug 2025 per my memory) are unverified here.
- Hume EVI's current (post-June 2026) status and features were not verified. Neither were Sesame's 2026 product status, Siri's 2026 conversational upgrade, or Shazam API terms.

---

## Q3. Which "wow" cases do users and reviewers love most in empathetic voice AI?

### Takeaway
Reviewers repeatedly highlight four things:
- Human-like imperfections and timing: breaths, chuckles, self-corrections, natural interruptions (Sesame).
- Tone that adapts to the user's emotion: Gemini affective dialog, Hume EVI.
- Laughter recognition: gpt-realtime.
- Whisper-back: Alexa.

The common thread is reciprocity: the assistant reacts to how something was said, not only to what was said.

### Cited Findings
- Sesame demo testers reported forming emotional bonds with the "Maya"/"Miles" voices. The press highlighted intentional imperfections (breaths, chuckles, self-corrections) and natural interruptions in a 28-minute test conversation. These are journalist characterizations, with no independent benchmark. — [CO/AI](https://getcoai.com/news/sesames-ai-voice-presence-creates-emotional-bonds-with-human-like-imperfections)
- Sesame frames the goal as "voice presence": making spoken interactions feel "real, understood, and valued". — [Sesame post (mirror)](https://martinsoler.10minhotel.com/?p=5747)
- Gemini affective dialog is marketed as matching tone to the user's frustration, confusion or joy. — [Google dev forum](https://discuss.google.dev/t/gemini-live-part-1-building-a-low-latency-telephone-voice-agent-with-freeswitch-and-adk-agents-powered-by-gemini-live/332641)
- Laughter recognition and mid-sentence language switching are headline features in press coverage of gpt-realtime. — [The Decoder](https://the-decoder.com/openais-real-time-api-picks-up-laughter-accents-and-switches-languages-in-real-time/)
- Whisper Mode (whispering back) was widely covered as a signature Alexa feature. — [Engadget](https://www.engadget.com/alexa-adaptive-volume-074739210.html)

### Inferences
These are ranked wow candidates for Joyce. The ranking is my judgment, combining cited product reception with Joyce's singing and emotion strengths.
1. **Whisper back + night mode (B2).** Cheap and unforgettable; proven by Alexa.
2. **Laughing along / reacting to laughter (C1),** plus "Будь здоров!" on a sneeze (C5).
3. **Singing together.** Detect that the user is singing (D1), join in their key (D6), continue the lyrics (D7). Offer a gentle pitch coach (D4) and hum-to-identify (D3). Also, baby crying → lullaby (G3). This is unique to Joyce because it sings.
4. **Recognizing who is speaking without being told (E1/I3)**, plus Russian grammatical-gender correctness (E5).
5. **Emotion-aware check-ins**, kept rare and gentle: sigh (C2), crying (C3), sad or tired voice relative to the person's baseline (B6/B10/I6).
6. **Natural turn-taking:** no interrupting while the user thinks (B9/F1), instant stop on barge-in (F2), "угу" backchannels during the user's stories (F3).
7. **Home guardian:** "чайник вскипел", doorbell, smoke alarm, "малыш плачет" (G1–G6), and "скажи, когда закипит" (I5).
8. **Silence when the user talks to someone else (F4)**, plus wake-free follow-ups.
9. **Kid mode by voice (E3).**
10. **Matching pace for elderly or rushed users (B5/E4).**

### Gaps
- No systematic user study or review aggregation ranking these features was found. The ranking above is inference.
- No Russian-language reviews of empathetic voice AI (e.g. Yandex Alice / SberDevices Salute emotional features) were researched. This is relevant competition for RU users and would be worth a separate pass.

---

## Q4. What are the known pitfalls (false positives, privacy, creepiness, cultural differences)?

### Takeaway
- **Emotion inference is scientifically contested and culturally variable.** The EU AI Act bans it in workplaces and schools and calls out low reliability, specificity and generalisability. The home-assistant context is not banned, but the same reliability caveats apply.
- **Sound-event features have false-positive problems.** Adaptive volume was reported as inconsistent, and Nest's own documentation is inconsistent about device support.
- **Always-on listening for non-wake-word sounds raises privacy questions.** Amazon also monetizes these features behind a paywall.

### Cited Findings
- **EU AI Act, Recital 44.** Cites "serious concerns about the scientific basis" of emotion recognition, because emotional expression varies "across cultures and situations, and even within a single individual". It names limited reliability, lack of specificity and limited generalisability, and links these systems to discriminatory outcomes and intrusiveness. The prohibition targets workplaces and education. — [EC AI Act Service Desk, Recital 44](https://ai-act-service-desk.ec.europa.eu/en/ai-act/recital-44); [Technology's Legal Edge](https://www.technologyslegaledge.com/?p=7255)
- **EPIC** argues emotion recognition should be banned in all contexts given its high risk and inefficacy. It notes that "sentiment analysis" and "affective computing" are used interchangeably. — [EPIC comments to Dutch DPA](https://epic.org/documents/epic-comments-to-dutch-dpa-on-emotion-recognition-prohibition-under-eu-ai-act/)
- **Emotional expression is not universal**, so models trained on one expression pattern may misread others. — [UX Collective](https://uxdesign.cc/the-ux-of-emotion-recognition-can-ai-truly-read-feelings-e26f16268e96)
- **Adaptive Volume inconsistency.** In a test with a PC fan, Alexa sometimes answered louder and sometimes at normal volume. — [TechRadar](https://www.techradar.com/news/amazon-echo-gets-new-adaptive-volume-controls-so-alexa-can-cut-through-the-noise)
- **Device-directed speech detection** is harder than wake-word detection. Accidental invocations are a major source of false activations. — [Apple ML research](https://machinelearning.apple.com/research/device-directed-speech); [arXiv 2312.03632](https://arxiv.org/html/2312.03632v1)
- **Amazon paywalled smoke/CO/glass-break detection.** Users who relied on the free Guard lost it unless they subscribe. Vendor changes can remove safety features. — [PCWorld](https://www.pcworld.com/article/2082386/alexa-guard-discountinued.html)
- **Child voice recognition requires explicit parental consent** (Google) or a parent-set-up profile (Amazon). — [Google Nest Help](https://support.google.com/googlenest/answer/9071584?hl=en-AU); [Amazon help](https://digprjsurvey.amazon.com/csad/help/node/G3UFSS8YLN4P59KD)
- **Vendor dependency.** Hume sunset its Expression Measurement API (June 2026), so products built on it had to migrate. — [Hume docs](https://dev.hume.ai/docs/expression-measurement/models/prosody)

### Inferences
- **Never state an emotion as fact.** Use tentative, optional phrasing ("кажется, ты устала — или мне показалось?") and accept correction. Keep check-ins rare: rate-limit them, e.g. at most one unsolicited mood remark per day.
- **Cultural and loudness norms differ.** A loud Russian or Kazakh family conversation is not anger. Calibrate to a per-speaker, per-household baseline (I6) rather than global thresholds. Kazakh and Russian prosody differ, and code-switching speakers may shift prosody between languages.
- **Russian-specific risks:**
  - Wrong grammatical gender for the user ("ты устал" to a woman) is very noticeable. Prefer profile data to acoustic gender guessing, and allow a neutral fallback.
  - The ты/вы register matters.
  - Sarcasm is common in Russian conversational style; literal replies to sarcasm feel robotic.
- **Creepiness tiers:**
  - Low: whisper-back, "будь здоров", laughing along, doorbell.
  - Medium: "ты звучишь уставшим", recognizing guests' voices.
  - High: commenting on illness, intoxication, accent or mood of other household members; recalling overheard side conversations; inferring lies.

  Do not productize deception detection or intoxication detection.
- **Privacy design:**
  - Run sound-event and voice-feature extraction on-device or as streaming features without storing raw audio.
  - Make voice enrollment opt-in per person, with explicit parental consent for kids.
  - Make the echoic ring buffer (I4) short and never uploaded unless invoked.
  - Give a clear "Joyce is listening for sounds" indicator.
- **False-positive pitfalls:**
  - TV speech triggering commands or being treated as a household member (E9/G7).
  - Joyce hearing its own TTS: needs echo cancellation, especially when singing loudly.
  - "угу" misread as barge-in.
  - Kettle mistaken for a smoke alarm. Safety alerts should prefer recall (catch every alarm) but confirm with the user.
  - Child ASR and child voices degrade the accuracy of every model.
  - Singing misread as commands.
- **Health claims.** Cough or hoarseness tracking must not be framed as diagnosis, to avoid medical-device regulatory exposure.

### Gaps
- No quantitative error rates for speech emotion recognition across cultures or languages, especially Russian and Kazakh, were found in this pass.
- No specific regulatory analysis for Kazakhstan or Russia (biometric voice data laws, e.g. Russia's 152-FZ on personal and biometric data) was researched. Voice embeddings may count as biometric data, which needs a separate legal check.
- No documented user-backlash cases specific to empathetic voice assistants (e.g. complaints about unsolicited mood comments) were found.
