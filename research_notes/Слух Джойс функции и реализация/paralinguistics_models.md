# Paralinguistic speech analysis models and APIs for real-time use alongside streaming STT (Soniox): state of the art as of October 2026

Research date: 2026-10-14. Note on method: huggingface.co, arxiv.org (abs pages) and several vendor sites could not be fetched directly from this environment (DNS/proxy block). Facts from Hugging Face model cards and arXiv papers were taken from search snippets, GitHub READMEs (fetched raw), or mirrors. Each one is marked accordingly. Wherever a number could not be checked, it is listed under Gaps rather than stated as fact.

---

## Q1. Which models or APIs to use per task (accuracy, latency, size, license, how to run)

### Takeaway
No single open model covers every task with a commercial license. The most practical commercial-OK stack is:
- **emotion2vec+ (FunASR model license, which allows commercial use with attribution)** for categorical emotion.
- **GigaAM-Emo (MIT)** for Russian emotion.
- **Whisper-AT (BSD) or SenseVoice-Small audio-event tags** for laughter, cough and crying.
- **PESTO / torchcrepe (MIT-style)** for pitch.
- **A DSP rule layer** (energy + F0 + spectral tilt + HNR) for whisper/loud/shout.

audEERING's dimensional emotion and age-gender models, and wav2small, are **non-commercial (CC BY-NC-SA 4.0)**. For commercial use they need a devAIce license.

### Cited Findings

**Task 1: Vocal effort (whisper / soft / normal / loud / shout)**
- There is no large shared benchmark. Most datasets are small or private, and most errors fall between adjacent categories. — [search summary of literature](https://www.isca-archive.org/interspeech_2023/kodali23_interspeech.pdf)
- A June 2026 preprint (Omidi and Hansen, UT Dallas) benchmarks WavLM against wav2vec2 and HuBERT for vocal effort on the AVID corpus. It reports that gradually unfreezing transformer layers improves robustness, and that labeled data remain scarce. — [emergentmind 2606.27543](https://www.emergentmind.com/papers/2606.27543)
- Kodali et al. (Speech Communication, Nov 2025) used 50 speakers and 4 classes (soft, normal, loud, very loud). They tested 15 feature sets and 7 classifiers. The best accuracy was **86.0%**, from a BiLSTM on combined cepstral and spectral features. — [ISI publication page](https://www.isi.edu/results/publications/29223/automatic-classification-of-vocal-intensity-categories-from-amplitude-normalized-speech-signals-by-comparing-acoustic-features-and-classifier-models)
- Kodali et al. at Interspeech 2023 used pre-trained wav2vec2 and Whisper embeddings for vocal-intensity classification. — [ISCA archive](https://www.isca-archive.org/interspeech_2023/kodali23_interspeech.pdf)
- A Leiden thesis (2025) trained a ResNet on the Aalto Vocal Intensity Dataset (4 classes) and reached **71%** accuracy. — [thesis PDF](https://theses.liacs.nl/pdf/2024-2025-HeLLin.pdf)
- "Cries and Whispers" classified whispered, normal and shouted speech in a large video-game speech corpus, raising accuracy from **81.1% to 87%** over plain MFCCs. — [HAL](https://hal.archives-ouvertes.fr/hal-00706261)
- The RISC shout corpus has shout-type labels and crowdsourced intensity ratings. Its authors say shout-type classification and intensity prediction "are still challenging". — [arXiv 2306.04143](https://export.arxiv.org/abs/2306.04143)
- Whisper detection specifically: an LSTM-based whisper detector exists (Alexa research). — [arXiv 1809.07832](https://arxiv.org/pdf/1809.07832)
- **openSMILE (the eGeMAPS extractor) is under the "audEERING Research License Agreement"**, not an OSI license. — [openSMILE LICENSE](https://raw.githubusercontent.com/audeering/opensmile/master/LICENSE)

**Task 2: Categorical emotion**
- **emotion2vec+**: three sizes.
  - seed: 201 h of training data.
  - base: ~90M parameters, 4,788 h.
  - large: ~300M parameters, 42,526 h, drawn from "40k out of 160k hours of speech emotion data".
  - Label set: **0 angry, 1 disgusted, 2 fearful, 3 happy, 4 neutral, 5 other, 6 sad, 7 surprised, 8 unknown**.
  - To run: `pip install -U funasr`, then `AutoModel(model="iic/emotion2vec_plus_large", hub="hf"|"ms")`, then `model.generate(wav, granularity="utterance", extract_embedding=True/False)`. The embedding is 768-d per utterance or T×768 per frame (base).
  - Code is MIT.
  - The repo publishes EmoBox numbers only as a chart image. No new emotion2vec versions appear after the October 2024 news item.
  - Source: [GitHub ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec)
- **emotion2vec+ weight license**: the HF card says `license: other` / `model-license`, pointing to the FunASR repo. — [HF card snippet](https://huggingface.co/emotion2vec/emotion2vec_plus_large/blob/7714a5c669f5711ac128c78f35732e7f9cc976ad/README.md)
  - The FunASR Model Open Source License v1.1 says: "You are free to use, copy, modify, and share [FunASR Software]". The restriction is that you "must attribute the source and author information and retain relevant model names". It has no non-commercial clause. It does include a clause that terminates the license for "malicious smearing" of the software, and it can be revised unilaterally. — [FunASR MODEL_LICENSE](https://raw.githubusercontent.com/modelscope/FunASR/main/MODEL_LICENSE)
  - One aggregator (proginn) lists Apache-2.0 for the model, which conflicts with the card. — [proginn](https://www.proginn.com/w/1547545)
- **SenseVoice-Small**:
  - Emotion labels: HAPPY, SAD, ANGRY, NEUTRAL, FEARFUL, DISGUSTED, SURPRISED (plus `emo_unk`).
  - Audio-event labels: BGM, Speech, Applause, Laughter, Cry, Sneeze, Breath, Cough.
  - The released checkpoint covers zh, yue, en, ja and ko only. **Russian is not supported.**
  - Speed: more than 5x faster than Whisper-Small and 15x faster than Whisper-Large.
  - License: MIT code, FunASR model license for the weights.
  - Deployment: funasr-onnx, libtorch, GGUF/llama.cpp, sherpa-onnx (third party).
  - No native streaming. The third-party "streaming-sensevoice" project offers pseudo-streaming at reduced accuracy.
  - Source: [GitHub FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- **Vox-Profile (USC/JHU, May 2025)** released WavLM-large and Whisper-large-v3 heads trained on MSP-Podcast: categorical "8 Emotions + Other" and arousal/valence/dominance. It also has age/sex, accent, voice quality (ParaSpeechCaps) and fluency (SEP28K+FluencyBank) heads. It is English only. The authors "advise the use of Vox-Profile under the RAIL license". — [GitHub vox-profile-release](https://raw.githubusercontent.com/tiantiaf0627/vox-profile-release/main/README.md); [arXiv 2505.14648](https://arxiv.org/pdf/2505.14648)
- **EmoBox** benchmark: 10 pre-trained models evaluated on **32 emotion datasets in 14 languages** (intra-corpus), plus a cross-corpus set on 4 datasets. — [arXiv 2406.07162](https://arxiv.org/html/2406.07162v1). Whether its Russian datasets (RESD/Dusha) are included could not be confirmed (see Gaps).

**Task 3: Dimensional emotion (arousal / valence / dominance)**
- **audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim**:
  - Wav2Vec2-Large-Robust pruned from 24 to 12 layers, fine-tuned on MSP-Podcast v1.7.
  - Outputs arousal, dominance and valence, each roughly in 0..1.
  - **License CC BY-NC-SA 4.0. "For commercial usage, a license for devAIce must be obtained."**
  - Source: [audeering/w2v2-how-to](https://github.com/audeering/w2v2-how-to); [Zenodo](https://zenodo.org/records/6221127)
- **Wav2Small (audEERING, arXiv 2408.13920)**:
  - **72K parameters**, quantized ONNX of **120 KB** in v4 (the v1 abstract says 60 KB).
  - Distilled from a teacher that sets an MSP-Podcast **valence CCC of 0.676**.
  - Same compute as MobileNetV4-Small (3.12M params) but less RAM.
  - Weights are at huggingface.co/dkounadis/wav2small.
  - Source: [GitHub dkounadis/wav2small](https://raw.githubusercontent.com/dkounadis/wav2small/main/README.md); [arXiv 2408.13920v4](https://arxiv.org/abs/2408.13920v4)
  - An aggregator lists audeering/wav2small as **cc-by-nc-sa-4.0**. — [mixpeek](https://mixpeek.com/model/audeering/wav2small)
- Vox-Profile WavLM-large and Whisper-large-v3 A/V/D heads are an alternative under RAIL (see Task 2). — [vox-profile README](https://raw.githubusercontent.com/tiantiaf0627/vox-profile-release/main/README.md)

**Task 4: Non-verbal vocalizations**
- **VocalSound** dataset (MIT CSAIL):
  - 21,024 clips of laughter, sighs, coughs, throat clearing, sneezes and sniffs, from 3,365 speakers, with age, gender and native-language metadata.
  - **CC BY-SA 4.0.**
  - Leaderboard: EfficientNet-B0 baseline **90.5%**, CLAP **97.95%** (6-class).
  - It has **no cry, yawn or scream class**.
  - Source: [GitHub YuanGongND/vocalsound](https://raw.githubusercontent.com/YuanGongND/vocalsound/main/README.md)
- **NonverbalTTS (deepvk, 2025)**: 17 h, 10 nonverbal types including laughter and coughs, plus 8 emotions. Built from VoxCeleb and Expresso, labeled with BEATs and checked by humans. — [arXiv 2507.13155](https://arxiv.org/pdf/2507.13155)
- **NonVerbalSpeech-38K (2025)**: frame-level nonverbal detector built on w2v-BERT 2.0 with LoRA. Weights and license were not confirmed. — [arXiv 2508.05385](https://arxiv.org/html/2508.05385v1)
- **Whisper-AT** (MIT CSAIL):
  - Frozen Whisper plus a TL-TR tagging head that outputs **527 AudioSet classes**. AudioSet includes laughter, crying/sobbing, cough, sigh, yawn, screaming, whistling, humming and singing.
  - Costs **less than 1% extra compute**, in the same forward pass as ASR.
  - AudioSet mAP: tiny 36.5 (3.8M AT params), base 37.6 (6.6M), small 39.8 (14.6M, or 6.9M at low-dim). large-v1 is about 42 (secondary source).
  - License: **"code and model weights are released under a BSD license … Commercial use is welcome."**
  - `at_time_res` sets the tagging window, for example 10 s.
  - Source: [GitHub YuanGongND/whisper-at](https://raw.githubusercontent.com/YuanGongND/whisper-at/main/README.md)
- **jrgillick/laughter-detection**: PyTorch frame-level laughter segmentation (Interspeech 2021, "Robust Laughter Detection in Noisy Environments"). Trained on Switchboard. Has a threshold parameter and a min_length parameter (default 0.2 s). — [GitHub](https://raw.githubusercontent.com/jrgillick/laughter-detection/master/README.md)
- SenseVoice-Small emits Laughter, Cry, Cough, Sneeze and Breath event tags. — [SenseVoice GitHub](https://github.com/FunAudioLLM/SenseVoice)

**Task 5: Singing, humming, whistling, pitch, song ID**
- **PESTO (Sony CSL)**:
  - About 130k parameters, with a self-supervised, transposition-equivariant objective.
  - Performance is close to CREPE, which has 800x more parameters.
  - New streaming mode (`streaming=True`, cached convolutions) with latency under 10 ms.
  - ONNX export runs at about **0.7 ± 0.03 ms per inference** and is "more than twice as fast as TorchScript".
  - Offline, it is about **12x faster than real time** with a 10 ms step, while CREPE takes about 12 minutes on the same file. The hardware is not specified.
  - Source: [PESTO README](https://raw.githubusercontent.com/SonyCSLParis/pesto/main/README.md); [arXiv 2508.01488](https://arxiv.org/abs/2508.01488)
- **torchcrepe**: MIT license, "tiny" and "full" capacities, Viterbi decoding by default. — [torchcrepe README / LICENSE](https://raw.githubusercontent.com/maxrmorrison/torchcrepe/master/README.md)
- Whisper-AT's AudioSet ontology includes Singing, Humming and Whistling classes. — [Whisper-AT label csv](https://github.com/YuanGongND/whisper-at/blob/main/audioset_label.csv)
- **ACRCloud**:
  - Offers music recognition, humming recognition, cover-song and offline recognition.
  - Pricing is behind a login. There is a 14-day trial.
  - An old 2019 forum post gives **$5.80 per 1,000 requests** beyond the free tier of 100 per day. This price is dated.
  - A third-party add-on recommends recording at least 7.2 s for humming.
  - Sources: [ACRCloud docs](https://docs.acrcloud.com/tutorials/recognize-music); [dbpoweramp forum 2019](https://forum.dbpoweramp.com/forum/other-topics/wishlist/39494-acrcloud-recognition-and-metadata?p=268842); [Librezam privacy doc](https://addons.mozilla.org/dsb/android/addon/librezam/privacy/)
- **AudD**: 10 free requests per day without a key, paid API key after that. Per the same third-party doc it **does not support humming or cover songs**. — [Librezam doc](https://addons.mozilla.org/dsb/android/addon/librezam/privacy/)

**Task 6: Sarcasm / hesitation / confidence / fatigue / stress**
- Vox-Profile ships a **fluency/disfluency** head (WavLM-large and Whisper-large-v3, trained on SEP28K+FluencyBank) that outputs specific disfluency types. This is the closest released model to a "hesitation" detector. — [vox-profile README](https://raw.githubusercontent.com/tiantiaf0627/vox-profile-release/main/README.md)
- The VoxParadox paper (ICML 2026) reports that audio LLMs "listen or read": paralinguistic cues fade in deeper encoder layers and at the encoder–LLM interface, and the LLM often ignores them. This matters for sarcasm, which depends on a mismatch between prosody and content. — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/60946); [alphaxiv 2605.27772](https://dev.alphaxiv.org/audio/2605.27772)

**Task 7: Age / gender / child**
- **audeering/wav2vec2-large-robust-24-ft-age-gender**:
  - 318M parameters, trained on aGender, Common Voice, TIMIT and VoxCeleb2.
  - **CC BY-NC-SA 4.0.**
  - The paper reports age MAE of **7.1–10.8 years** depending on dataset, and **≥91.1% accuracy for 3-class female/male/child**.
  - Sources: [PromptLayer listing](https://www.promptlayer.com/models/wav2vec2-large-robust-24-ft-age-gender); [arXiv 2306.16962](https://arxiv.org/pdf/2306.16962)
- Vox-Profile `wavlm-large-age-sex`: sex (2-class) plus age regression, trained on CommonVoice+TIMIT+VoxCeleb (age-enriched), under RAIL. — [vox-profile README](https://raw.githubusercontent.com/tiantiaf0627/vox-profile-release/main/README.md)

### Inferences
- **Vocal effort**: a DSP rule layer is the fast and cheap default. The features would be calibrated RMS/LUFS relative to a per-speaker baseline, F0 median and range, spectral tilt (alpha ratio or Hammarberg index), HNR, and the voiced fraction (whisper means unvoiced with high HNR loss). These can be computed in plain numpy/librosa with no licensing issue. A small learned head on a shared SSL embedding could improve on it.
  - The best published 4-class accuracies are 71–86%, so even trained models are imperfect on adjacent classes.
  - **Absolute level is unreliable** because of microphone gain, distance and AGC. Speaker-relative normalization is essential. This follows from the "amplitude-normalized" framing in Kodali.
  - Avoid openSMILE in a commercial product unless audEERING grants a license. Parselmouth (Praat bindings) is, as far as I know, GPL-3.0 (not verified here). Using it in closed-source server code may be acceptable but needs legal review.
- **Categorical emotion, commercial and multilingual**: emotion2vec+ base (~90M) is the best default. It is trained on 40k hours, the FunASR license permits commercial use with attribution, and it runs in FunASR. For Russian, add GigaAM-Emo (MIT). See Q2.
- **Non-verbal**: Whisper-AT (BSD, commercial OK) or SenseVoice event tags are the cheapest broad detectors. For better laughter, sigh and cough accuracy, train a small head on VocalSound (CC BY-SA, so the trained weights may carry share-alike obligations to review) plus NonverbalTTS. Cry, yawn and scream coverage has to come from AudioSet-trained taggers (Whisper-AT, BEATs, PANNs).
- **Singing vs speech**: the cheapest signals are AudioSet tags (Singing, Humming, Whistling) from Whisper-AT, plus PESTO pitch statistics. Singing shows long stable voiced notes and quantized pitch. Whistling shows a narrow-band, high F0 with no formants.
  - **In-tune estimation** can be done with PESTO and DSP: compute each sustained note's deviation in cents from the nearest equal-tempered semitone, or from the key estimated over a window. No packaged model for this was found.
- **Song ID**: ACRCloud is the commercial option that also does humming. AudD does not do humming.
- **Sarcasm, confidence, fatigue, stress**: no released, validated, commercial-OK speech-only model was found.
  - Realistic proxies: arousal and valence (stress), speech rate, pause and filled-pause statistics from Soniox word timestamps plus a disfluency head (hesitation), and F0 and energy variability (confidence).
  - Sarcasm needs both text and prosody, which an LLM can judge using the transcript plus prosodic features passed as text.
  - Any "fatigue" or "stress" claim should be presented as low-confidence.

### Gaps
- Measured CPU and GPU latency per second of audio was not found for emotion2vec+, GigaAM-Emo, the audeering models or Whisper-AT in any primary source.
  - Rough expectation, not verified: ~90–100M-param SSL encoders on 1 s of audio take tens of ms on a modern CPU core with ONNX, and a few ms on GPU. ~300M large models take about 3x longer. Benchmark this locally.
- The emotion2vec+ benchmark numbers (EmoBox, SenseVoice comparison) exist only as images in the repo. The exact accuracies were not retrieved.
- The ParaLBench (arXiv 2411.09349) results tables were not retrieved. Only its scope was confirmed: 10 datasets, 13 tasks, 14 acoustic foundation models. — [arXiv 2411.09349](https://arxiv.org/pdf/2411.09349)
- No primary source was found for wTIMIT, CHAINS or Expresso vocal-effort usage, and no trained, released, commercial-OK vocal-effort classifier was found.
- RMVPE details could not be retrieved; the repo README fetch returned an empty page.
- ACRCloud and AudD 2026 price sheets: none found. Pricing must be confirmed with each vendor.
- Hume vocal-burst-style open alternatives covering yawn and scream at vocal-burst granularity: none found beyond the AudioSet taggers.

---

## Q2. Which options work for Russian

### Takeaway
For Russian categorical emotion, **GigaAM-Emo (SberDevices/Salute, MIT, ONNX export, trained on the Russian v1 SSL line)** and the **xbgoose HuBERT/WavLM Dusha finetunes** (5 classes: neutral, angry, positive, sad, other) are the purpose-built options. SenseVoice-Small does **not** support Russian. emotion2vec+ is language-agnostic in principle, but its Russian accuracy is unverified. Qwen3-Omni accepts Russian speech input.

### Cited Findings
- **Dusha** (SberDevices): about 350 h and more than 300k Russian recordings, crowd-annotated. It has an acted subset (more balanced) and a real-life podcast subset (unbalanced). — [arXiv 2212.12266](https://arxiv.org/pdf/2212.12266)
- **xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned**:
  - Base model facebook/hubert-large-ls960-ft. Classes: **neutral, angry, positive, sad, other**.
  - Test **accuracy 0.86, balanced accuracy 0.76, macro-F1 0.81**.
  - Trained with most layers frozen, on half the training set, for 2 epochs. Expects 16 kHz audio.
  - HuBERT-base and WavLM-base variants also exist.
  - Source: [HF model card](https://huggingface.co/xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned); [Azure catalog](https://ai.azure.com/catalog/models/xbgoose-hubert-base-speech-emotion-recognition-russian-dusha-finetuned)
  - The license for these finetunes was not confirmed.
- A 2025 study compared wav2vec2, HuBERT and WavLM on Dusha and RESD. **WavLM was best on Dusha at 0.8782 accuracy**. The split was not confirmed. — [nbpublish article 70472](https://authoren.nbpublish.com/fil/article_70472.html)
- **GigaAM** (salute-developers):
  - Conformer, 220M–600M parameters, **MIT license**, ONNX export via `model.to_onnx` (fp16 recommended on GPU).
  - Lines: v1 (wav2vec2-style, 50k h of Russian pretraining, includes `emo`); v2; v3 (HuBERT-CTC, 700k h pretraining); multilingual (2M h, 70+ languages).
  - "GigaAM-Emo outperforms existing models by 15% Macro F1-Score". The baseline and dataset are not named in the README.
  - API: `gigaam.load_model("emo").get_probs(audio)`.
  - Source: [GitHub salute-developers/GigaAM](https://github.com/salute-developers/GigaAM); [raw README](https://raw.githubusercontent.com/salute-developers/GigaAM/main/README.md)
- **SenseVoice-Small**: the released checkpoint supports zh, yue, en, ja and ko only. — [SenseVoice GitHub](https://github.com/FunAudioLLM/SenseVoice)
- **Qwen3-Omni**: speech input in 19 languages **including Russian**. The open **Qwen3-Omni-30B-A3B-Captioner** is a "detailed, low-hallucination" audio captioner, also available as an Alibaba Cloud Model Studio API. — [Qwen3-Omni README](https://raw.githubusercontent.com/QwenLM/Qwen3-Omni/main/README.md)
- Vox-Profile models cover **English-speaking voices only**. — [arXiv 2505.14648](https://arxiv.org/pdf/2505.14648)
- Behavioral Signals: one directory listing says it supports English only, which conflicts with broader multilingual claims elsewhere. — [Capterra](https://www.capterra.com/p/238079/Oliver-API/)

### Inferences
- **Best Russian categorical SER that is commercially safe: GigaAM-Emo (MIT).** The xbgoose finetunes work but their license is unclear, and they are trained with frozen layers on half of Dusha.
- **Dusha's label set (positive / neutral / sad / angry / other) differs from the 9-class emotion2vec set.** A product needs a unified label mapping, for example happy→positive, with disgusted, fearful and surprised folded into "other" for Russian.
- **Language-independent tasks transfer reasonably to Russian.** These are DSP vocal effort, PESTO pitch, AudioSet-based non-verbal tags (Whisper-AT), and age/gender. Valence models trained on English MSP-Podcast are known to transfer worse across languages than arousal models. This is general field knowledge, not verified here.
- **A GigaAM v3 or multilingual SSL encoder** (MIT, Russian-heavy pretraining) is a strong candidate shared backbone for Russian-first products (see Q3).

### Gaps
- RESD dataset size and license, and any RESD-trained released model: not retrieved.
- GigaAM-Emo's exact label set and its Dusha crowd/podcast metrics: the README does not enumerate them, and arXiv was not reachable.
- Whether EmoBox includes RESD or Dusha, and emotion2vec+ accuracy on Russian: not found.
- License of the xbgoose Dusha finetunes and of the Dusha dataset itself: not confirmed.

---

## Q3. Can one shared encoder with multiple small heads save compute?

### Takeaway
Yes. Whisper-AT shows the pattern officially: frozen ASR encoder plus a tagging head at less than 1% extra compute. emotion2vec, GigaAM and WavLM all expose frame-level and utterance-level embeddings for training light heads. A single ~90–300M SSL encoder run once per 1–3 s window, with linear or MLP heads for emotion, A/V/D, non-verbal, vocal effort, age/gender and singing, is the most compute-efficient design. A separate ~130k-param PESTO handles pitch.

### Cited Findings
- Whisper-AT freezes the Whisper backbone and trains a Time-and-Layer-wise Transformer on layer-wise representations ([num_layer, 25, dim]). It adds less than 1% compute and gives "audio events, in addition to spoken text, in a single forward pass". — [Whisper-AT README](https://raw.githubusercontent.com/YuanGongND/whisper-at/main/README.md)
- emotion2vec's original version claimed SOTA on IEMOCAP "with only linear layers" on top of frozen features. FunASR's `extract_embedding=True` returns utterance (768-d) or frame (T×768) features. — [emotion2vec GitHub](https://github.com/ddlBoJack/emotion2vec)
- Vox-Profile trains separate task heads on WavLM-large and Whisper-large-v3 for age/sex, accent, voice quality, fluency, categorical emotion and A/V/D. This shows one backbone family can serve many traits. — [vox-profile README](https://raw.githubusercontent.com/tiantiaf0627/vox-profile-release/main/README.md)
- audEERING built age and gender as **two custom heads on one wav2vec2 backbone**. — [arXiv 2306.16962](https://arxiv.org/pdf/2306.16962)
- The "Heard but Not Heeded" CMU paper (Sept 2026) found speaking style **is encoded in late encoder layers** (linear probing, leave-one-speaker-out) of Whisper-large-v2, Qwen2-Audio-7B, Qwen2.5-Omni-7B and Chroma-4B on Expresso, but is lost before output. This supports probing encoders directly with light heads instead of asking the LLM. — [arXiv 2609.00727](https://arxiv.org/pdf/2609.00727); [HF papers](https://huggingface.co/papers/2609.00727)
- Wav2Small shows that distilling a large A/V/D teacher into a 72K-param student is feasible. — [wav2small README](https://raw.githubusercontent.com/dkounadis/wav2small/main/README.md)

### Inferences
Soniox runs in the cloud, so the paralinguistic encoder must run on our side.

- **Option A (multilingual, commercial OK)**: emotion2vec+ base as the shared backbone (~90M, FunASR license). Use its native 9-class head, and add trained heads for:
  - A/V/D: distill from audeering or Vox-Profile teacher outputs. Check whether training on outputs of NC-licensed models is allowed, since it likely violates CC BY-NC-SA's spirit; prefer human-labeled MSP-Podcast under its own license.
  - Vocal effort.
  - VocalSound non-verbal classes.
  - Age/gender.
- **Option B (Russian-first)**: GigaAM v3 or multilingual SSL encoder (MIT) with the GigaAM-Emo head plus custom heads.
- **Option C**: Whisper-AT-style tagging heads on a Whisper encoder. This only makes sense if a Whisper encoder already runs locally, which it does not with Soniox.
- **Keep PESTO (pitch) and the DSP features outside the encoder.** They are nearly free and give interpretable, frame-accurate signals.
- **Use layer-weighted pooling across encoder layers** rather than only the last layer. Both Whisper-AT and the CMU probing result show paralinguistic information sits in intermediate and late layers.

### Gaps
- No published multi-head benchmark combining emotion, vocal effort, non-verbal, age and singing on one backbone was found. Accuracy loss compared with dedicated finetunes is unknown.
- License compatibility of distilling from CC BY-NC-SA teachers was not researched legally.

---

## Q4. Unified models and audio LLMs vs specialized models

### Takeaway
Audio LLMs (Qwen3-Omni, Gemini 2.5, GPT-4o-audio) score well on general audio-reasoning benchmarks (MMAU about 62–78). But 2026 studies consistently show they **encode paralinguistic cues and then ignore them**, leaning on lexical content. They are also too heavy (30B MoE) for cheap per-second real-time use. Use them for occasional rich captioning, not as the always-on paralinguistic detector.

### Cited Findings
- **"Heard but Not Heeded: Paralinguistic Information Encoding and Loss in Audio-Language Models"** (arXiv 2609.00727; Koduru, Alharthi, Rita Singh, Bhiksha Raj; CMU LTI):
  - Audio LMs encode speaking style in late encoder layers but largely discard it before output. Predictions end up either content-driven or acoustic-driven.
  - Models studied: Whisper-large-v2, Qwen2-Audio-7B-Instruct, Qwen2.5-Omni-7B and Chroma-4B, on Expresso.
  - Methods: CKA, linear probing, open-ended tone prediction, and a content–prosody leakage metric.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2609.00727); [HF papers](https://huggingface.co/papers/2609.00727)
  - Only the abstract and snippets were read. The numeric probe and output accuracies were not retrieved.
- **VoxParadox (arXiv 2605.27772, ICML 2026)**: paralinguistic cues degrade in deeper encoder layers and at the encoder–LLM interface. Even when the cues are present in audio tokens, the LLM frequently ignores them. — [ICML poster](https://icml.cc/virtual/2026/poster/60946)
- **ParaPairAudioBench (arXiv 2606.24648)**: 5,175 audio pairs across style, rate, emphasis, age and gender. LALM judges **lag humans by 32 percentage points on average**. — [arXiv 2606.24648](https://arxiv.org/pdf/2606.24648)
- **AEQ-Bench (arXiv 2601.10513)**: Qwen3-Omni audio captions were overly positive. Adding them lowered agreement with human judges, and direct audio judging reached only 43% consistency with humans. — [arXiv 2601.10513](https://arxiv.org/pdf/2601.10513)
- **STEB (arXiv 2606.25529)** uses Qwen3-Omni-Captioner, followed by a text LLM, to extract emotion and style fields. This shows captioner-plus-LLM used as an annotation pipeline. — [arXiv 2606.25529](https://arxiv.org/pdf/2606.25529)
- **MMAU-v05.15.25 scores**, from the Qwen3-Omni README table:

  | Model | MMAU |
  |---|---|
  | GPT-4o-Audio | 62.5 |
  | Gemini-2.5-Flash | 71.8 |
  | Gemini-2.5-Pro | 77.4 |
  | Qwen2.5-Omni | 65.5 |
  | Qwen3-Omni-30B-A3B-Instruct | 77.5 |
  | Qwen3-Omni-Thinking | 75.4 |
  | Qwen3-Omni-Flash-Instruct | 77.6 |
  | Qwen3-Omni-Flash-Thinking | 76.5 |

  Column order was inferred from the table header sequence. — [Qwen3-Omni README](https://raw.githubusercontent.com/QwenLM/Qwen3-Omni/main/README.md)
- Qwen3.5-Omni-Plus is reported at MMAU 82.2 vs Gemini 3.1 Pro at 81.1 (press coverage). — [the-decoder](https://the-decoder.com/?p=33841)
- **SenseVoice** is the main "unified small model": ASR, language ID, SER and audio events in one non-autoregressive pass. It does not cover Russian. — [SenseVoice GitHub](https://github.com/FunAudioLLM/SenseVoice)
- **Whisper-AT** gives ASR plus 527-class tags in one pass. — [Whisper-AT README](https://raw.githubusercontent.com/YuanGongND/whisper-at/main/README.md)

### Inferences
- **For real-time per-second signals, specialized SSL heads beat audio LLMs** on cost, latency and reliability. MMAU measures general audio QA, not calibrated emotion or effort classification.
- **An audio LLM is useful asynchronously.** For example, send a 5–10 s clip to Qwen3-Omni-Captioner (API) or Gemini when the cheap detectors flag something interesting ("possible singing", "high arousal"), to get a natural-language description. The caption should be treated as soft evidence because of the positivity bias and content leakage.
- Kimi-Audio, Step-Audio, MiniCPM-o: no paralinguistic-specific benchmark numbers were retrieved in this session (see Gaps).

### Gaps
- EmoBox, ParaLBench, VoxEval and AIR-Bench per-model emotion accuracies for audio LLMs vs emotion2vec: not retrieved, because the tables are in PDFs that could not be fetched.
- Qwen3-Omni's own SER numbers (e.g., MELD) and VocalSound numbers: not found in the README sections parsed.
- Kimi-Audio, Step-Audio 2, MiniCPM-o 2.6/4 and GPT-4o-audio paralinguistic accuracy, and API pricing per audio minute: not researched due to the tool budget.

---

## Q5. Status and pricing of commercial paralinguistic APIs (Oct 2026)

### Takeaway
**Hume's Expression Measurement API is gone.** API use and result downloads ended on 14 June 2026, and the SDK namespace was removed after a 15 June sunset. Third-party coverage says Hume EVI and TTS end on 13 November 2026 (unverified). audEERING devAIce remains, with quote-only pricing. Behavioral Signals is alive with directory-listed per-use pricing. Empath's 2026 status is unclear.

### Cited Findings
- **Hume official docs**: Expression Measurement "is being sunset". May 14, 2026 was the last day to create jobs in the Playground. June 14, 2026 was the last day to use the API and download results. — [Hume docs (via search snippet)](https://dev.hume.ai/docs/expression-measurement/faq)
- **hume-typescript-sdk release notes**: the API was "removed following its sunset on 6/15/2026", and the `hume.expressionMeasurement` namespace is no longer available. — [GitHub releases](https://github.com/HumeAI/hume-typescript-sdk/releases)
- Third-party (forasoft, 2026): Hume closed Expression Measurement on 14 June 2026. Per that article, Hume's 2 October 2026 changelog says **EVI and TTS APIs end on 13 November 2026**. This is unverified against Hume's own changelog. — [forasoft](https://www.forasoft.com/blog/article/real-time-ai-emotion-software)
- A competitor (Imentiv AI) markets itself as a replacement for Hume Expression Measurement. — [Imentiv blog](https://imentiv.ai/blog/replacing-hume-ais-expression-measurement-api-heres-what-imentiv-ai-offers/)
- **Conflicting earlier signal**: one search summary said no sunset notice was found and cited a June 8, 2026 outage. That is superseded by the official docs and SDK changelog above. — [rywalker](https://rywalker.com/research/hume-evi)
- **audEERING devAIce**: SDK, Web API and XR plug-in. Pricing is not public (contact sales). audEERING requires a devAIce license for commercial use of its open CC BY-NC-SA models. — [theaireport](https://www.theaireport.ai/tooldatabase/audeering); [w2v2-how-to](https://github.com/audeering/w2v2-how-to)
- **Behavioral Signals API ("Oliver")**:
  - Emotion outputs: neutral, happy, frustrated, angry, sad, plus positivity and strength.
  - Real-time and batch. Also offers voice-deepfake detection.
  - Directory-listed pricing: **from $0.02 per use**, deepfake tier **$0.25 per use**. Unverified with the vendor.
  - One listing says English only.
  - Sources: [Capterra](https://www.capterra.com/p/238079/Oliver-API/); [Behavioral Signals platform](https://platform.behavioralsignals.com/?p=390)
- **Empath (Tokyo, founded 2017)**: Web Empath API with 4 emotions (joy, anger, calm, sorrow) from intonation, pitch, speed and volume. The sources are old, and no 2026 status was found. — [VentureBeat](https://venturebeat.com/business/empaths-ai-measures-emotion-from-voice); [SpeechTechMag](https://www.speechtechmag.com/Articles/News/Speech-Technology-News/Smartmedical-Launches-Web-Empath-API-to-Identify-Emotion-from-Speech-109964.aspx)
- **Alibaba Cloud Model Studio** offers a Qwen3-Omni-Captioner API, a paid audio-captioning endpoint. — [Qwen3-Omni README](https://raw.githubusercontent.com/QwenLM/Qwen3-Omni/main/README.md)

### Inferences
- Given Hume's exit and the opaque or English-centric pricing of the remaining vendors, **self-hosting open models is the lower-risk path** for a Russian-language real-time product.
- audEERING devAIce is the one vendor whose license would also unlock its strong open A/V/D and age-gender checkpoints for commercial use. That is worth a quote request if dimensional emotion and age matter.

### Gaps
- Official 2026 price sheets for devAIce, Behavioral Signals, Empath, ACRCloud and AudD: none found.
- Whether Empath still operates in 2026: unknown.
- Russian support for devAIce and Behavioral Signals: not confirmed.
- Hume EVI/TTS end date (13 Nov 2026): only a secondary source.
