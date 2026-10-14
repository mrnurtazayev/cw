# Joyce's "ear": architecture, latency and cost trade-offs (cascade sidecar vs native-audio LLM vs hybrid)

Research date: 2026-10-14. Method note: direct page fetches were blocked in this environment (DNS/proxy refusal for soniox.com, ai.google.dev, openai.com, arxiv.org), so all facts below come from web-search result extracts of the cited pages. Where the extract came from an aggregator rather than the vendor, it is flagged. All prices are list/pay-as-you-go and should be re-checked on the vendor page before budgeting. Prices marked "(dated: …)" carry the snapshot date the source reported.

## D. What Soniox already gives us "for free" (baseline)

### Takeaway
Soniox `stt-rt-v5` costs about $0.12 per audio hour (June 2026) and already includes word-level timestamps, a speaker label per token, language ID, endpoint detection and real-time translation. It does not output emotion, paralinguistic or sound-event labels. That gap is what the "ear" has to fill.

### Cited Findings
- Real-time model `stt-rt-v5` is listed at **$0.12/hour** (~$0.002/min); async `stt-async-v5` at **$0.10/hour**; the real-time price is marked "updated June 2026". 1,000 h of real-time audio ≈ $120/month. Diarization, language identification and timestamps are included in the rate. (Dated: June 2026, from Soniox's own comparison page) — [Soniox: Compare STT API pricing](https://soniox.com/compare/stt/pricing)
- By default Soniox does not store real-time audio or transcripts and does not train on customer data — [Soniox: Compare STT API pricing](https://soniox.com/compare/stt/pricing)
- WebSocket endpoint `wss://stt-rt.soniox.com/transcribe-websocket`, model id `stt-rt-v5`. Responses carry a `tokens` array; each token has `text`, optional `start_ms` / `end_ms` (omitted for translation-status tokens), `confidence`, `is_final`, and `speaker` (when diarization is on) — [Soniox WebSocket API](https://soniox.com/docs/api-reference/stt/websocket-api); [Soniox Node SDK types](https://soniox.com/docs/sdk/node-SDK/reference/types); [Real-time transcription](https://soniox.com/docs/stt/rt/real-time-transcription)
- Non-final tokens (`is_final:false`) are provisional and can be revised. Final tokens never change — [Soniox Real-time transcription](https://soniox.com/docs/stt/rt/real-time-transcription)
- `enable_endpoint_detection` lets the model detect when the speaker stops and finalize tokens immediately. A special `<end>` token marks the end of speech — [Soniox Endpoint detection](https://soniox.com/docs/stt/rt/endpoint-detection); [Python SDK real-time](https://soniox.com/docs/sdk/python-SDK/stt/realtime-transcription)
- Manual finalization: the client sends `{"type":"finalize"}` and the server emits a `<fin>` marker token when finalization is complete — [Soniox Manual finalization](https://soniox.com/docs/stt/rt/manual-finalization)
- A third-party proxy shows a v5 config with diarization + endpoint detection and a **500 ms max endpoint delay** parameter (unverified against Soniox docs) — [SLNG docs: Soniox Real-Time v5](https://docs.slng.ai/api-reference/stt/soniox-speech-ai-real-time-v5/speech-ai-real-time-v5-ws)
- Real-time speech-to-text translation exists as a documented feature — [Soniox RT translation](https://soniox.com/docs/translation/stt-translation/rt-translation)
- **No STT emotion or audio-event output was found.** Soniox's emotion and tone "audio tags" (`[happy]`, `[sad]`, `[laughs]`, `[sobs]`…) belong to its **TTS** product: they go into text sent for synthesis and are not produced by transcription. Soniox TTS v2 was announced 2026-08-11 at **$0.70 per generated hour** — [Soniox TTS emotion & tone docs](https://soniox.com/docs/tts/concepts/emotion-and-tone); [Soniox TTS v2 blog](https://soniox.com/blog/soniox-tts-v2)

### Inferences
- Soniox's per-token `start_ms`/`end_ms` plus `speaker` are enough to segment the raw audio ring buffer into utterances and speaker turns. The sidecar models can then run on exact audio slices without a second VAD or diarizer in the cloud path. A tiny local VAD is still useful for barge-in and for gating models.
- `<end>` (endpoint) is the natural trigger for per-utterance paralinguistic inference. Its timing sets how long the sidecar has to finish before the LLM call starts.
- Soniox's speaker label is only a per-session cluster ID ("1", "2"…). It does not say who the speaker is across sessions. A speaker-embedding model plus a stored voiceprint is still needed to recognise "this is Arman again".

### Gaps
- No full token JSON sample was retrieved verbatim (pages could not be fetched). Field names above come from search extracts of the docs.
- No info found on a "v4 vs v5" feature diff, or on whether language-ID is per token (a `language` field) in v5. Check the docs page.
- Pricing for translation and whether it is billed separately was not confirmed.

## A. Cascade "sidecar": Soniox for text + parallel small models

### Takeaway
Small specialist models (Silero VAD, a speaker-embedding model such as WeSpeaker ResNet34 or ECAPA, emotion2vec+, CED-Tiny/Mini audio tagging, plus DSP pitch/energy/rate) are each tiny (5–100M params). They can run in ONNX Runtime on a CPU, on-device or on a cheap cloud vCPU. Cost per conversation hour stays well under $0.10, and added latency on the response path is ~0 if they run in parallel and are finished by the time Soniox emits `<end>`. Exact ms/RTF numbers for this model set could not be found and must be benchmarked.

### Cited Findings
**Model sizes and speeds**
- Silero VAD: a 30 ms+ chunk takes **<1 ms on one CPU thread** (README claim, also with ONNX). A contributor notes they never measured it precisely and that two processes saturated their CPU — [silero-vad on PyPI](https://pypi.org/project/silero-vad); [Lhotse PR discussion](https://github.com/lhotse-speech/lhotse/pull/1160)
- CED audio tagging (AudioSet, 527 classes): **CED-Tiny 5.5M params, mAP 48.1** (AS-2M); CED-Mini 9.6M, 49.0; CED-Small 22M, 49.6; CED-Base 86M, 50.0. CED-Tiny "runs on a common CPU as fast as a comparable MobileNetV3". It uses 64-dim mel and non-overlapping 16×16 patches (248 patches per 10 s clip vs 1212 for AST) — [CED model card](https://huggingface.co/mispeech/ced-mini); [CED GitHub](https://github.com/richermans/ced); [CED paper arXiv 2308.11957](https://arxiv.org/pdf/2308.11957)
- emotion2vec+ (`iic/emotion2vec_plus_base`): 9 classes (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown). The base model was fine-tuned on 4,788 h. The large variant is ~300M params (third-party figure). No base param count was found — [emotion2vec GitHub](https://github.com/ddlBoJack/emotion2vec); [PromptLayer emotion2vec+ large](https://www.promptlayer.com/models/emotion2vecpluslarge)
- WeSpeaker VoxCeleb ResNet34-LM speaker embedding: `speaker-embedding.onnx` is **26.5 MB**, ~6.6M params, 256-dim embedding (the param count comes from a secondary source) — [HF hbredin/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM/tree/main); [WeSpeaker ResNet34 MLX port](https://ai.gitcode.com/Delicate02/WeSpeaker-ResNet34-LM-MLX-Ascend)
- ECAPA-TDNN's multi-branch topology "does not obtain a very satisfactory inference speed on GPU". It is better suited to CPU or batch than to low-latency GPU use — [TMS paper arXiv 2203.09098](https://arxiv.org/pdf/2203.09098)
- An existing open project (emotionedge-cpp) runs ASR → emotion2vec+ (valence/arousal/dominance) → translation → TTS entirely in ONNX Runtime on CPU or GPU with no Python, targeting <800 ms end to end. It reports CUDA vs CPU emotion outputs agree within 0.002, and an INT8 DistilRoBERTa at 4 ms vs 10 ms FP32 on CPU — [emotionedge-cpp](https://github.com/ardhendudebnath/emotionedge-cpp)
- sherpa-onnx (k2-fsa) ships CPU aarch64 builds and runs on Raspberry Pi. A Pi 5 guide uses a 3D-Speaker embedding model with 4 threads for two-speaker diarization, and the Pi ran at ~69 °C. No RTF was given — [DietPi forum Pi 5 guide](https://dietpi.com/forum/t/use-a-raspberry-pi5-16-gb-ram-512-gb-nvme-to-transcribe-phone-calls-etc-locally/25489); [sherpa_onnx package](https://pub-web-s1.flutter-io.cn/packages/sherpa_onnx)
- No published CPU latency or RTF benchmark was found for emotion2vec + ECAPA in ONNX int8. The recommendation is to export, quantize and time on the target CPU (RTF = inference time / audio duration) — [DEV: ECAPA-TDNN + ONNX Runtime](https://dev.to/kiarina/grouping-utterances-by-speaker-with-ecapa-tdnn-and-onnx-runtime-411b)

**Where to run: compute prices (all dated)**
- AWS `c7g.large` (2 vCPU Graviton3) us-east-1: **$0.0725/h**; `c8g.large` **$0.0798/h**; c8g 1-yr savings plan $0.0526/h, 3-yr $0.0350/h (aggregator snapshot, Jul 16 2026). Sydney c7g.large is $0.0944/h — [Holori c7g.large](https://calculator.holori.com/aws/ec2/c7g.large/us-east-1); [Holori c8g.large](https://calculator.holori.com/aws/ec2/c8g.large/us-east-1); [DoiT c7g.large ap-southeast-2](https://www.doit.com/compute/spot/ap-southeast-2/c7g.large)
- Hetzner repriced on **2026-06-15**: CCX13 (dedicated vCPU) went from €15.99 to **€42.99/month** (+169%, DE/FI) and CAX21 (Arm) rose +31.2%. Cost-optimized CX/CAX lines were shown as "currently not available" at times. Hetzner Cloud is no longer the obvious bargain it was in 2025 — [privatedevops](https://privatedevops.com/news/hetzner-june-2026-cloud-price-increase-what-to-do); [Capital & Compute](https://capitalandcompute.net/blog/hetzner-price-increase-2026/); [findstack](https://findstack.com/resources/hetzner-price-increase-2026); [wz-it](https://wz-it.com/en/blog/hetzner-price-increase-june-2026-cpx-ccx-alternatives/)
- Modal serverless GPU base per-second rates: T4 $0.000164 (~$0.59/h), L4 $0.000222 (~$0.80/h), A10 $0.000306 (~$1.10/h), L40S $0.000542 (~$1.95/h). Region pinning adds 1.5–1.75× and non-preemptible adds 3×. CPU is $0.0000131/core-s (~$0.047/core-h) and memory $0.00000222/GiB-s (aggregators) — [Spheron: Modal pricing 2026](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/); [Beam: Modal pricing explained](https://www.beam.cloud/blog/modal-pricing-explained)
- RunPod Serverless Flex bills per second from worker start to stop. The 16 GB class (A4000/A4500/RTX 4000) costs $0.00016/s (~$0.58/h) — [RunPod serverless pricing docs](https://docs.runpod.io/serverless/pricing)

**Framework structure**
- Pipecat: `ParallelPipeline` creates branches that **each receive all upstream frames**, and processors pass frames on instead of consuming them. Audio can therefore fan out to the STT branch and to custom paralinguistic processors at once. There is no built-in emotion processor, so it is custom code — [Pipecat pipeline docs](https://docs.pipecat.ai/guides/learn/pipeline); [Pipecat STT docs](https://docs.pipecat.ai/pipecat/learn/speech-to-text.md)
- LiveKit Agents: you subclass `Agent` and override pipeline nodes such as `stt_node()`. The docs show wrapping `Agent.default.stt_node` to run custom logic before and after default STT, which is where an audio tee to sidecar models can go. Realtime-model agents expose a separate `realtime_audio_output_node` — [LiveKit: Pipeline nodes & hooks](https://docs.livekit.io/agents/logic/nodes.md); [LiveKit agents-js Agent reference](https://docs.livekit.io/reference/agents-js/classes/agents.voice.Agent.html)
- TEN Framework (Agora): a graph runtime of "extensions" with command, data, audio-frame and video-frame flows declared in `property.json`. A voice agent is RTC → STT → LLM → TTS nodes, and providers are swapped by changing the node's `addon` field. Sidecar extensions can subscribe to the same audio-frame flow — [TEN Framework overview](https://docs.agora.io/en/ai/ten-agent/framework-overview); [TEN core concepts](https://docs.agora.io/en/ai/ten-agent/core-concepts)

### Inferences
- **Compute envelope (estimate, must benchmark).** VAD costs <1 ms per 30 ms (≈3% of one thread). CED-Tiny on a 1–2 s window every ~1 s, a ResNet34 speaker embedding once per utterance, emotion2vec+ base once per utterance and DSP features together plausibly need roughly 0.05–0.2 CPU-core-seconds per audio-second on a modern x86 or Graviton core with int8/FP16 ONNX. On that assumption one `c7g.large` ($0.0725/h) serves ~10–40 concurrent streams, which works out to **~$0.002–$0.01 per conversation-hour**. Even the pessimistic case of one dedicated `c7g.large` per stream is only **~$0.07–0.08/h**. Modal CPU at ~$0.047/core-h × 0.2 cores ≈ **$0.01/h** plus memory.
- **Serverless GPU is a poor fit for the sidecar.** These models are CPU-friendly. A GPU only pays off with batching across many simultaneous streams, and per-utterance invocations would add cold-start and network latency. An always-on L4 on Modal is ~$0.80/h base (~$2.40/h non-preemptible), which is only worth it at roughly 50+ concurrent streams.
- **On-device (phone, Raspberry Pi 5, Jetson).** Marginal cost is $0 and audio never leaves the device, which is the best privacy option. A Pi 5 is likely to handle VAD + CED-Tiny + ResNet34 speaker embedding in real time. emotion2vec+ base on a Pi is uncertain because there is no sourced RTF (treat it as per-utterance only, possibly 100s of ms). A Jetson Orin or a recent phone NPU should handle all four.
- **Shared-encoder multi-head design (Phase 3 optimisation).** One mid-size SSL encoder (e.g., a WavLM- or emotion2vec-style backbone) can run once per window with light heads for emotion, arousal/valence, speaker embedding and events. This cuts compute ~2–3× against separate models, but it means fine-tuning heads and losing off-the-shelf checkpoints. It is worth it only once the per-model version proves useful.
- **Batching.** On a cloud CPU, batch utterances from multiple sessions per inference call (micro-batches of 10–50 ms) to use int8 GEMM efficiently. On device there is no batching, so use 1–2 intra-op threads to avoid stealing cycles from the audio path.
- **Feeding results to the LLM.** Prepend a compact, structured tag block to the user turn, e.g. `<ear speaker="Arman" spk_conf=0.91 emotion="sad" emo_conf=0.62 arousal=0.3 valence=-0.4 pitch="low,flat" rate="slow" events=["sigh","tv_background"] lang="ru"/>`. Include only fields above a confidence threshold, and tell the system prompt these are noisy cues. JSON costs a little more in tokens than inline XML-style tags (~30–60 tokens/turn either way), which is negligible for a fast LLM.

### Gaps
- No sourced ms or RTF figures for emotion2vec+ base, WeSpeaker ResNet34, ECAPA or CED-Tiny on x86, Graviton, Pi 5, Jetson or phone NPUs. All throughput numbers above are estimates that need a benchmark.
- No official parameter count for emotion2vec+ base was found.
- No sourced "shared-encoder multi-head" production example for voice agents.
- RunPod Flex prices for L4/A10/T4-class GPUs were not retrieved.

## B. End-to-end native-audio LLMs as the ear

### Takeaway
Gemini Live (3.1 Flash Live Preview; 2.5 Flash native audio) charges ~$3/1M audio-input tokens (≈$0.005/min ≈ **$0.30 per listened hour**) and offers affective dialog and proactive audio. OpenAI's gpt-realtime charges $32/1M audio-input tokens. Both are an order of magnitude more expensive per hour than Soniox + sidecar, and they replace rather than complement Joyce's LLM and TTS choices. Research (CMU, Sept 2026) shows audio LLMs **encode** speaking style well internally but **lose most of it at the output**. Qwen2.5-Omni gets 53.7% output tone accuracy against ~84% for a linear probe on its own encoder, so specialised probes or classifiers still beat asking an audio LLM "how did they sound?" on controlled style labels.

### Cited Findings
**Google Gemini Live / native audio**
- Official Gemini API pricing page (as surfaced in search): **Gemini 3.8 Live, 3.8 Live Extended Thinking and Gemini 3.1 Flash Live Preview** share a paid-tier rate card. Input is **$0.75/1M text**, **$3.00/1M audio (~$0.005/min)** and $1.00/1M image/video. Output is **$4.50/1M text** and **$12.00/1M audio (~$0.018/min)**, with thinking tokens billed as output — [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing); third-party confirmation of 3.1 Flash Live audio-in at $3 — [The Rundown: Gemini 3.1 Flash Live](https://www.therundown.ai/tools/gemini-3-1-flash-live)
- Gemini 3.8 Live was reportedly announced 2026-09-15. One third-party article says Google's docs gave no separate 3.8 Live price at that point, which conflicts with the rate card above. Treat the 3.8 Live price as unconfirmed — [aireiter (FR)](https://aireiter.com/fr/blog/gemini-3-8-live-api-pricing); [aireiter (PT)](https://aireiter.com/pt/blog/gemini-3-8-live-api-pricing)
- The pricing page also lists `gemini-2.5-flash-native-audio-preview-12-2025`, described as tuned for better pacing, naturalness and mood. Its exact row price was not confirmed. Aggregators conflict ($3/$12, or $3/$2) — [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing); [cloudprice: 2.5 Flash Live Native Audio](https://cloudprice.net/models/google_vertex_ai/gemini-live-2.5-flash-native-audio)
- Audio tokenisation for Live is stated as **25 tokens/second (~1,500 tokens/min)** by a blog citing Google's model pages — [tokenkarma: Gemini Live API pricing](https://tokenkarma.app/blog/gemini-live-api-pricing-voice-agents-2026/)
- **Proactive audio** (the model may choose not to answer irrelevant input; `proactivity.proactive_audio`) and **affective dialog** (`enable_affective_dialog`, which adapts response style to the user's emotion and tone) both require the `v1alpha` API version. The capabilities guide's examples use `gemini-3.1-flash-live-preview`, and there is a migration guide from Gemini 2.5 Flash Live — [Live API capabilities guide](https://ai.google.dev/gemini-api/docs/live-guide); [gemini_ex Live API guide](https://gemini-ex.hexdocs.pm/live_api.html)
- With proactive audio, **input tokens are charged while the model is listening**, and output tokens only when it responds — [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
- Developers reported 1007 "Unknown name" errors when sending these fields to older native-audio models (user reports) — [Google AI forum thread](https://discuss.ai.google.dev/t/new-live-api-features-dont-work-through-the-api-proactive-audio-and-affective-dialong/84326)
- A separate forum thread discusses Live API billing of audio tokens per second including silent audio — [Google AI forum: Live API pricing & silent audio](https://discuss.ai.google.dev/t/live-api-pricing-audio-tokens-second-silent-audio/92653)

**OpenAI gpt-realtime**
- The official model page for **gpt-realtime-1.5** lists **audio $32/1M input, $64/1M output**, and **text $4/1M input, $16/1M output** — [OpenAI gpt-realtime-1.5 model page](https://developers.openai.com/api/docs/models/gpt-realtime-1.5)
- Aggregators list `gpt-realtime-mini` at $0.60/$2.40 per 1M (most likely text rates), and `gpt-realtime-2.1` at $4 in and $24 out (text). Audio rates for the mini and 2.x models were not confirmed — [Maxim/Bifrost gpt-realtime-mini](https://www.getmaxim.ai/bifrost/llm-cost-calculator/provider/openai/model/gpt-realtime-mini.md); [Maxim/Bifrost gpt-realtime-2.1 (Azure)](https://www.getmaxim.ai/bifrost/llm-cost-calculator/provider/azure/model/gpt-realtime-2.1)

**Open models (self-host)**
- Qwen3-Omni technical report: **theoretical cold-start first-packet latency of 234 ms (audio) and 547 ms (audio+video)**, using vLLM + torch.compile + CUDA Graphs. Latency rises with concurrency — [Qwen3-Omni Technical Report arXiv 2509.17765](https://arxiv.org/html/2509.17765v1)
- After vLLM-Omni optimisations, mean audio time-to-first-packet at **concurrency 64** fell from ~5,884 ms to **~632 ms** (Jul 2026) — [vLLM blog: serving Qwen3-Omni in vLLM-Omni](https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization)
- A Qwen3.5-Omni technical report exists (arXiv 2604.15804). For the 30B MoE backbone, a deployment guide estimates **~60 GB FP16** (does not fit an L40S 48 GB) and **~30 GB at FP8**, and recommends an 80 GB GPU for production — [Qwen3.5-Omni report](https://arxiv.org/html/2604.15804v2); [Spheron: deploy Qwen3.5-Omni](https://www.spheron.network/blog/deploy-qwen3-5-omni-gpu-cloud/)
- Step-Audio 2 introduced **StepEval-Audio-Paralinguistic**: 550 clips across 11 dimensions (gender, age, timbre, emotion, pitch, rhythm, speed, style, vocal activity, sound event, scenario, vocal sounds), mostly Chinese podcast audio plus AudioSet, CochlScene and VocalSound, judged by ASR + LLM. The authors claim SOTA over GPT-4o Audio, Kimi-Audio and Qwen-Omni. Per-task numbers were not retrieved — [Step-Audio 2 Technical Report arXiv 2507.16632](https://arxiv.org/pdf/2507.16632)

**Quality of paralinguistic understanding: the CMU paper**
- "**Heard but Not Heeded: Paralinguistic Information Encoding and Loss in Audio-Language Models**", arXiv **2609.00727** (Sept 2026), by Koduru, Alharthi, Singh and Raj (CMU). It studies Whisper-large-v2, Qwen2-Audio-7B-Instruct, Qwen2.5-Omni-7B and Chroma-4B on Expresso (same sentences in different speaking styles), using CKA, leave-one-speaker-out linear probes, open-ended tone prediction and a content–prosody leakage metric. Core finding: **style is encoded in late encoder layers but largely discarded before output** — [arXiv 2609.00727](https://arxiv.org/abs/2609.00727); [HF papers page](https://huggingface.co/papers/2609.00727)
- Reported numbers (from secondary summaries): linear-probe style accuracy peaks at **~84% (Qwen2.5-Omni, encoder layer ~22)**, ~82% (Qwen2-Audio, ~layer 20) and ~85% (Chroma), against a 14.3% chance level. **Output-level open-ended tone accuracy is 53.7% for Qwen2.5-Omni, 19.7% for Qwen2-Audio and 22.6% for Chroma.** On content–prosody leakage, Qwen2-Audio scores 0.977, Chroma 0.844 and Qwen2.5-Omni 0.272 (lower means more acoustics-driven). The authors attribute Qwen2.5-Omni's better result to its speech-generation training objective. Only relative comparisons of the leakage metric are meaningful — [pith.science summary](https://pith.science/paper/2609.00727); [alphaXiv](https://www.alphaxiv.org/abs/2609.00727)
- Related 2026 work on the same failure mode: "Do Audio LLMs Listen or Read? … VoxParadox" (ICML 2026), "Do Audio Language Models Use Paralinguistic Evidence? Counterfactual Audits" (arXiv 2608.06718) and "Resurfacing Paralinguistic Awareness in Large Audio Language Models" (arXiv 2603.11947) — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/60946); [arXiv 2608.06718](https://arxiv.org/pdf/2608.06718); [arXiv 2603.11947](https://arxiv.org/pdf/2603.11947)

### Inferences
- **Gemini Live as "ear only" (listening, text-out):** 25 tok/s × 3,600 s = 90k tokens/h × $3/1M ≈ **$0.27/h** of audio streamed, plus a few cents of text output. Proactive audio bills listening time, so silence in an always-on session costs the same. If Joyce also used it as the voice (12 $/1M audio out ≈ $0.018/min), and Joyce speaks ~40% of the time: ≈ $0.27 + 0.4×60×$0.018 ≈ **$0.70/h** before any per-turn context re-billing. Context accumulation across turns in long sessions may raise this. Check Google's docs; it is not verified here.
- **gpt-realtime as ear:** at the commonly cited OpenAI rate of ~10 audio tokens/s for input (not verified in this session), 36k tokens/h × $32/1M ≈ **$1.15/h** for input alone. Output at $64/1M is far more. This is ~10× Gemini Live and ~100× the sidecar.
- **Self-hosted Qwen3/3.5-Omni:** one 80 GB-class GPU (~$2–4/h on serverless or cloud) serving tens of concurrent streams works out to roughly **$0.05–0.30 per stream-hour** at good utilisation. At low concurrency (a single user) it is $2+/h, because you pay for an idle GPU. The engineering burden is high (multi-stage thinker/talker serving).
- **Latency:** using a native-audio model as the whole brain removes the STT→LLM hop (theoretical ~234 ms first packet for Qwen3-Omni). It forces Joyce onto that model's voice, though, and loses the chosen Google TTS (Gemini TTS / singing) and the "fast LLM brain". Used as a **parallel** ear alongside the cascade, it adds no response latency, but its result arrives ~0.5–2 s after the utterance ends (estimate). That is usable for the next turn, not the current one.
- **Quality:** the CMU numbers suggest that for controlled style categories (whisper, laugh, sad, happy…), a probe or classifier on encoder features (~84%) is more reliable than an open-ended LLM description (≤54%). Audio LLMs are better at **holistic, open-ended** descriptions ("tired but trying to sound cheerful", judging singing pitch and expressiveness). They are a "second opinion", not the primary detector.

### Gaps
- Official Gemini Live per-turn context billing rules and session-length caps were not verified.
- No public, comparable benchmark of Gemini Live or gpt-realtime emotion recognition accuracy vs emotion2vec+ was found.
- Kimi-Audio, MiniCPM-o and Moshi latency and VRAM numbers were not retrieved in this pass.
- No official audio rates were confirmed for gpt-realtime-mini / gpt-realtime-2.x.

## C. Hybrid: cheap always-on cascade + on-demand audio-LLM

### Takeaway
Run the sidecar (A) on every utterance. Send only flagged audio slices (low-confidence emotion, laughter/crying events, singing, the user asking "how did I sound?") to a standard (non-Live) Gemini Flash or Flash-Lite audio call, or to a self-hosted Omni model. Escalating ~5% of audio adds well under $0.02/h, and when done asynchronously it adds no latency to normal replies.

### Cited Findings
- Gemini 3.1 Flash-Lite is listed by aggregators at $0.25/1M input and $1.50/1M output (text). An older Flash-Lite line lists **$0.30/1M audio input**. No audio-specific 3.1 Flash-Lite rate was confirmed — [Morph: Gemini API pricing 2026](https://www.morphllm.com/gemini-api-pricing); [cloudprice: Gemini Flash Lite](https://cloudprice.net/models/google-gemini-flash-lite)
- Gemini 3.1 Flash Live: $3/1M audio-in (for comparison: Live is ~10× the cheapest batch audio rates) — [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- Latency budgets in 2026 voice-agent guides: ~800 ms total = VAD/capture 50 + STT 150 + LLM TTFT 400 + TTS first chunk 150 + network 50. Tighter sub-500 ms P95 budgets allot STT first partial 100–150, LLM TTFT 200–300 and TTS 80–150 ms. Endpointing is a hidden 100–300 ms. A single 300 ms tool call pushes time-to-first-audio to ~1,100 ms — [FutureAGI: sub-500ms voice AI](https://futureagi.com/blog/sub-500ms-voice-ai-guide-2026/); [FutureAGI: measuring latency](https://futureagi.com/blog/how-to-measure-voice-ai-latency-2026/); [Vapi playbook](https://vapi.ai/playbook/build); [Morph: voice agent latency](https://www.morphllm.com/voice-agent-api)

### Inferences
- **Cost of escalation:** 3 min/h escalated × 60 s × ~25–32 tok/s ≈ 4.5–5.8k audio tokens. At $0.30–$1.00/1M that is **<$0.01/h**. Even at the Live rate ($3/1M) it is ~$0.02/h. Text prompt and output add ~$0.001.
- **Latency:** an on-demand call takes ~1–3 s (estimate, unsourced). Never block the reply on it. Inject its result into the next turn's context, or, when the user explicitly asks for an evaluation (e.g. judging their singing), have Joyce say a short filler line while the call runs. That is a 300 ms tool call in the budget above, but the user expects it.
- Typical triggers: emotion confidence < 0.5 or classifier disagreement with transcript sentiment (sarcasm), CED events like laughter/crying/singing/music, a speaker-ID mismatch, and explicit meta-questions.

### Gaps
- Real per-call latency for a short (5–15 s) audio clip on Gemini Flash / Flash-Lite generateContent was not found.
- An official per-second audio token rate for non-Live Gemini models (commonly 32 tok/s) was not re-verified in this session.

## Summary: estimated added cost and latency per conversation hour (A vs B vs C)

### Takeaway
Against a baseline ear cost of $0.12/h (Soniox RT v5), **A adds ≈ $0–0.08/h** (on-device: $0; shared cloud CPU: ~$0.002–0.01/h; one dedicated small VM per stream: ~$0.07/h) **and ~0 ms** to the response path (≤50–150 ms if the LLM call waits for per-utterance emotion). **B adds ≈ $0.27/h (Gemini Live ear only) to ~$1.15+/h (gpt-realtime audio in)**, or $0.05–$2+/h self-hosted depending on concurrency. Latency is ~0 if run in parallel (results lag 0.5–2 s), and it restructures the whole stack if used as the brain. **C ≈ A + $0.005–0.02/h**, with async escalations off the response path.

### Cited Findings
- Soniox RT v5 $0.12/h (June 2026) — [Soniox pricing compare](https://soniox.com/compare/stt/pricing)
- AWS c7g.large $0.0725/h; c8g.large $0.0798/h (Jul 2026) — [Holori](https://calculator.holori.com/aws/ec2/c7g.large/us-east-1)
- Modal L4 ~$0.80/h base; CPU ~$0.047/core-h — [Spheron](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/)
- Gemini 3.1 Flash Live audio-in $3/1M (~$0.005/min), audio-out $12/1M (~$0.018/min) — [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)
- gpt-realtime-1.5 audio-in $32/1M, out $64/1M — [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-realtime-1.5)
- Qwen3-Omni first packet 234 ms theoretical; ~632 ms mean at concurrency 64 — [Qwen3-Omni report](https://arxiv.org/html/2509.17765v1); [vLLM blog](https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization)

### Inferences
| Option | Added $/conversation-hour (est.) | Added latency on reply path | Paralinguistic quality |
|---|---|---|---|
| A on-device (phone/Pi5/Jetson) | $0 | ~0 ms (parallel); +50–150 ms only if waiting on emotion | Good for fixed labels (emotion, speaker, events); no open-ended nuance |
| A cloud CPU (shared, batched) | ~$0.002–0.01 | ~0 ms + network tee (audio already streamed) | same |
| A cloud CPU (1 VM per stream) | ~$0.07–0.08 | same | same |
| A serverless GPU | ~$0.8–2.4/h per GPU; only cheap at ≥50 streams | + cold starts | same |
| B Gemini Live as ear (parallel) | ~$0.27–0.30 (listening) | 0 ms on current turn; info lags 0.5–2 s | Holistic, but lossy per CMU paper |
| B Gemini Live as whole brain+voice | ~$0.7+ (replaces LLM+TTS costs) | could reduce total; locks voice/LLM | affective dialog built in |
| B gpt-realtime | ~$1.15+ input alone | as above | similar class |
| B self-host Qwen3/3.5-Omni | $0.05–0.3 at high concurrency; $2+ for one user | 234–632 ms first packet | best open style-tracking per CMU, still ~54% |
| C (A + escalation ~5%) | A + ~$0.005–0.02 | 0 ms (async), 1–3 s only on explicit asks | best of both |

### Gaps
- None of the A-option numbers is benchmarked. They rest on model-size reasoning and must be validated with a 1-day ONNX benchmark on the target hardware.

## Recommended "fast and cheap" architecture and phased roadmap

### Takeaway
Keep Soniox → fast LLM → Google TTS as the critical path. Add a sidecar that tees the same PCM stream into a ring buffer and runs tiny ONNX models keyed off Soniox token timestamps and `<end>`. Inject a compact `<ear …/>` tag into the LLM turn. Escalate rare or rich cases to a cheap Gemini audio call. Prefer on-device for VAD, speaker ID and events (privacy, $0), and a small shared cloud CPU for emotion if the device is weak.

### Cited Findings
- Pipecat `ParallelPipeline` fan-out — [Pipecat docs](https://docs.pipecat.ai/guides/learn/pipeline); LiveKit `stt_node` override — [LiveKit nodes](https://docs.livekit.io/agents/logic/nodes.md); TEN graph audio-frame flows — [TEN overview](https://docs.agora.io/en/ai/ten-agent/framework-overview)
- Model building blocks: Silero VAD, CED-Tiny/Mini, emotion2vec+, WeSpeaker ResNet34 ONNX — sources as in section A
- Gemini Live `enable_affective_dialog` / `proactive_audio` (v1alpha) for a future native-audio mode — [Live API guide](https://ai.google.dev/gemini-api/docs/live-guide)

### Inferences
**Phase 0 (days): free signals.** Turn on Soniox diarization, endpoint detection and language ID. Derive DSP features from the ring buffer per utterance: RMS loudness, pitch mean/variance (YIN/pYIN), speech rate (tokens per second from Soniox `start_ms`/`end_ms`), pause lengths and overlap/interruptions. Inject them as tags. Cost $0, latency ~1–5 ms.

**Phase 1 (1–2 weeks): core sidecar.** Silero VAD (barge-in and gating), WeSpeaker ResNet34 or ECAPA voiceprint with an enrolled-user cache (cosine similarity against stored centroids, updated with an EMA on high-confidence utterances), emotion2vec+ base per utterance (9 classes + confidence), CED-Tiny on 1–2 s sliding windows for events (laughter, crying, music, singing, TV, dog, door…). Export all to ONNX, quantize to int8 (dynamic quantization for linear layers), benchmark RTF, and run on the client if possible, else on a shared Graviton/x86 CPU pool.

**Phase 2 (2–4 weeks): hybrid escalation.** Add a rule engine that, on triggers, sends the last N seconds (Opus/FLAC slice from the ring buffer) to a Gemini Flash/Flash-Lite audio call with a structured-output schema (`{tone, emotions[], sarcasm, singing:{in_tune, expressiveness}, notes}`). Results go into memory and the next turn.

**Phase 3 (later): optimise and personalise.** Use a shared-encoder multi-head model fine-tuned on Joyce's own logged and consented data (per-user emotion baseline, so "sad for this user" is relative). Consider probe heads on an audio-LLM encoder (the CMU finding shows encoder layers carry ~84% style info). Optionally A/B a Gemini Live (3.1 Flash Live / 3.8 Live) "native mode" with affective dialog for premium users, keeping the cascade as default for cost.

### Gaps
- No real-world A/B data on whether paralinguistic tags measurably improve user satisfaction in assistants.

## Engineering tips: ring buffers, windows, latency hygiene, speaker cache, privacy

### Takeaway
Treat the ear as asynchronous enrichment. It must never sit in series between Soniox's endpoint and the LLM call unless it is guaranteed to finish within ~50–100 ms. Use a timestamp-addressable ring buffer so any model can pull exact slices aligned to Soniox tokens.

### Cited Findings
- Soniox tokens carry `start_ms`/`end_ms`/`speaker`, final tokens are immutable, `<end>` marks the endpoint, and `<fin>` follows manual finalize — [Soniox WebSocket API](https://soniox.com/docs/api-reference/stt/websocket-api); [Endpoint detection](https://soniox.com/docs/stt/rt/endpoint-detection)
- Endpointing costs 100–300 ms. Aggressive VAD cuts latency but interrupts users. Streaming overlap makes perceived latency ≈ max(STT partial, LLM TTFT) + TTS first audio + orchestration — [FutureAGI measuring latency](https://futureagi.com/blog/how-to-measure-voice-ai-latency-2026/); [FutureAGI sub-500ms](https://futureagi.com/blog/sub-500ms-voice-ai-guide-2026/)
- Silero VAD was trained on 30 ms chunks and supports longer ones directly — [silero-vad](https://pypi.org/project/silero-vad)
- CED processes fixed-size 16×16 patches, so short windows (1–2 s) are cheap — [CED GitHub](https://github.com/richermans/ced)
- Soniox does not store real-time audio by default — [Soniox pricing compare](https://soniox.com/compare/stt/pricing)

### Inferences
- **Ring buffer:** 16 kHz mono int16, 30–60 s capacity (~1–2 MB), indexed by the same stream clock as the samples sent to Soniox. Record the stream-start offset so Soniox `start_ms` maps directly to buffer indices. Pre-allocate it and use a lock-free single-producer/multi-consumer design.
- **Windows:** use **1–2 s sliding windows with 0.5–1 s hop** for event tagging and running arousal (it catches laughter or a sigh mid-turn). Use **per-utterance** windows (Soniox final tokens between endpoints, merged by speaker) for emotion2vec and speaker embedding, which are more stable on 1–8 s of speech. Skip utterances under ~0.8 s for speaker ID.
- **Start early:** run emotion/speaker on the partial utterance as soon as ~1.5 s of speech exists, then refine at `<end>`. The tag is then almost always ready when the LLM call fires. If not ready within a ~50 ms grace period, send the LLM call without it and attach the result to the next turn or memory.
- **Speculative LLM start:** if Joyce already starts the LLM on stable partials, put ear tags in a late-bound "system addendum" slot, or only into the final request, so they don't break prefix caching of the stable prompt.
- **Speaker profile cache:** store 1–3 L2-normalised centroids per enrolled user (256-d floats ≈ 1 KB). Match with cosine distance and a threshold tuned on your own data. Update centroids with an EMA only on high-confidence, clean utterances. Map Soniox's per-session speaker label to the identity once per session and re-verify every N utterances. Keep it on device or encrypted at rest. A voiceprint is biometric data (GDPR Art. 9 and similar), so get explicit consent.
- **Privacy:** run VAD, speaker ID and event tagging on device and send only text tags to the cloud. Escalation clips go to a cloud audio LLM only on explicit triggers, with user-visible consent and no retention (short-lived, in-memory slices).
- **Don't double-pay for audio:** with Gemini Live as a parallel ear, silence is billed while listening, so gate it with VAD or use non-Live per-clip calls instead.
- **Batching in cloud:** gather utterances from multiple sessions into micro-batches (≤20 ms wait) per model, pin ONNX Runtime intra-op threads, and keep separate thread pools from the audio I/O loop.
- **Calibrate and smooth:** apply hysteresis/EMA on emotion over turns, report only above confidence thresholds, and tell the LLM the tags are probabilistic. This avoids "You sound sad!" false positives, which damage trust more than missing a cue.

### Gaps
- No vendor or engineering-blog source was found that specifically documents ring-buffer and timestamp alignment between Soniox tokens and sidecar models. These tips are standard practice, not sourced.
