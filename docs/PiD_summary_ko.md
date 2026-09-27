# PiD 전수조사 · 활용 · 수익화 정리 (한국어)

> 이 문서는 `bmshin94/PiD` 저장소를 처음부터 끝까지 훑어보며 나눈 대화를 정리한 기록이다.
> 작성일: 2026-09-27

---

## 0. 관련 링크 모음

| 구분 | 주소 |
|---|---|
| **이 저장소 (작업 중)** | https://github.com/bmshin94/PiD |
| **원본 공식 저장소 (NVIDIA Toronto AI Lab)** | https://github.com/nv-tlabs/PiD |
| 논문 (arXiv) | https://arxiv.org/abs/2605.23902 |
| 프로젝트 페이지 | https://research.nvidia.com/labs/sil/projects/pid/ |
| 모델 가중치 (HuggingFace) | https://huggingface.co/nvidia/PiD |
| 결과 비교 페이지 | https://research.nvidia.com/labs/sil/projects/pid/comparison.html |
| ComfyUI 통합 PR | https://github.com/Comfy-Org/ComfyUI/pull/14103 |
| ComfyUI 지원 요청 이슈 | https://github.com/Comfy-Org/ComfyUI/issues/14095 |
| 커뮤니티 노드 (ComfyUI-PiD) | https://github.com/Merserk/ComfyUI-PiD |
| 커뮤니티 포크 (업스케일러) | https://github.com/eskilnor/nvidia-PiD-upscaler |
| 학습 데이터셋 (MultiAspect-4K-1M) | https://w2genai-lab.github.io/UltraFlux/ |
| PixelDiT (백본 원저작) | https://pixeldit.github.io/ |

---

## 1. PiD란 무엇인가

### 한 줄 정의

**PiD = Pixel Diffusion Decoder.** 이미지 생성 모델(FLUX, SDXL, Qwen-Image 등)의 마지막 단계인
**VAE/RAE 디코더를 통째로 대체**하는 플러그 앤 플레이 부품이다.
잠재값(latent)을 입력받아 **고해상도 픽셀 공간에서 직접 확산(denoising)** 을 수행해
**디코딩과 업스케일링을 한 번에** 끝낸다.

### 기존 방식과의 비교

```
[기존] 프롬프트 → LDM → 512x512 latent → VAE 디코더 → 2048px → 별도 업스케일러 → 4K
[PiD]  프롬프트 → LDM → 512x512 latent → PiD (4 step) → 2048~4096px  ✅ 끝
```

| | VAE 디코더 | PiD |
|---|---|---|
| 성격 | 결정론적 복원 (복사기) | 생성형 확산 (화가) |
| 없던 디테일 | 복원 불가 | **생성 가능** |
| 4K 출력 | 별도 업스케일러 필요 | 단일 패스 |
| 확산 스텝 | - | **4 step** (DMD 증류) |
| 파이프라인 변경 | - | 디코더 한 줄만 교체 |

### 저장소 기본 정보

| 항목 | 내용 |
|---|---|
| 원본 | NVIDIA Toronto AI Lab (`nv-tlabs/PiD`) — 약 1.1k stars / 61 forks |
| 라이선스 | **Apache 2.0** (상업적 이용 자유) |
| 언어/스택 | Python 3.12 고정, PyTorch 2.10 + CUDA 12.8, diffusers 0.37, transformers 4.57 |
| 규모 | 344개 파일, Python 약 64,000줄 |
| 논문 저자 | Yifan Lu, Qi Wu, Jay Zhangjie Wu, Zian Wang, Huan Ling, Sanja Fidler, Xuanchi Ren |

---

## 2. 폴더 전수조사 결과

### 최상위 구조

```
PiD/
├── README.md                 # 16KB, 사용법 총정리 (가장 먼저 읽을 문서)
├── pyproject.toml            # 의존성 버전 고정 (torch==2.10.0 등)
├── uv.lock                   # 148KB, uv 잠금 파일 (재현성)
├── environment.yml           # conda (학습 시 CUDA 툴킷 설치용)
├── verify_env.py             # 환경 진단 스크립트 (import + CUDA 체크)
├── justfile                  # just 태스크 러너 (lint / format)
├── .pre-commit-config.yaml   # ruff 기반 커밋 훅
├── CONTRIBUTING.md           # DCO 서명 필수
├── LICENSE                   # Apache 2.0
├── CLAUDE.md                 # 프로젝트 페르소나 가이드
├── assets/      (9)          # 데모 이미지 + manifest jsonl
├── figures/     (1)          # teaser.jpg
├── docs/        (15)         # 문서, 영문/중문 병기
├── docker/      (4)          # NGC 베이스 학습용 Dockerfile
├── scripts/     (8)          # 학습 진입점, 데이터 준비, 검증
└── pid/         (296)        # 본체
```

### `pid/` 내부

```
pid/
├── _ext/imaginaire/  (126)  NVIDIA 내부 학습 프레임워크 "Imaginaire4" 복사본
│   ├── trainer.py                분산 학습 루프
│   ├── checkpointer/dcp.py       PyTorch DCP 분산 체크포인트
│   ├── lazy_config/              Detectron2식 "코드가 곧 설정" 시스템
│   ├── datasets/webdataset/      대용량 tar 샤드 스트리밍 데이터로더
│   └── utils/easy_io/            로컬 / S3 / HTTP 통합 IO 추상화
│
└── _src/  (170)  PiD 고유 코드
    ├── inference/   (15)  ← 실사용 진입점
    │   ├── from_ldm.py             텍스트 → LDM → PiD 디코드
    │   ├── from_clean.py           이미지 → VAE 인코드 → PiD 디코드 (초해상도)
    │   ├── from_boogu.py           Boogu-Image 전용 (옵션)
    │   ├── decoder.py              공용 PiD 로딩 / 디코딩 런타임
    │   ├── checkpoint_registry.py  (backbone, ckpt_type) → 체크포인트 경로
    │   ├── pipeline_registry.py    diffusers 파이프라인 정의 + 백본별 기본값
    │   ├── step_capture.py         LDM 중간 스텝 x_t 캡처 콜백
    │   └── prompts/                예제 프롬프트 txt
    ├── networks/    (7)
    │   ├── pixeldit_official.py    PixelDiT 백본
    │   ├── pid_net.py              PiD 본체 = PixelDiT + ControlNet식 LQ 주입
    │   ├── lq_projection_2d.py     저품질(LQ) 잠재값 → 조건 특징
    │   └── discriminators.py       증류용 판별자
    ├── models/      (7)   pixeldit_model / pid_model(teacher) / pid_distill_model(student)
    ├── tokenizers/  (11)  flux / flux2 / sdxl / qwenimage VAE, RAE / Scale-RAE 디코더
    ├── configs/     (45)  Hydra 실험 설정 (2k / 2kto4k / v1.5)
    ├── datasets/    (18)  증강기 + 열화(degradation) 시뮬레이터
    ├── callbacks/   (17)  학습 중 샘플링 / 모니터링 / wandb
    ├── evaluations/ (17)  PSNR·SSIM·LPIPS + DeQA 화질 점수 모델
    ├── losses/      (3)   DMD 증류 손실 + REPA 표현 정렬 손실
    └── modules/     (7)   DPM-Solver 샘플러, EMA, 컨디셔너
```

### 코드에서 확인한 핵심 설계

1. **`PidNet(PixDiT_T2I)`** — PixelDiT를 상속하고 **ControlNet 스타일로 LQ(저품질) 잠재값을
   N개 블록마다 게이트 주입**. 게이트는 `sigma_aware_per_token` 방식이라 노이즈 레벨에 따라
   조건 강도가 자동 조절된다.
2. **4-step 증류** — 배포 체크포인트가 전부 `distill_4step`. DMD 증류로 수십 스텝을 4스텝으로
   압축했기 때문에 "빠르다"가 성립한다.
3. **`step_capture.py`의 조기 종료 트릭** — LDM을 끝까지 돌리지 않고 중간 `x_t`를 가로채
   PiD에 넘긴다. PiD가 생성 능력이 있으므로 "덜 익은" 잠재값도 완성해 주고, 결과적으로
   **LDM 스텝까지 절약**된다.
4. **VAE 가중치 로컬 로딩** — 런타임에 HuggingFace를 다시 받지 않고 `checkpoints/`에서 읽는다.

### 체크포인트 3종

| 타입 | 특징 | 용도 |
|---|---|---|
| `2k` | 2048px 전용, 가장 선명 | 2K 결과물 |
| `2kto4k` | 2K~4K 가변 (v1) | 구버전, sd3 / sdxl |
| `2kto4k_v1pt5` | v1.5. 색 정확도 향상, 모서리 격자 아티팩트 제거, 애니·작은 얼굴 데이터 추가 | **4K 기본 선택** |

### 지원 백본 (12종)

`flux`, `flux2`, `flux2-klein-4b`, `flux2-klein-9b`, `sd3`, `sdxl`,
`qwenimage`, `qwenimage-2512`, `zimage`, `zimage-turbo`, `dinov2`(RAE), `siglip`(Scale-RAE)

---

## 3. 설치 및 사용법

### 하드웨어 / 환경 요구사항

| 항목 | 요구사항 |
|---|---|
| OS | **Linux 전용** (pyproject에 `sys_platform == 'linux'` 명시). Windows는 WSL2 |
| GPU | **NVIDIA GPU 필수** (CUDA 12.8) |
| VRAM | 2K는 24GB급 권장, 4K + Qwen 계열은 offload 없으면 80GB급 |
| Python | **3.12 고정** (`>=3.12,<3.13`) |
| 아키텍처 | x86_64 또는 aarch64 |

### 설치 A — uv (추론용, 권장)

```bash
pip install uv
uv python install 3.12
uv sync --frozen
source .venv/bin/activate
PYTHONPATH=. python verify_env.py      # [PASS] 나오면 성공
```

### 설치 B — 기존 환경 재활용

PyTorch(CUDA), `transformers>=4.57`, `diffusers>=0.37`가 이미 있다면:

```bash
pip install hydra-core omegaconf pyyaml \
    attrs einops loguru termcolor fvcore iopath wandb \
    imageio opencv-python-headless pandas \
    safetensors sentencepiece boto3 botocore
```

### 설치 C — conda (학습용)

```bash
conda env create -f environment.yml
conda activate pid
python -m pip install -e . --group full \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

### 체크포인트 다운로드

```bash
hf download nvidia/PiD --local-dir . --include "checkpoints/*"
```

### 실행 1 — 텍스트 → 이미지 (`from_ldm`)

```bash
PYTHONPATH=. python -m pid._src.inference.from_ldm --backbone flux \
    --prompt "A photorealistic tabby cat on a rustic wooden table, morning light" \
    --ldm_inference_steps 28 --save_xt_steps 24 \
    --output_dir ./results/demo/flux \
    --pid_inference_steps 4
```

4K + 4:3 비율로 뽑으려면 `--resolution 4096,3072 --pid_ckpt_type 2kto4k_v1pt5` 추가.
`--resolution`은 **최종 출력 크기**이며 LDM은 자동으로 1/4 해상도에서 돌아간다.

멀티 GPU:

```bash
PYTHONPATH=. torchrun --nproc_per_node=4 -m pid._src.inference.from_ldm \
    --backbone zimage --prompt_file pid/_src/inference/prompts/prompt_creative.txt \
    --ldm_inference_steps 50 --save_xt_steps 46 --compile \
    --output_dir ./results/zimage
```

### 실행 2 — 이미지 초해상도 (`from_clean`)

```bash
PYTHONPATH=. python -m pid._src.inference.from_clean --backbone flux \
    --input_path ./my_photo.jpg --prompt "a detailed photo" \
    --degrade_sigmas 0.0 \
    --output_dir ./results/upscaled \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
```

여러 장은 `--manifest assets/clean_image_manifest.jsonl` 사용.

### 주요 옵션 치트시트

| 옵션 | 의미 | 권장값 |
|---|---|---|
| `--backbone` | 어떤 생성모델용 디코더인지 | `flux` |
| `--pid_ckpt_type` | 체크포인트 종류 | 2K는 `2k`, 4K는 `2kto4k_v1pt5` |
| `--pid_inference_steps` | PiD 확산 스텝 | **4** |
| `--resolution` | 최종 출력 크기 | `4096,3072` 등 |
| `--save_xt_steps` | LDM 중간 잠재값 캡처 지점 | flux 24 / zimage 46 |
| `--scale` | 업스케일 배수 | 4 (siglip만 8) |
| `--cpu_offload` | VRAM 부족 시 | Qwen 계열 사실상 필수 |
| `--compile` | torch.compile 가속 | 배치 처리 시 |

### 백본별 권장 스텝

| Backbone | 기본 LDM steps | 권장 캡처 latent |
|---|---|---|
| flux / sd3 | 28 | step 24 |
| sdxl | 30 | step 26 |
| flux2 / zimage | 50 | step 46 |
| qwenimage / -2512 | 50 | step 44 |
| flux2-klein-4b/9b | 4 | x0 |
| zimage-turbo | 9 | x0 |

### 학습

```bash
# PixelDiT 파인튜닝
PYTHONPATH=. torchrun --nproc_per_node=4 -m scripts.train \
  --config=pid/_src/configs/pid_training/config.py \
  -- experiment="pixeldit_text_to_image_finetune_res_2048"
```

데이터는 MultiAspect-4K-1M을 받아 `scripts/sharding_wds.py`로 WebDataset 샤드로 변환한다.
자세한 내용은 `docs/training.md`, `docs/hydra_EN.md`, `docs/dataloader_EN.md` 참고.

---

## 4. 자주 묻는 것 정리

### 이것은 플러그인 / 스킬 / MCP 인가?

**셋 다 아니다.** Claude Code 플러그인도, Skill도, MCP 서버도 아니다.

| 구분 | 해당 여부 | 비고 |
|---|---|---|
| Claude Code 플러그인 | 아니오 | `.claude-plugin/` 없음 |
| Skill | 아니오 | `SKILL.md` 없음 |
| MCP 서버 | 아니오 | MCP 프로토콜 구현 없음 |
| Python 패키지 | **예** | `pyproject.toml`, `name = "pid"` |
| CLI 도구 | **예** | `python -m pid._src.inference.from_ldm ...` |
| 학습 프레임워크 | **예** | Hydra + FSDP 분산 학습 포함 |
| ComfyUI 노드 | 간접적으로 예 | PiD가 ComfyUI 본체에 머지됨 (PR #14103) |

### API 토큰이 필요한가?

**추론 자체는 어떤 API 키도 필요 없다. 100% 로컬 실행.**

| 상황 | 필요 여부 | 토큰 |
|---|---|---|
| PiD 추론 (로컬 GPU) | 불필요 | - |
| 체크포인트 다운로드 | 대체로 불필요 | 레이트리밋 시 `HF_TOKEN` |
| FLUX.1-dev 백본 받기 | **필요** | `HF_TOKEN` (gated 모델) |
| 학습 로깅 | 필요 | `WANDB_API_KEY` |
| S3 데이터셋 | 필요 | AWS 자격증명 (boto3) |
| OpenAI / Anthropic 등 | 전혀 무관 | - |

### 왜 GitHub에서 유명한가

1. **NVIDIA 공식 + Apache 2.0** — 상업 이용이 자유로운 연구실 코드는 드물다.
2. **Plug-and-Play** — 디코더 한 줄만 교체. 기존 LoRA / ControlNet / 워크플로 그대로 유지.
3. **ComfyUI 통합** — 2026-05-27 본체 머지. 코딩 없이 노드 하나로 체험 가능 → 확산 폭발.
4. **모두가 겪던 통증** — "AI 이미지를 4K로 올리면 왜 흐릿한가"를 정확히 해결.
5. **지원 백본이 넓다** — 주류 모델 12종 커버. 내가 쓰는 모델이 목록에 있을 확률이 높다.
6. **업데이트 속도** — 5/25 릴리스 → 5/27 ComfyUI → 6/2 SDXL·Qwen → 7/9 v1.5 + 학습코드 → 7/14 Boogu.
7. **학습 코드까지 전부 공개** — teacher 학습, DMD 증류, 데이터 준비, Docker까지.
8. **속도도 확보** — 4스텝 증류 + `torch.compile`.
9. **문서 품질** — 복붙 가능한 예제, 백본별 표, 영문/중문 병기 문서 15개.

### 로컬 에이전트 구축에 도움이 되는가

**직접적으로는 무관, 간접적으로는 매우 유용.**

- 직접: LLM 추론 루프도, 툴 호출도, 메모리도 없다. LLM 에이전트 코드는 0줄.
- 간접 1: **에이전트가 쓸 이미지 도구**로 래핑하기 좋다 (MCP 서버로 `upscale_image` 노출).
  API 비용이 0원이라 로컬 에이전트 철학과 잘 맞는다.
- 간접 2: **로컬 GPU 서빙 노하우 교재**. `model_loader.py`(체크포인트 동적 로딩),
  `decoder.py`(모델 상주·재사용), `context_parallel.py`(큰 입력 분할),
  `easy_io/`(로컬·S3·HTTP 통합 IO) 등은 그대로 응용 가능하다.
- 간접 3: **Hydra / LazyConfig 실험 관리 패턴**은 에이전트 조합 실험에도 유용하다.

### React나 PHP로 만들 수 있는가

**모델 실행은 불가, 서비스 레이어는 전부 가능.**

- 불가: PiD 모델 실행 자체는 PyTorch + CUDA 영역이다. JS/PHP에 등가물이 없고,
  DiT 트랜스포머 + 4K 픽셀 공간은 브라우저 메모리로 감당이 안 된다.
- 가능: UI, 인증, 결제, 크레딧, 큐, 스토리지, 관리자 — 전부 React/PHP로 구현한다.

권장 아키텍처:

```
React (Next.js)            업로드 UI / 진행률 / 비포·애프터 슬라이더 / 결제
        │ REST · WebSocket
PHP (Laravel)              인증 / 크레딧 차감 / 결제 / 작업 큐 등록 / 관리자
        │ Redis Queue · RabbitMQ
Python (FastAPI) GPU 워커   from_clean.py 호출 / 모델 메모리 상주 / S3 업로드  ← 여기만 파이썬
```

실전 주의사항:

| 항목 | 주의 |
|---|---|
| 모델 로딩 | 수십 초 소요. **요청마다 로드하면 안 된다.** 프로세스에 상주시킬 것 |
| 동시성 | GPU 1장 = 기본 1작업. 큐로 직렬화 |
| 타임아웃 | 4K는 수 초~수십 초. **비동기 + 폴링/WebSocket 필수** |
| VRAM | OOM 시 워커 사망. 헬스체크 + 자동 재시작 |
| 비용 | GPU 시간당 과금. idle 시 스케일 다운 |
| torch.compile | 첫 요청만 느림 → 워커 시작 시 워밍업 호출 |

---

## 5. 수익화 아이디어

### 전제 — 왜 돈이 되는 구조인가

| 유리한 점 | 의미 |
|---|---|
| Apache 2.0 | 로열티·매출 공유 0원, 상업 서비스 탑재 자유 |
| API 키 불필요 | 변동비가 GPU 임대료/전기세뿐 → 원가 구조 우위 |
| NVIDIA 브랜드 | 마케팅 신뢰도를 무료로 확보 |
| 4스텝 증류 | 처리 속도 → GPU 회전율 → 단위 원가 하락 |
| 검증된 통증 | Topaz, Magnific 등이 이미 유료 수요를 증명 |

불리한 점도 명확하다. **NVIDIA GPU 인프라 비용이 들고, 오픈소스라 경쟁자도 같은 코드를 쓴다.**
따라서 차별화는 코드가 아니라 **UX · 데이터 · 워크플로 · 도메인 특화**에서 나와야 한다.

### 아이디어 1 — 웹 업스케일링 SaaS (추천 1순위)

- 타깃: AI 아티스트, 이커머스 셀러, 인쇄/굿즈 업체, 웹툰 작가
- 차별점: 기존 업스케일러는 픽셀 보간, PiD는 **디테일 생성**. AI 생성 이미지에 특히 강함
- 가격: Free 5장/월 → Basic 9,900원(100장) → Pro 29,900원(500장 + 4K + API)
- 기술: React + Laravel + FastAPI GPU 워커
- 초기 비용: RTX 4090 서버 임대 월 30~60만원, 또는 서버리스 GPU(RunPod/Modal) 사용량 과금
- 손익분기: Basic 기준 월 60~100명이면 인프라비 커버
- 킬러 기능: 비포/애프터 슬라이더, ZIP 배치 업로드, 백본 자동 감지, 인쇄 프리셋(A4 300dpi 등)

로드맵: 1주 품질 검증 → 2~3주 GPU 워커 + 큐 → 4~5주 React 랜딩 + 무료 체험 →
6주 결제·크레딧 → 7주 커뮤니티 런칭.

### 아이디어 2 — ComfyUI 프리미엄 노드팩 / 워크플로

- 근거: PiD가 ComfyUI에 머지됐지만 체크포인트 수동 다운로드, 경로 설정, VRAM 튜닝이 여전히 장벽
- 상품: 원클릭 설치 스크립트, VRAM별 최적 워크플로 JSON 30종, 배치 자동화 노드, 백본 자동 매칭
- 판매처: Gumroad, Patreon, Ko-fi
- 가격: 단품 $15~29 / 멤버십 월 $5~10
- 장점: **인프라 비용 0원**, 순수 지식 판매
- 전략: 코어는 무료로 공개해 스타를 모으고, 편의 기능과 지원만 유료화

### 아이디어 3 — 도메인 특화 파인튜닝 (진입장벽 = 해자)

| 분야 | 근거 |
|---|---|
| 웹툰 / 애니 | v1.5가 애니 데이터를 추가했지만 한국 웹툰체는 별도 학습 필요 |
| 패션 / 이커머스 | 원단 질감, 스티치 디테일이 매출에 직결 |
| 부동산 / 건축 | 투시도 4K 렌더링 |
| 의료 영상 | 단가는 높지만 규제 주의 |
| 인물 사진 | 작은 얼굴 복원이 PiD 강점 |
| 문서 / OCR | 작은 글씨 복원 |

수익 모델: 커스텀 디코더 학습 용역(건당 500만~3000만원), 가중치 라이선스, 결과물 단위 과금.
현실: 멀티 GPU(A100/H100 8장급)가 필요해 진입장벽이 높고, 그것이 곧 해자다.
클라우드 GPU 단기 임대로 PoC → 계약 후 본 학습 순서를 권장.

### 아이디어 4 — 에이전트용 MCP 툴 / 로컬 AI 스위트

```
LocalVision MCP Suite
├── pid_upscale      PiD 초해상도
├── pid_decode       잠재값 → 픽셀
├── batch_process    폴더 일괄 처리
└── quality_score    DeQA 화질 평가 (PiD에 내장)
```

- 셀링포인트: **데이터가 외부로 나가지 않음 + API 비용 0원** → 기업 영업에 강력
- 가격: 개인 무료 / 팀 $29·월 / 엔터프라이즈 온프레미스 라이선스
- 타이밍: MCP 생태계 성장기 = 선점 기회

### 아이디어 5 — B2B API

```bash
curl -X POST https://api.example.com/v1/upscale \
  -H "Authorization: Bearer sk-..." \
  -F "image=@photo.jpg" -F "scale=4"
```

- 타깃: 이커머스 플랫폼, 사진 앱, 인쇄 서비스, 게임사
- 가격: 1장당 30~100원 (원가가 전기세라 공격적 가격 책정 가능)
- 필수: SLA, 레이트리밋, 웹훅, 문서

### 아이디어 6 — 교육 콘텐츠 / 컨설팅

| 상품 | 가격대 |
|---|---|
| 온라인 강의 | 10~30만원 |
| 유료 뉴스레터 | 월 5천~1만원 |
| 기업 세미나 | 회당 100~300만원 |
| 도입 컨설팅 | 시간당 15~30만원 |

강의 주제는 저장소 안에 전부 있다: ControlNet식 조건 주입(`pid_net.py`),
DMD 증류(`dmd_losses.py`), FSDP + 컨텍스트 병렬, WebDataset 샤딩,
Hydra LazyConfig, torch.compile 최적화.
GPU 없이 시작 가능하며, 다른 아이디어의 마케팅 채널로도 작동한다.

### 아이디어 7 — 틈새 버티컬 앱 (소자본)

| 앱 | 컨셉 | 강점 |
|---|---|---|
| AI 굿즈 프린트샵 | 업스케일 → 포스터/티셔츠 제작·배송 | 업스케일은 무료 미끼, 수익은 실물에서 |
| 옛날 사진 복원 | 흑백 사진 고해상도화 | 감성 소구, 명절 시즌 특수 |
| 셀러용 배치 처리 | 100장 일괄 업스케일 | 이커머스 셀러는 지불의사 높음 |
| 게임 텍스처 업스케일 | 레트로 텍스처 4K | 모딩 커뮤니티 Patreon 문화 |
| 웹툰 작가 도구 | 콘티 → 고해상도 원고 보조 | 국내 웹툰 시장 강세 |

### 추천 실행 순서

```
1단계 (0~2개월)  아이디어 2 (ComfyUI 워크플로) + 6 (콘텐츠)
                 → GPU 투자 없이 시장 검증 + 개인 브랜딩
2단계 (2~5개월)  아이디어 1 (웹 SaaS)
                 → React/Laravel 강점 활용, 1단계 팔로워가 초기 유저
3단계 (6개월~)   아이디어 4 (MCP 툴) 또는 5 (B2B API)
장기 해자        아이디어 3 (도메인 파인튜닝)
```

**당장 이번 주에 할 일**

1. 클라우드 GPU(RunPod 등) 1시간 임대해 `from_clean.py` 실제 실행
2. 본인 이미지 10장 업스케일 후 Topaz / Magnific 등과 결과 비교
3. 1장당 처리 시간 + VRAM 사용량 기록 (모든 가격 책정의 기초 데이터)
4. 비교 이미지로 블로그·SNS 글 1건 발행 → 반응으로 시장 검증

---

## 6. 요약 한 장

| 질문 | 답 |
|---|---|
| 뭐하는 건가 | VAE 디코더를 대체하는 픽셀 확산 디코더. 디코딩 + 업스케일을 한 번에 |
| 언제 쓰나 | 4K 이미지 생성, 기존 이미지 초해상도, ComfyUI 워크플로, 연구, 파인튜닝 |
| 플러그인/스킬/MCP인가 | 전부 아니다. Python 패키지 + CLI + 학습 프레임워크 |
| API 토큰 | 추론은 불필요. gated 백본은 `HF_TOKEN`, 학습 로깅은 `WANDB_API_KEY` |
| 왜 유명한가 | NVIDIA + Apache 2.0 + plug-and-play + ComfyUI 통합 + 빠른 업데이트 |
| 로컬 에이전트 | 직접 무관, 에이전트용 이미지 툴 / GPU 서빙 교재로는 매우 유용 |
| React/PHP | 모델 실행은 불가, 서비스 레이어는 전부 가능 (GPU 워커만 Python) |
| 수익화 | SaaS · ComfyUI 노드팩 · 도메인 파인튜닝 · MCP 툴 · B2B API · 교육 · 버티컬 앱 |

---

*이 문서는 저장소 전수조사(344개 파일, Python 약 64,000줄) 결과를 바탕으로 작성되었다.*
