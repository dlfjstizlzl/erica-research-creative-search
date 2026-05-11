# 🚀 Creative Search: Evolutionary AI Idea Pipeline

**Creative Search**는 단순한 LLM 답변을 넘어, 다양한 전문가 페르소나의 충돌과 진화론적 알고리즘(변이 및 결합)을 통해 **혁신적이고 실행 가능한 해결책**을 찾아내는 비동기 AI 파이프라인입니다.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)

---

## ✨ 핵심 기능 (Core Features)

### 🧠 다차원 페르소나 생성 (Multi-Persona Generation)
*   **80개 이상의 전문 페르소나**: 마케터, 시스템 아키텍처, 행동 과학자, 게임 디자이너 등 다양한 시각에서 초기 아이디어 씨앗(Seed)을 생성합니다.

### 🧬 진화론적 아이디어 루프 (Evolutionary Loop)
*   **Mutation (변이)**: 기존 아이디어의 핵심 메커니즘을 유지하면서 타겟이나 방식을 비틉니다.
*   **Recombination (결합)**: 상충하는 두 아이디어 사이의 전략적 갈등(Conflict)을 찾아내고, 이를 해결하는 제3의 통합 모델을 합성합니다.
*   **Pareto Selection**: 창의성, 적합성, 실행 가능성을 기준으로 우수한 아이디어만 다음 세대로 전승합니다.

### 🛠️ 로컬 환경 최적화 (Local Backend Stability)
*   **전역 동시성 제어 (Global Semaphore)**: 로컬 GPU 자원 고갈을 방지하기 위해 모델 실행을 지능적으로 직렬화/병렬화합니다. (기본 2개 동시 처리)
*   **강력한 JSON 파싱 & 자가 수정**: LLM의 불안정한 출력을 정규표현식과 `ast.literal_eval`로 보정하며, 실패 시 스스로 프롬프트를 수정하여 재시도합니다.
*   **실시간 퍼포먼스 모니터링**: 각 단계 및 모델별 소요 시간을 초 단위로 추적하여 병목 구간을 시각화합니다.

---

## 🚀 시작하기 (Quick Start)

### 1. 필수 조건
*   [Ollama](https://ollama.ai/) 설치 및 실행
*   사용 권장 모델: `llama3.1:8b`, `huihui_ai/aya-expanse-abliterated:8b` 등 (8B급 이상 권장)

### 2. 환경 설정 (`.env`)
프로젝트 루트에 `.env` 파일을 생성하고 설정을 최적화합니다.

```bash
# 로컬 Ollama 주소 및 타임아웃(초)
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=300

# 모델 구성 (추론형 모델보다는 일반 모델이 채점/결합 시 유리함)
OLLAMA_GENERATOR_MODELS=llama3.1:8b,huihui_ai/aya-expanse-abliterated:8b
JUDGE_MODEL=llama3.1:8b
EXTRACTOR_MODEL=llama3.1:8b
COMBINER_MODEL=llama3.1:8b

# 파이프라인 설정
GENERATOR_SAMPLE_COUNT=5      # 초기 생성 개수
SEARCH_MAX_GENERATIONS=2      # 진화 반복 횟수
```

### 3. 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run streamlit_app.py
```

---

## 📊 파이프라인 워크플로우 (Workflow)

1.  **Reframing**: 입력된 문제를 AI가 다각도에서 분석하여 더 혁신적인 질문으로 재정의합니다.
2.  **Base Generation**: 다양한 페르소나가 초기 아이디어를 제안합니다.
3.  **Evolutionary Stage**: 
    *   아이디어를 변이시키고 결합하여 새로운 형태를 만듭니다.
    *   LLM-as-a-Judge가 각 아이디어에 점수를 매깁니다.
4.  **Final Selection**: 파레토 최적(Pareto Optimal)에 가까운 최종 결과물을 선정합니다.

---

## 🛠️ 기술적 하이라이트 (Technical Highlights)

*   **Async/Await 기반**: 모든 LLM 요청은 비동기로 처리되어 대기 시간을 최소화합니다.
*   **Robust Parsing**: Llama 3.1 등에서 발생하는 중첩 따옴표(`""value""`) 문제를 자동으로 해결하는 정규식 필터가 내장되어 있습니다.
*   **Performance Tracking**: 각 단계의 소요 시간(`elapsed seconds`)을 기록하여 최적화 데이터를 제공합니다.
*   **Telemetry**: 모든 실행 결과는 `/results` 폴더에 JSON 형태로 저장되어 사후 분석이 가능합니다.

---

## 🏗️ 상세 기술 아키텍처 (Technical Deep Dive)

### 1. 비대칭 모델 아키텍처 (Asymmetric Model Architecture)
본 프로젝트는 파이프라인의 각 단계마다 요구되는 지능의 성격에 따라 모델을 다르게 배치합니다.
*   **Generation Stage**: 창의성과 다양한 시각을 위해 Gemma 4B, Aya 8B 등 여러 모델을 풀(Pool)로 운영하여 아이디어의 '유전자 다양성'을 확보합니다.
*   **Judging & Synthesis Stage**: 일관된 기준의 평가와 정교한 구조적 결합을 위해 Llama 3.1 8B와 같은 논리적 안정성이 높은 모델을 고정 배치합니다.

### 2. 다단계 자가 수정 파싱 시스템 (Multi-stage Self-Correction)
로컬 LLM의 불안정한 응답 형식을 해결하기 위해 다음과 같은 다단계 방어 기제를 가집니다.
*   **Regex Extraction**: 응답 내의 불필요한 서술문을 제거하고 JSON 블록만 추출합니다.
*   **AST Literal Evaluation**: 표준 `json` 라이브러리가 실패할 경우, Python의 `ast.literal_eval`을 통해 싱글 쿼트(`'`)나 비표준 형식을 복구합니다.
*   **Recursive Self-Correction**: 파싱 실패 시 에러 메시지를 LLM에게 피드백으로 전달하여, 스스로 형식을 교정하여 재응답하도록 최대 2회 재시도합니다.

### 3. 전역 자원 오케스트레이션 (Global Resource Orchestration)
로컬 GPU VRAM의 한계를 극복하기 위해 `asyncio.Semaphore` 기반의 중앙 제어 시스템을 구현했습니다.
*   수십 개의 비동기 요청이 발생하더라도, 실제 추론(Inference) 단계에서는 설정된 값(기본 2개)만큼만 GPU를 점유하도록 스케줄링하여 시스템 다운을 방지합니다.

### 4. 진화론적 선택 및 니칭 (Evolutionary Selection & Niching)
단순히 합계 점수가 높은 아이디어만 남기지 않습니다.
*   **Pareto Front**: 독창성(Novelty)과 실행 가능성(Feasibility) 사이의 상충 관계를 고려하여 최적의 균형을 가진 아이디어를 선별합니다.
*   **Diversity Preservation**: 유사한 아이디어가 풀을 점유하지 않도록 임베딩 기반의 유사도 필터링을 통해 전략적 방향성이 다른 아이디어들을 골고루 보존합니다.

---

## ⚠️ 주의사항 (Notes)
*   로컬 GPU VRAM이 부족할 경우 `.env`에서 병렬 처리 수준을 낮추거나 모델 크기를 조절하세요.
*   **Reasoning(생각하는) 모델**은 갈등 분석 단계에서 토큰 제한으로 인해 결과가 끊길 수 있으므로 일반 인스트럭션 모델 사용을 권장합니다.

---
