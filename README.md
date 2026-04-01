# creative-search

Ollama 기반의 아이디어 탐색 시스템입니다.  
하나의 답을 바로 내는 대신, 여러 seed 아이디어를 만들고, mutation과 recombination으로 탐색 공간을 확장한 뒤, 최종적으로 `best_practical`, `best_balanced`, `best_wild`를 고릅니다.

## 개요

이 프로젝트는 다음 문제를 다룹니다.

- 한 번의 프롬프트로 그럴듯한 답 하나를 받는 것
- 여러 아이디어를 생성하고, 서로 변형/조합하면서 더 넓은 해 공간을 탐색하는 것

현재 구조는 `탐색 시스템` 쪽에 가깝습니다.

- `base generation`: 초기 seed 아이디어 생성
- `mutation`: 부모 아이디어를 전략적으로 비틀어 새 후보 생성
- `recombination`: 서로 다른 부모 아이디어를 충돌 기반으로 조합
- `pool`: 현재 살아있는 후보군 유지
- `archive`: 전체 lineage와 세대 기록 보존
- `final ranking`: practical / balanced / wild 기준으로 최종 선택

## 주요 특징

- Ollama 로컬 모델 사용
- 다중 모델 + 다중 페르소나 기반 base generation
- iterative search loop
- mutation / recombination 기반 아이디어 확장
- archive 기반 lineage 추적
- Streamlit 대시보드 지원

## 디렉터리 구조

```text
creative-search/
├── main.py
├── config.py
├── .env
├── .env.example
├── requirements.txt
├── streamlit_app.py
├── core/
│   ├── models.py
│   └── utils.py
├── llm/
│   └── ollama_client.py
├── pipeline/
│   ├── archive.py
│   ├── combiner.py
│   ├── filter.py
│   ├── generator.py
│   ├── mutator.py
│   ├── pool.py
│   ├── runner.py
│   ├── scoring.py
│   └── selection.py
├── prompts/
│   ├── base_generation.txt
│   ├── combination.txt
│   ├── diversity_filter.txt
│   └── mutation.txt
├── data/
│   └── problems.json
└── results/
```

## 파이프라인 플로우

현재 전체 흐름은 아래와 같습니다.

```text
Problem Input
  ↓
Base Generation
  ↓
Initial Scoring
  ↓
Pool + Archive Init
  ↓
[Iterative Search Loop]
  ├─ Parent Selection
  ├─ Mutation
  ├─ Recombination
  ├─ Candidate Scoring
  ├─ Pool Update
  └─ Archive Update
  ↓
Final Filter
  ↓
Final Selection
  ├─ Best Practical
  ├─ Best Balanced
  └─ Best Wild
  ↓
Save JSON
```

### 단계별 설명

1. `generator`
- 여러 persona와 generator model을 조합해 초기 seed를 만듭니다.

2. `scoring`
- novelty, problem_fit, mechanism_clarity 등 점수를 계산합니다.

3. `pool`
- 현재 세대에서 살아 있는 후보군을 유지합니다.

4. `mutation`
- 부모 1개를 기반으로 새 전략 후보를 만듭니다.

5. `recombination`
- 부모 2개를 충돌 기반으로 결합해 새 operating model을 만듭니다.

6. `archive`
- 각 후보의 origin, generation, parent_ids, survived 여부를 기록합니다.

7. `filter`
- obvious duplicate와 near-duplicate를 제거합니다.

8. `selection`
- 최종적으로 practical / balanced / wild best를 고릅니다.

## 점수 구조

아이디어에는 대략 아래 점수가 붙습니다.

- `novelty`
- `problem_fit`
- `mechanism_clarity`
- `mutation_distance`
- `mutation_quality`
- `combination_quality`
- `feasibility`
- `risk`
- `creativity`

이 점수는:

- 후보 비교
- parent selection
- final best selection
- 결과 분석

에 사용됩니다.

## 환경 설정

1. `.env.example`을 복사해서 `.env`를 만듭니다.

```bash
cp .env.example .env
```

2. 필요한 Ollama 모델을 준비합니다.

예시:

- `gemma3:4b`
- `qwen3:8b`
- `embeddinggemma`

3. `.env`를 프로젝트 목적에 맞게 수정합니다.

주요 항목:

- `OLLAMA_HOST`
- `OLLAMA_MODEL`
- `OLLAMA_GENERATOR_MODELS`
- `COMBINER_MODEL`
- `OLLAMA_EMBED_MODEL`
- `GENERATOR_SAMPLE_COUNT`
- `MUTATION_COUNT`
- `SEARCH_MAX_GENERATIONS`
- `COMBINATION_PAIR_COUNT`
- `POOL_MAX_SIZE`

## 실행 방법

### 1. CLI 실행

문제를 직접 넣어서 실행:

```bash
python3 main.py --problem "도시 내 음식물 쓰레기를 50% 줄일 수 있는 방법"
```

`data/problems.json`의 문제를 인덱스로 실행:

```bash
python3 main.py --index 0
```

실행 결과는 `results/run_YYYYMMDD_HHMMSS.json`에 저장됩니다.

### 2. Streamlit 대시보드 실행

의존성 설치:

```bash
pip install -r requirements.txt
```

대시보드 실행:

```bash
streamlit run streamlit_app.py
```

대시보드에서 가능한 것:

- problem 입력 후 파이프라인 실행
- 실행 로그 실시간 확인
- recent runs 탐색
- best 결과 확인
- base / mutation / combination / filtered / archive 상세 보기

## `.env` 예시 역할 배치

현재 추천 예시는 다음과 같습니다.

- base generation:
  - `OLLAMA_GENERATOR_MODELS=gemma3:4b,huihui_ai/aya-expanse-abliterated:8b`
- mutation:
  - `OLLAMA_MODEL=gemma3:4b`
- recombination:
  - `COMBINER_MODEL=qwen3:8b`
- scoring embeddings:
  - `OLLAMA_EMBED_MODEL=embeddinggemma`

이 구성은:

- seed 다양성은 넓게 확보
- mutation은 빠르게 반복
- recombination은 더 강한 composition 모델 사용

이라는 의도입니다.

## 결과 JSON 구조

실행 결과에는 보통 다음 필드가 들어갑니다.

- `problem`
- `output_language`
- `base_ideas`
- `mutated_ideas`
- `combined_ideas`
- `filtered_ideas`
- `best_practical`
- `best_balanced`
- `best_wild`
- `archive`
- `archive_summary`

각 idea에는 보통 아래 정보가 포함됩니다.

- `id`
- `title`
- `strategy_type`
- `description`
- `mechanism`
- `target_user`
- `execution_context`
- `expected_advantage`
- `origin_type`
- `parent_id`
- `parent_ids`
- `generation`
- `depth`
- `source_model`
- `source_persona`
- `scores`
- `score_meta`

## 추천 사용 방식

문제가 너무 넓으면 결과가 과도하게 추상적이 되기 쉽습니다.  
가능하면 아래처럼 문제를 조금 더 구체화하는 것을 권장합니다.

나쁜 예:

- `지속가능한 발전을 위한 프로젝트`

좋은 예:

- `도시 내 음식물 쓰레기를 50% 줄일 수 있는 방법`
- `대학생의 지루한 반복 과제 완료율을 높이는 방법`
- `예산 1천만원 이하로 지역 카페 일회용컵 사용을 줄이는 프로젝트`

## 현재 한계

- 문제 문장 품질에 결과가 민감합니다.
- recombination 품질은 mutation보다 아직 불안정할 수 있습니다.
- novelty가 일부 run에서 높게 나오는 경향이 있습니다.
- filtered pool은 shortlist라기보다 dedupe된 broad pool에 가깝습니다.

## 다음 개선 후보

- recombination pair selection 강화
- concrete project mode / open exploration mode 분리
- archive lineage 시각화 강화
- final ranking weight 튜닝

## 참고

이 프로젝트는 로컬 Ollama 서버가 실행 중이라는 가정을 전제로 합니다.
