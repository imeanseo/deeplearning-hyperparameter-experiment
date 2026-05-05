# 실용자연어처리 HW1 보고서 (초안)

## 1. 과제 목적 및 제약

- 데이터셋: `Sp1786/multiclass-sentiment-analysis-dataset`
- 목표: **5개의 report된 성능 중 최소 1개 이상이 기존 모델(67.38%)보다 향상**
- 제약 조건:
  - `MLP` 모델 구조 자체는 변경하지 않음
  - `BERT`, `RNN` 등 구조 변경/대체 모델은 사용하지 않음
  - 하이퍼파라미터 및 벡터화/전처리는 변경 가능

---

## 2. 실험 설정

### 2-1. 공통 설정

- 분류 클래스: negative(0), neutral(1), positive(2)
- 분할: train / validation / test
- 학습 프레임워크: PyTorch
- 튜닝 도구: Weights & Biases (Bayesian Sweep)

### 2-2. 변경한 하이퍼파라미터 (5개 이상)

본 과제에서 변경한 주요 하이퍼파라미터:

1. `learning_rate`
2. `hidden_size`
3. `dropout`
4. `weight_decay`
5. `batch_size`
6. `num_epochs`
7. `vectorizer` 종류 (BoW / TF-IDF / GloVe / MiniLM / MPNet)
8. TF-IDF 세부 옵션 (`max_features`, `min_df`, 전처리 함수 등)

> 각 하이퍼파라미터의 의미와 사용 이유는 4절(실험별 분석)에 포함.

---

## 3. 5개 모델 성능 요약 (필수)

| 실험 | 방법 | Test Accuracy | Baseline 대비 |
|---|---|---:|---:|
| Exp1 | BoW Baseline | 67.38% | - |
| Exp2 | BoW + Sweep | 68.70% | +1.32%p |
| Exp3 | TF-IDF + Sweep | 68.30% | +0.92%p |
| Exp4 | GloVe300 + Sweep | 62.00% | -5.38%p |
| Exp5 | MiniLM + Sweep | 67.82% | +0.44%p |
| Exp6 | MPNet + Sweep | 66.28% | -1.10%p |

### 과제 조건 충족 여부

- **충족**
- 근거: Exp2(68.70%), Exp3(68.30%), Exp5(67.82%)가 baseline(67.38%)보다 높음

---

## 4. 실험별 요약 분석 (간략)

### Exp1. BoW Baseline
- 목적: 비교 기준점 확보
- 결과: 67.38%
- 해석: 기준 성능

### Exp2. BoW + Sweep
- 변경 이유: 하이퍼파라미터 자동 탐색으로 일반화 성능 개선
- 핵심 파라미터: `lr`, `hidden_size`, `dropout`, `batch_size`, `num_epochs`, `weight_decay`
- 결과: 68.70% (+1.32%p)
- 해석: baseline 대비 개선 성공

### Exp3. TF-IDF + Sweep
- 변경 이유: 단문 감성 데이터에서 단어 중요도(IDF) 반영이 유리
- 결과: 68.30% (+0.92%p)
- 해석: 과제 제약(MLP 고정) 내에서 가장 안정적으로 성능 향상

### Exp4. GloVe300 + Sweep
- 변경 이유: 분산표현 임베딩 활용 시도
- 결과: 62.00%
- 해석: 평균 임베딩 과정에서 순서/부정 패턴 정보 손실

### Exp5. MiniLM + Sweep
- 변경 이유: 문맥 임베딩 기반 성능 향상 기대
- 결과: 67.82% (baseline 소폭 상회)
- 해석: MLP 고정 제약 아래에서 임베딩 장점이 충분히 발휘되지 않음

### Exp6. MPNet + Sweep
- 변경 이유: 더 큰 문장 임베딩 모델 효과 확인
- 결과: 66.28%
- 해석: 고차원 임베딩 대비 일반화 성능 개선 실패

---

## 5. 추가 개선 실험 (선택/가산점)

Exp3 기반 전처리/피처 엔지니어링 실험:

- v6: 68.36%
- v7: 68.61%
- v8: **69.13%** (최고)
- v9: 68.76%
- v10: 69.13% (v8과 동일)

해석:
- 전처리 + 핸드크래프트 피처 + sweep 조합이 가장 효과적이었음
- 다만 70%는 도달하지 못했고, dev-test gap이 남음

---

## 6. 결론

1. 과제 요구사항(5개 실험, baseline 대비 향상) 충족
2. 최종 제출 모델은 `Exp3_v8_Final_Sweep` (69.13%)
3. 기본 5개 실험 중 최고 성능은 `Exp3_TFIDF_Sweep` (68.30%)
4. 데이터 특성상 단순 키워드 기반 신호를 잘 활용하는 TF-IDF가 가장 효율적

---

## 7. 제출 체크리스트

- [ ] `W5_HW1_PracticalNLP_{학번}_{이름}.ipynb` 저장 (출력 포함)
- [ ] 최고 성능 모델 코드 포함 (Exp3_v8 기준)
- [ ] 5개 실험 구현 내용 + 성능 + 간략 분석 포함
- [ ] 각 실험 결과 **스크린샷** 첨부
- [ ] 모델 구조(MLP) 변경하지 않았음을 보고서에 명시

---

## 8. 스크린샷 배치 가이드

각 실험 섹션에 아래 2개 이미지를 넣으면 깔끔합니다.

1. 해당 실험의 학습 로그(Dev/Test)
2. W&B sweep 결과(최적 config + metric)

권장 캡션 예시:
- `그림 1. Exp3_v8 학습 결과 (Best Dev 70.05, Test 69.13)`
- `그림 2. Exp3_v8 W&B Sweep 최적 하이퍼파라미터`
