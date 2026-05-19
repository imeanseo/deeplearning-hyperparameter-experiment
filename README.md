# Deep Learning Hyperparameter Experiment

트위터/SNS 스타일 텍스트 **3-class 감성 분류** 과제.
다양한 텍스트 벡터화 방법 + W&B Sweep으로 하이퍼파라미터를 최적화하며 성능을 비교 분석했습니다.

- **데이터셋**: [`Sp1786/multiclass-sentiment-analysis-dataset`](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset)
- **클래스**: negative(0) / neutral(1) / positive(2)
- **모델**: MLP (구조 고정, 하이퍼파라미터만 조정)
- **베이스라인**: 67.38% (BoW, 기본 설정)

---

## 📊 실험 결과 요약

| 실험 | 벡터화 방법 | Dev 정확도 | Test 정확도 | 베이스라인 대비 |
|------|------------|-----------|------------|---------------|
| Exp1 | BoW (베이스라인) | — | 67.38% | — |
| Exp2 | BoW + W&B Sweep | — | 68.26% | +0.88%p |
| Exp3 | TF-IDF + W&B Sweep | — | 68.30% | +0.92%p |
| Exp4 | GloVe 300d + W&B Sweep | — | 62.00% | -5.38%p |
| Exp5 | MiniLM 384d + W&B Sweep | — | 66.07% | -1.31%p |
| Exp6 | MPNet 768d + W&B Sweep | — | 66.28% | -1.10%p |
| **Exp3-v8** | **TF-IDF + 전처리 + 핸드크래프트 + Sweep** | **70.05%** | **69.13%** | **+1.75%p 🏆** |

> **최종 제출 모델**: Exp3-v8 (`submit/W5_HW1_PracticalNLP_202200274_고민서.ipynb`)

### Ablation: BoW vs TF-IDF (동일 전처리·핸드크래프트)

| 실험 | 벡터화 | Test 정확도 | 해석 |
|------|--------|------------|------|
| Exp2-v2 | BoW + 전처리 + 핸드크래프트 + Sweep | 68.70% | 동일 피처 엔지니어링을 BoW에 적용 |
| Exp3-v8 | TF-IDF + 전처리 + 핸드크래프트 + Sweep | **69.13%** | TF-IDF가 +0.43%p 우위 |

### 전처리 개선 실험 (Exp3 기반)

| 버전 | 핵심 변경 | Test 정확도 |
|------|-----------|------------|
| Exp3 base | TF-IDF + Sweep | 68.30% |
| v6 | 백틱 처리 + 슬랭 정규화 | 68.36% |
| v7 | 핸드크래프트 피처 concat | 68.61% |
| **v8** | **전처리 + 핸드크래프트 + Sweep** | **69.13% 🏆** |
| v9 | v8 + StandardScaler | 68.76% (역효과) |
| v10 | v8 best config + epoch 100 | 69.13% (v8과 동일, Dev peak Epoch 33) |

---

## 📁 파일 구조

```
├── submit/                              # 최종 제출 파일
│   ├── W5_HW1_PracticalNLP_202200274_고민서.ipynb  # HW1 제출본 (Exp3-v8, Dev 70.05% / Test 69.13% 🏆)
│   ├── W5_HW1_PracticalNLP_202200274_고민서.pdf    # HW1 실험 보고서
│   └── W5_HW2_PracticalNLP_202200274_고민서.ipynb  # HW2 제출본 (Exp3-v8 감성 예측)
├── notebooks/
│   ├── Exp1_BoW_Baseline.ipynb          # 베이스라인 (67.38%)
│   ├── Exp2_BoW_Sweep.ipynb             # BoW + Sweep (68.26%)
│   ├── Exp3_TFIDF_Sweep.ipynb           # TF-IDF + Sweep (68.30%)
│   ├── Exp4_GloVe100_Sweep.ipynb        # GloVe 100d
│   ├── Exp4_GloVe300_Sweep.ipynb        # GloVe 300d (62.00%)
│   ├── Exp5_MiniLM_Sweep.ipynb          # MiniLM 384d (66.07%)
│   ├── Exp5_v2_MiniLM_Fixed.ipynb       # MiniLM (seed 고정)
│   ├── Exp6_MPNet_Sweep.ipynb           # MPNet 768d (66.28%)
│   ├── exp2_improvements/
│   │   └── Exp2_v2_BoW_Enhanced_Sweep.ipynb  # BoW 추가 개선 sweep
│   └── exp3_improvements/
│       ├── Exp3_v6_Preprocessing.ipynb  # 백틱/슬랭 전처리 (68.36%)
│       ├── Exp3_v7_Handcraft.ipynb      # 핸드크래프트 피처 (68.61%)
│       ├── Exp3_v9_Scaled.ipynb         # StandardScaler 시도 (68.76%)
│       └── Exp3_v10_Final.ipynb         # v8 config + epoch 100 (69.13%)
│   # Exp3-v8 최종 학습 노트북 → submit/W5_HW1_...ipynb 로 제출
├── analysis/
│   ├── analyze_data.py                  # 데이터 기초 분석
│   └── analyze_data_deep.py             # 데이터 심층 분석
├── docs/
│   ├── HW1_Report_Draft.md              # 실험 보고서
│   ├── 실험_결과_정리.md                # 상세 실험 기록
│   ├── HW1_Results_Summary.ipynb
│   ├── HW2_AllModels.ipynb
│   └── HW2_PracticalNLP.ipynb
├── README.md
└── .gitignore
```

> ⚠️ 모델 체크포인트(`.pt`), 데이터셋(`data/`), W&B 로그(`wandb/`)는 `.gitignore`로 제외됩니다.

---

## 🔑 핵심 인사이트

### 왜 TF-IDF가 가장 효과적이었나?
이 데이터는 **트위터/SNS 단문** 스타일로, 감성이 특정 키워드에 직접 표현됩니다.
- "yummy", "love", "hate", "sucks" 같은 키워드가 직접적인 신호
- TF-IDF의 IDF 가중치가 이런 감성 키워드를 자동으로 강조
- GloVe/Transformer의 문맥 이해 능력이 이 데이터셋에서는 오히려 과잉

### 전처리가 중요한 이유 (23.7% 데이터 영향)
```python
# 백틱 문제: 7,402개 텍스트에서 축약어 분리 실패
"don`t"  → "don't"  → "do not"   # 백틱 → 어포스트로피 → 분리
"I`m"    → "I'm"    → "I am"
"can`t"  → "can't"  → "cannot"
```

### 핸드크래프트 피처가 효과적인 이유
TF-IDF가 포착 못 하는 신호를 직접 추가:

| 피처 | negative | neutral | positive |
|------|----------|---------|----------|
| 느낌표 평균 | 0.425 | 0.379 | **0.664** |
| 물음표 평균 | 0.145 | **0.201** | 0.084 |

---

## 🛠 실행 방법

### 환경 설정
```bash
pip install torch scikit-learn datasets wandb sentence-transformers gensim
```

### W&B 로그인
```python
import wandb
wandb.login()
```

### 추천 실행 순서
1. `notebooks/Exp1_BoW_Baseline.ipynb` — 베이스라인 확인
2. `notebooks/Exp3_TFIDF_Sweep.ipynb` — 핵심 실험
3. `submit/W5_HW1_PracticalNLP_202200274_고민서.ipynb` — 최고 성능(Exp3-v8) 재현
   - W&B Sweep 없이 검증: 노트북 내 **「교수님 검증용」** 셀 실행 (Dev 70.05% / Test 69.13%)
4. `submit/W5_HW2_PracticalNLP_202200274_고민서.ipynb` — 감성 예측 데모 (체크포인트 `best_model_exp3_v8_final.pt` 필요)

> **Colab GPU 권장**: T4 기준 실험당 약 30~90분 소요

---

## 📈 W&B 실험 기록

프로젝트: [`nlp-hw1`](https://wandb.ai/imeanseo_/nlp-hw1)

---

## 💡 주요 발견

1. **단순한 모델이 복잡한 모델보다 나을 수 있다** — 데이터 특성에 맞는 방법 선택이 핵심
2. **전처리의 중요성** — 백틱 처리 하나만으로도 23.7% 데이터의 토크나이징이 개선됨
3. **Sweep의 한계** — 입력 공간이 바뀌면 기존 best config가 최적이 아닐 수 있음
4. **스케일 불균형 주의** — TF-IDF(0~1)와 다른 스케일의 피처를 concat할 때 정규화 필요
