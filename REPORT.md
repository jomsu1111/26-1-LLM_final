# Stop, Verify, or Explore? 최종 보고서 초안

## 초록

이 프로젝트는 autoregressive LLM의 reasoning task에서 one-step adaptive test-time compute allocation을 실험한다. 각 GSM8K 문제에 대해 LLM이 먼저 initial reasoning trace와 initial answer를 생성하고, 이후 lightweight controller가 STOP, VERIFY, SC-3 중 정확히 하나의 action을 선택한다. STOP은 initial answer를 그대로 사용하고, VERIFY는 모델이 문제를 다시 풀어 기존 답을 검토/수정하게 하며, SC-3는 추가 독립 풀이 2개를 생성해 총 3개 답변으로 voting을 수행한다.

목표는 모든 문제에 추가 compute를 사용하는 대신, 현재 reasoning state를 보고 extra compute가 가치 있을 때만 사용하는 것이다. `Qwen/Qwen2.5-1.5B-Instruct`와 `Qwen/Qwen2.5-3B-Instruct`를 사용한 실험에서, extra compute는 일부 문제에서 명확한 accuracy gain을 만들었다. Oracle routing은 Always STOP보다 크게 높은 성능을 보였고, learned controller도 calibration과 model choice에 따라 그 이득의 일부를 회복했다.

## 문제 설정

본 프로젝트의 전체 구조는 다음과 같다.

```text
Question -> Initial reasoning -> STOP / VERIFY / SC-3 -> Final answer
```

이 설정은 multi-step MDP나 reinforcement learning이 아니다. Initial answer 생성 이후 한 번만 routing decision을 내리는 one-step post-hoc routing 문제다.

Action은 다음 세 가지다.

| Action | 설명 | 추가 compute |
|---|---|---|
| STOP | initial answer를 그대로 final answer로 사용 | 없음 |
| VERIFY | 문제를 다시 풀고 이전 reasoning/answer와 비교하여 답을 유지 또는 수정 | generation 1회 |
| SC-3 | 독립 풀이 2개를 추가 생성하고 총 3개 답으로 plurality vote | generation 2회 |

Action utility는 다음과 같이 정의한다.

```text
utility(action) = correct(action) - lambda * normalized_cost(action)
```

기본 cost normalizer는 average SC-3 total token cost이며, 주요 평가는 `lambda=0.1`을 사용했다.

## 구현

코드베이스는 다음 기능을 구현한다.

- GSM8K dataset loading 및 deterministic shuffle
- HuggingFace Transformers 기반 generation
- JSONL rollout 저장 및 resume/cache
- regex 기반 final answer extraction
- numeric correctness checking
- STOP / VERIFY / SC-3 rollout
- oracle utility analysis
- train / validation / test split
- fixed baselines
- learned controller 학습
- evaluation CSV 생성
- matplotlib plot 생성
- model comparison script

구현된 controller variant는 다음과 같다.

1. LogisticRegression 기반 oracle-action classifier
2. RandomForestClassifier 기반 oracle-action classifier
3. Validation split 기반 extra-compute threshold calibration
4. Action별 correctness와 cost를 따로 예측하는 value predictor

## 실험 설정

Dataset:

- GSM8K test split
- `seed=7`로 deterministic shuffle

Models:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

Generation settings:

```text
temperature = 0.8
max_new_tokens_initial = 192
max_new_tokens_verify = 256
max_new_tokens_sc3 = 192
```

주요 실험:

- Qwen2.5-1.5B, 300 examples
- 공정한 model-size 비교를 위한 Qwen2.5-1.5B, 100 examples
- Qwen2.5-3B, 100 examples

## 결과

### Qwen2.5-1.5B, 300 examples

| Method | Accuracy | Avg total tokens | Utility |
|---|---:|---:|---:|
| Always STOP | 0.253 | 299.68 | 0.222 |
| Always VERIFY | 0.370 | 916.16 | 0.275 |
| Always SC-3 | 0.287 | 966.53 | 0.187 |
| Oracle Router | 0.473 | 435.65 | 0.428 |
| Logistic calibrated | 0.373 | 624.21 | 0.309 |
| Random forest calibrated | 0.430 | 484.83 | 0.380 |
| Value predictor | 0.400 | 778.28 | 0.319 |

Oracle Router는 Always STOP 대비 accuracy를 `0.253`에서 `0.473`으로 끌어올렸다. 동시에 Always VERIFY보다 훨씬 적은 token cost를 사용했다. Learned controller 중에서는 calibrated random forest가 가장 좋았으며, accuracy `0.430`, utility `0.380`을 달성했다.

### Qwen2.5-3B, 100 examples

| Method | Accuracy | Avg total tokens | Utility |
|---|---:|---:|---:|
| Always STOP | 0.320 | 299.87 | 0.289 |
| Always VERIFY | 0.540 | 912.76 | 0.446 |
| Always SC-3 | 0.340 | 969.57 | 0.240 |
| Oracle Router | 0.600 | 472.56 | 0.551 |
| Logistic calibrated | 0.350 | 350.99 | 0.314 |
| Random forest calibrated | 0.500 | 482.26 | 0.450 |
| Value predictor | 0.540 | 646.98 | 0.473 |

3B 모델에서는 VERIFY가 특히 강했다. Always VERIFY accuracy가 `0.540`까지 상승했고, Oracle Router는 `0.600`을 달성했다. Value predictor는 Always VERIFY와 같은 accuracy를 보이면서 더 적은 token cost를 사용했다.

### 모델 비교

공정한 model-size comparison을 위해 1.5B와 3B를 모두 100 shuffled GSM8K examples에서 비교했다.

| Model | STOP acc | VERIFY acc | SC-3 acc | Oracle acc | Best controller acc |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.290 | 0.440 | 0.320 | 0.530 | 0.470 |
| Qwen2.5-3B | 0.320 | 0.540 | 0.340 | 0.600 | 0.540 |

3B 모델은 STOP, VERIFY, Oracle, best learned controller accuracy 모두에서 1.5B보다 높은 성능을 보였다. 특히 VERIFY accuracy가 `0.440 -> 0.540`으로 크게 증가했다. 이는 더 큰 모델이 verification-style compute를 더 잘 활용할 수 있음을 시사한다.

## 논의

첫 번째 핵심 결과는 value-of-compute signal이 존재한다는 점이다. Extra compute는 모든 문제에서 필요한 것은 아니지만, 일부 문제에서는 명확한 성능 향상을 만든다. Oracle Router는 Always STOP보다 일관되게 높은 accuracy를 보이며, Always VERIFY보다 훨씬 적은 compute로 좋은 utility를 달성한다.

두 번째로, 이 실험에서는 VERIFY가 SC-3보다 강했다. SC-3는 추가 generation 2회를 사용하지만, Qwen2.5 instruct model과 GSM8K 설정에서는 단순 self-consistency보다 re-solve and verify 방식이 더 효과적이었다.

세 번째로, learned controller는 부분적으로 작동하지만 Oracle과의 gap이 남아 있다. 1.5B 300-example 실험에서는 calibrated random forest가 가장 좋은 controller였고, 3B 100-example 실험에서는 value predictor가 가장 좋았다. 이는 controller choice와 calibration이 routing 성능에 큰 영향을 준다는 것을 보여준다.

마지막으로, multiple models 실험에서도 같은 현상이 유지되었다. 3B에서도 Oracle Router는 Always STOP보다 훨씬 높았고, 1.5B보다 VERIFY와 Oracle 성능이 모두 개선되었다. 따라서 value-of-compute routing 현상은 특정 모델 하나에만 국한된 결과가 아니다.

## 한계

- Dataset은 GSM8K만 사용했다.
- 3B 실험은 100 examples이며, 1.5B의 가장 안정적인 main analysis는 300 examples다.
- Oracle action label은 imbalanced하다. 대부분의 example은 utility 기준 STOP이 최적이다.
- Entropy, token log-probability, self-reported confidence feature는 아직 구현하지 않았다.
- SC-3는 추가 sample 2개만 사용한다. SC-5 이상에서는 결과가 달라질 수 있다.

## 재현 방법

### 1.5B main rollout

```bash
python3 scripts/run_rollouts.py \
  --dataset gsm8k \
  --model_name Qwen/Qwen2.5-1.5B-Instruct \
  --max_examples 300 \
  --shuffle \
  --seed 7 \
  --output_path outputs/gsm8k_rollouts_qwen300_shuffle_v2.jsonl \
  --overwrite \
  --max_new_tokens_initial 192 \
  --max_new_tokens_verify 256 \
  --max_new_tokens_sc3 192 \
  --temperature 0.8
```

### 3B model comparison rollout

```bash
python3 scripts/run_rollouts.py \
  --dataset gsm8k \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --max_examples 100 \
  --shuffle \
  --seed 7 \
  --output_path outputs/gsm8k_rollouts_qwen25_3b_100_shuffle_v2.jsonl \
  --overwrite \
  --max_new_tokens_initial 192 \
  --max_new_tokens_verify 256 \
  --max_new_tokens_sc3 192 \
  --temperature 0.8
```

### Model comparison

```bash
python3 scripts/compare_models.py \
  --model_eval qwen2.5-1.5b-100 outputs/eval_qwen100_shuffle_v2_all \
  --model_eval qwen2.5-3b-100 outputs/eval_qwen25_3b_100_v2_compare_all \
  --output_csv outputs/model_comparison_qwen15b100_vs_qwen3b100.csv \
  --output_plot outputs/model_comparison_qwen15b100_vs_qwen3b100.png
```

## 결론

본 프로젝트는 autoregressive LLM reasoning에서 adaptive test-time compute routing이 의미 있는 문제임을 보였다. Extra compute는 일부 문제에서 큰 accuracy gain을 만들지만, 모든 문제에 무작정 적용하면 token cost가 커진다. Oracle routing은 가능한 성능 상한을 보여주며, learned state-aware controller는 그 이득의 일부를 회복한다. 또한 Qwen2.5-1.5B와 Qwen2.5-3B 모두에서 value-of-compute signal이 관찰되었고, 특히 더 큰 3B 모델은 VERIFY action을 더 효과적으로 활용했다.

