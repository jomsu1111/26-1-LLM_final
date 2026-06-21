# Stop, Verify, or Explore?

**State-Aware Value-of-Compute Routing for Autoregressive LLM Reasoning**

이 프로젝트는 autoregressive LLM의 reasoning task에서 **한 번의 post-hoc routing decision**으로 test-time compute를 적응적으로 배분하는 실험 프레임워크입니다.

핵심 질문은 다음과 같습니다.

> 모든 문제에 추가 compute를 쓰는 대신, 현재 reasoning state를 보고 STOP / VERIFY / SC-3 중 하나를 고르면 accuracy-compute trade-off를 개선할 수 있는가?

전체 파이프라인은 아래와 같습니다.

```text
Question
  -> Initial reasoning + initial answer
  -> Controller chooses one action: STOP / VERIFY / SC-3
  -> Final answer
```

이 프로젝트는 multi-step RL이나 MDP가 아닙니다. Initial answer가 생성된 뒤 **정확히 한 번만 routing**하는 one-step setting입니다.

## Action 정의

각 문제에서 initial reasoning을 먼저 생성한 뒤, 세 action을 모두 rollout합니다.

| Action | 설명 | 추가 compute |
|---|---|---|
| STOP | initial answer를 그대로 final answer로 사용 | 없음 |
| VERIFY | LLM이 문제를 다시 풀고 기존 풀이와 비교하여 답을 유지/수정 | verification generation 1회 |
| SC-3 | independent reasoning trace 2개를 추가 생성하고 총 3개 답으로 plurality vote | generation 2회 |

SC-3 voting은 normalized answer 기준 plurality vote를 사용하며, 동률이면 initial answer를 선택합니다.

## Utility

각 action의 utility는 다음과 같이 정의합니다.

```text
utility(action) = correct(action) - lambda * normalized_cost(action)
```

여기서 `correct(action)`은 정답 여부, `normalized_cost`는 action total token cost를 normalizer로 나눈 값입니다. 기본 normalizer는 average SC-3 total token cost입니다.

Oracle router는 각 example에서 utility가 가장 큰 action을 선택합니다.

## 구현 범위

현재 구현된 기능:

- GSM8K dataset loader
- HuggingFace Transformers 기반 generation
- GPU 자동 사용
- seed 고정
- JSONL rollout 저장 및 resume/cache
- answer extraction 및 numeric correctness checking
- STOP / VERIFY / SC-3 rollout
- oracle analysis
- train / validation / test split
- baselines
  - Always STOP
  - Always VERIFY
  - Always SC-3
  - Random Router
  - Length Threshold Router
  - Oracle Router
- learned controllers
  - LogisticRegression oracle-action classifier
  - RandomForest oracle-action classifier
  - validation calibration
  - value predictor variant
- evaluation CSV 생성
- matplotlib plot 생성

아직 구현하지 않은 기능:

- entropy / token log-probability features
- self-reported confidence feature
- MATH500 등 cross-dataset evaluation

## Repository 구조

```text
configs/
  default.yaml
scripts/
  run_rollouts.py
  build_oracle.py
  train_controller.py
  evaluate_methods.py
  plot_results.py
  compare_models.py
src/
  datasets.py
  model_utils.py
  prompts.py
  answer_extraction.py
  generation.py
  features.py
  cost.py
  oracle.py
  controllers.py
  evaluation.py
  splits.py
  utils.py
outputs/
```

## 설치

```bash
pip install -r requirements.txt
```

## Smoke Test

모델 다운로드 없이 빠르게 pipeline만 확인하려면 mock model을 사용합니다.

```bash
python3 scripts/run_rollouts.py \
  --dataset gsm8k \
  --max_examples 10 \
  --mock_model \
  --output_path outputs/gsm8k_rollouts_mock.jsonl \
  --overwrite

python3 scripts/build_oracle.py \
  --rollout_path outputs/gsm8k_rollouts_mock.jsonl \
  --lambda_values 0.0 0.05 0.1 0.2 \
  --output_dir outputs/oracle_mock
```

Qwen2.5-3B-Instruct를 실제로 사용하는 10-example smoke test는 다음과 같습니다. 기존 pipeline을 그대로 사용하고 output 이름만 `qwen25_3b`로 분리합니다.

```bash
python3 scripts/run_rollouts.py \
  --dataset gsm8k \
  --model_name Qwen/Qwen2.5-3B-Instruct \
  --max_examples 10 \
  --shuffle \
  --seed 7 \
  --output_path outputs/gsm8k_rollouts_qwen25_3b_10_shuffle_v2.jsonl \
  --overwrite \
  --max_new_tokens_initial 192 \
  --max_new_tokens_verify 256 \
  --max_new_tokens_sc3 192 \
  --temperature 0.8

python3 scripts/build_oracle.py \
  --rollout_path outputs/gsm8k_rollouts_qwen25_3b_10_shuffle_v2.jsonl \
  --lambda_values 0.0 0.05 0.1 0.2 \
  --output_dir outputs/oracle_qwen25_3b_10_shuffle_v2
```

## 실험 계획

실험은 다음 순서로 진행했습니다.

1. **Small rollout sanity check**
   - `Qwen/Qwen2.5-1.5B-Instruct`
   - GSM8K 20 examples
   - 목적: VERIFY / SC-3 action이 실제로 initial answer를 개선하는지 확인

2. **Prompt 및 sampling 개선**
   - VERIFY prompt를 단순 검토가 아니라 “문제를 처음부터 다시 풀고 기존 답과 비교”하도록 수정
   - SC-3 prompt에 fresh approach / different decomposition instruction 추가
   - dataset `--shuffle` 추가
   - temperature를 `0.8`로 설정

3. **qwen100 rollout**
   - GSM8K shuffled 100 examples
   - 목적: extra compute가 유의미한 regime인지 확인

4. **qwen300 rollout**
   - GSM8K shuffled 300 examples
   - 목적: 더 안정적인 oracle/controller 비교

5. **Controller 비교**
   - calibrated logistic classifier
   - calibrated random forest classifier
   - value predictor

## 주요 실험 커맨드

### qwen300 rollout

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

### Qwen2.5-3B 100-example model comparison experiment

가점 조건인 multiple models 비교를 위해 Qwen2.5-3B-Instruct도 같은 GSM8K shuffled setting에서 실행할 수 있습니다.

Rollout:

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

Oracle:

```bash
python3 scripts/build_oracle.py \
  --rollout_path outputs/gsm8k_rollouts_qwen25_3b_100_shuffle_v2.jsonl \
  --lambda_values 0.0 0.05 0.1 0.2 \
  --output_dir outputs/oracle_qwen25_3b_100_shuffle_v2
```

Controllers:

```bash
python3 scripts/train_controller.py \
  --rollout_path outputs/gsm8k_rollouts_qwen25_3b_100_shuffle_v2.jsonl \
  --lambda_value 0.1 \
  --controller_type state_aware \
  --classifier_type logistic \
  --controller_variant oracle_classifier \
  --calibrate \
  --output_dir outputs/controllers_qwen25_3b_100_v2_compare \
  --seed 7

python3 scripts/train_controller.py \
  --rollout_path outputs/gsm8k_rollouts_qwen25_3b_100_shuffle_v2.jsonl \
  --lambda_value 0.1 \
  --controller_type state_aware \
  --classifier_type random_forest \
  --controller_variant oracle_classifier \
  --calibrate \
  --output_dir outputs/controllers_qwen25_3b_100_v2_compare \
  --seed 7

python3 scripts/train_controller.py \
  --rollout_path outputs/gsm8k_rollouts_qwen25_3b_100_shuffle_v2.jsonl \
  --lambda_value 0.1 \
  --controller_type state_aware \
  --controller_variant value_predictor \
  --output_dir outputs/controllers_qwen25_3b_100_v2_compare \
  --seed 7
```

Evaluation:

```bash
python3 scripts/evaluate_methods.py \
  --rollout_path outputs/gsm8k_rollouts_qwen25_3b_100_shuffle_v2.jsonl \
  --controller_path outputs/controllers_qwen25_3b_100_v2_compare/state_aware_logistic_calibrated_lambda_0.1.pkl \
  --controller_path outputs/controllers_qwen25_3b_100_v2_compare/state_aware_random_forest_calibrated_lambda_0.1.pkl \
  --controller_path outputs/controllers_qwen25_3b_100_v2_compare/state_aware_value_predictor_lambda_0.1.pkl \
  --lambda_value 0.1 \
  --output_dir outputs/eval_qwen25_3b_100_v2_compare_all \
  --eval_split all \
  --seed 7
```

Plot:

```bash
python3 scripts/plot_results.py \
  --eval_dir outputs/eval_qwen25_3b_100_v2_compare_all \
  --oracle_dir outputs/oracle_qwen25_3b_100_shuffle_v2 \
  --output_dir outputs/plots_qwen25_3b_100_v2_compare_all
```

### Oracle analysis

```bash
python3 scripts/build_oracle.py \
  --rollout_path outputs/gsm8k_rollouts_qwen300_shuffle_v2.jsonl \
  --lambda_values 0.0 0.05 0.1 0.2 \
  --output_dir outputs/oracle_qwen300_shuffle_v2
```

### Controller 학습

Calibrated logistic classifier:

```bash
python3 scripts/train_controller.py \
  --rollout_path outputs/gsm8k_rollouts_qwen300_shuffle_v2.jsonl \
  --lambda_value 0.1 \
  --controller_type state_aware \
  --classifier_type logistic \
  --controller_variant oracle_classifier \
  --calibrate \
  --output_dir outputs/controllers_qwen300_v2_compare \
  --seed 7
```

Calibrated random forest classifier:

```bash
python3 scripts/train_controller.py \
  --rollout_path outputs/gsm8k_rollouts_qwen300_shuffle_v2.jsonl \
  --lambda_value 0.1 \
  --controller_type state_aware \
  --classifier_type random_forest \
  --controller_variant oracle_classifier \
  --calibrate \
  --output_dir outputs/controllers_qwen300_v2_compare \
  --seed 7
```

Value predictor:

```bash
python3 scripts/train_controller.py \
  --rollout_path outputs/gsm8k_rollouts_qwen300_shuffle_v2.jsonl \
  --lambda_value 0.1 \
  --controller_type state_aware \
  --controller_variant value_predictor \
  --output_dir outputs/controllers_qwen300_v2_compare \
  --seed 7
```

### Evaluation

```bash
python3 scripts/evaluate_methods.py \
  --rollout_path outputs/gsm8k_rollouts_qwen300_shuffle_v2.jsonl \
  --controller_path outputs/controllers_qwen300_v2_compare/state_aware_logistic_calibrated_lambda_0.1.pkl \
  --controller_path outputs/controllers_qwen300_v2_compare/state_aware_random_forest_calibrated_lambda_0.1.pkl \
  --controller_path outputs/controllers_qwen300_v2_compare/state_aware_value_predictor_lambda_0.1.pkl \
  --lambda_value 0.1 \
  --output_dir outputs/eval_qwen300_v2_compare_test \
  --eval_split test \
  --seed 7
```

전체 300개 sanity analysis:

```bash
python3 scripts/evaluate_methods.py \
  --rollout_path outputs/gsm8k_rollouts_qwen300_shuffle_v2.jsonl \
  --controller_path outputs/controllers_qwen300_v2_compare/state_aware_logistic_calibrated_lambda_0.1.pkl \
  --controller_path outputs/controllers_qwen300_v2_compare/state_aware_random_forest_calibrated_lambda_0.1.pkl \
  --controller_path outputs/controllers_qwen300_v2_compare/state_aware_value_predictor_lambda_0.1.pkl \
  --lambda_value 0.1 \
  --output_dir outputs/eval_qwen300_v2_compare_all \
  --eval_split all \
  --seed 7
```

### Plot

```bash
python3 scripts/plot_results.py \
  --eval_dir outputs/eval_qwen300_v2_compare_all \
  --oracle_dir outputs/oracle_qwen300_shuffle_v2 \
  --output_dir outputs/plots_qwen300_v2_compare_all
```

## 주요 결과

아래 결과는 `Qwen/Qwen2.5-1.5B-Instruct`, GSM8K shuffled 300 examples, `lambda=0.1` 기준입니다.

### 전체 300 examples 기준

| Method | Accuracy | Avg total tokens | Utility | STOP | VERIFY | SC-3 |
|---|---:|---:|---:|---:|---:|---:|
| Always STOP | 0.253 | 299.68 | 0.222 | 300 | 0 | 0 |
| Always VERIFY | 0.370 | 916.16 | 0.275 | 0 | 300 | 0 |
| Always SC-3 | 0.287 | 966.53 | 0.187 | 0 | 0 | 300 |
| Random Router | 0.330 | 757.43 | 0.252 | 86 | 109 | 105 |
| Length Threshold Router | 0.267 | 651.80 | 0.199 | 150 | 0 | 150 |
| Oracle Router | 0.473 | 435.65 | 0.428 | 234 | 57 | 9 |
| Logistic calibrated | 0.373 | 624.21 | 0.309 | 143 | 153 | 4 |
| Random forest calibrated | 0.430 | 484.83 | 0.380 | 210 | 86 | 4 |
| Value predictor | 0.400 | 778.28 | 0.319 | 73 | 215 | 12 |

### Held-out test split 기준

Test split은 47 examples입니다.

| Method | Accuracy | Avg total tokens | Utility | STOP | VERIFY | SC-3 |
|---|---:|---:|---:|---:|---:|---:|
| Always STOP | 0.191 | 303.89 | 0.160 | 47 | 0 | 0 |
| Always VERIFY | 0.298 | 922.11 | 0.202 | 0 | 47 | 0 |
| Always SC-3 | 0.277 | 974.49 | 0.176 | 0 | 0 | 47 |
| Oracle Router | 0.404 | 431.53 | 0.360 | 37 | 6 | 4 |
| Logistic calibrated | 0.298 | 690.77 | 0.226 | 18 | 28 | 1 |
| Random forest calibrated | 0.213 | 473.96 | 0.164 | 34 | 13 | 0 |
| Value predictor | 0.298 | 831.85 | 0.212 | 8 | 39 | 0 |

### Qwen2.5-3B 100 examples 기준

Multiple models 조건을 위해 같은 GSM8K shuffled setting에서 `Qwen/Qwen2.5-3B-Instruct`도 100 examples로 실행했습니다.

| Method | Accuracy | Avg total tokens | Utility | STOP | VERIFY | SC-3 |
|---|---:|---:|---:|---:|---:|---:|
| Always STOP | 0.320 | 299.87 | 0.289 | 100 | 0 | 0 |
| Always VERIFY | 0.540 | 912.76 | 0.446 | 0 | 100 | 0 |
| Always SC-3 | 0.340 | 969.57 | 0.240 | 0 | 0 | 100 |
| Oracle Router | 0.600 | 472.56 | 0.551 | 72 | 25 | 3 |
| Logistic calibrated | 0.350 | 350.99 | 0.314 | 92 | 7 | 1 |
| Random forest calibrated | 0.500 | 482.26 | 0.450 | 71 | 28 | 1 |
| Value predictor | 0.540 | 646.98 | 0.473 | 45 | 53 | 2 |

3B test split은 14 examples라 variance가 크지만, sanity check로는 Oracle accuracy `0.643`, value predictor accuracy `0.571`이 나왔습니다.

### Model comparison

아래 표는 공정한 model-size 비교를 위해 1.5B와 3B를 모두 100 examples 기준으로 비교한 결과입니다. 1.5B의 더 안정적인 main analysis는 위의 300-example 결과를 사용했습니다.

| Model | STOP acc | VERIFY acc | SC-3 acc | Oracle acc | Best controller acc | Best controller |
|---|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B, 100 ex. | 0.290 | 0.440 | 0.320 | 0.530 | 0.470 | state-aware controller |
| Qwen2.5-3B, 100 ex. | 0.320 | 0.540 | 0.340 | 0.600 | 0.540 | value predictor |

결과적으로 3B는 1.5B보다 STOP, VERIFY, Oracle, best learned controller 모두에서 더 높은 accuracy를 보였습니다. 특히 VERIFY 성능이 `0.440 -> 0.540`으로 크게 증가했고, Oracle도 `0.530 -> 0.600`으로 상승했습니다.

## 결과 해석

### 1. Extra compute signal은 존재한다

전체 300 examples 기준으로 Always STOP accuracy는 `0.253`이지만 Oracle Router는 `0.473`입니다. 즉 어떤 문제에서는 VERIFY 또는 SC-3가 initial answer를 실제로 개선합니다.

Oracle action distribution은 다음과 같습니다.

```text
STOP:   234 / 300
VERIFY:  57 / 300
SC-3:     9 / 300
```

이는 “항상 추가 compute를 쓰는 것”이 아니라, **필요한 일부 문제에서만 추가 compute를 쓰는 selective routing**이 중요하다는 것을 보여줍니다.

### 2. VERIFY는 SC-3보다 강했다

이 설정에서는 Always VERIFY accuracy가 `0.370`, Always SC-3 accuracy가 `0.287`입니다. SC-3는 추가 generation 2회를 사용하지만, Qwen2.5-1.5B-Instruct와 GSM8K 설정에서는 VERIFY보다 약했습니다.

### 3. Learned controller는 부분적으로 작동하지만 Oracle gap이 남아 있다

전체 300 examples 기준 calibrated random forest는 accuracy `0.430`, utility `0.380`으로 learned controller 중 가장 좋았습니다. Oracle utility `0.428`과는 아직 차이가 있지만, Always VERIFY보다 더 적은 token cost로 더 높은 utility를 냅니다.

하지만 held-out test split에서는 controller 성능이 아직 불안정합니다. Test examples가 47개로 크지 않고, oracle label이 imbalanced하기 때문입니다.

### 4. Calibration은 중요하다

초기 classifier는 VERIFY / SC-3를 과하게 고르는 경향이 있었습니다. Validation split에서 extra-compute threshold를 calibration하면 token cost를 줄이고 utility를 개선할 수 있습니다.

예를 들어 random forest calibration은 validation에서 다음 threshold를 선택했습니다.

```text
VERIFY threshold: 0.50
SC-3 threshold:  0.60
```

### 5. Multiple models에서도 같은 현상이 유지된다

3B 모델에서도 Oracle Router는 Always STOP보다 훨씬 높은 accuracy를 보였습니다. 이는 value-of-compute signal이 특정 작은 모델의 우연한 현상이 아니라, 모델 크기를 바꿔도 관찰되는 실험 패턴임을 시사합니다.

동시에 3B에서는 VERIFY가 특히 강해졌습니다. 이는 더 큰 모델이 자기 검증 또는 재풀이 action을 더 잘 활용할 수 있음을 보여줍니다.

## 생성되는 파일

`run_rollouts.py`:

- rollout JSONL
- 각 row에 initial generation, STOP, VERIFY, SC-3, features, error field 포함

`build_oracle.py`:

- `oracle_analysis.csv`
- `oracle_per_example.csv`

`train_controller.py`:

- controller `.pkl`
- metadata `.json`

`evaluate_methods.py`:

- `results_summary.csv`
- `per_example_predictions.csv`
- `action_distribution.csv`
- `transition_table.csv`

`plot_results.py`:

- `accuracy_vs_token_cost.png/pdf`
- `utility_vs_lambda.png/pdf`
- `oracle_action_distribution.png/pdf`
- `controller_action_distribution.png/pdf`
- `transition_table.png/pdf`

`compare_models.py`:

- 여러 model의 `results_summary.csv`를 합친 model comparison CSV
- model별 accuracy-token cost comparison plot

예시:

```bash
python3 scripts/compare_models.py \
  --model_eval qwen2.5-1.5b outputs/eval_qwen300_v2_compare_all \
  --model_eval qwen2.5-3b outputs/eval_qwen25_3b_100_v2_compare_all \
  --output_csv outputs/model_comparison_qwen15b_vs_qwen3b.csv \
  --output_plot outputs/model_comparison_qwen15b_vs_qwen3b.png
```

공정한 100-example model-size comparison:

```bash
python3 scripts/compare_models.py \
  --model_eval qwen2.5-1.5b-100 outputs/eval_qwen100_shuffle_v2_all \
  --model_eval qwen2.5-3b-100 outputs/eval_qwen25_3b_100_v2_compare_all \
  --output_csv outputs/model_comparison_qwen15b100_vs_qwen3b100.csv \
  --output_plot outputs/model_comparison_qwen15b100_vs_qwen3b100.png
```

## Assumptions

- GSM8K gold answer는 `####` 뒤 문자열에서 추출합니다.
- Numeric answer 비교에서는 comma, dollar sign, integer/float formatting을 normalize합니다.
- Final answer는 `Final answer: <answer>` regex로 추출합니다.
- SC-3 tie-break는 initial answer입니다.
- Real model token cost는 tokenizer count 기반입니다.
- `--shuffle`은 seed에 따라 deterministic하게 dataset을 섞은 뒤 `--max_examples`를 적용합니다.

## 결론

이 프로젝트의 핵심 결론은 다음과 같습니다.

> Extra test-time compute는 모든 문제에 필요한 것은 아니지만, 일부 문제에서는 명확한 accuracy gain을 만든다. Oracle routing은 STOP 대비 큰 개선을 보이며, learned controller도 calibration과 model choice에 따라 일부 trade-off 개선을 달성한다. 따라서 state-aware value-of-compute routing 문제는 실험적으로 의미가 있으며, 남은 병목은 더 좋은 state feature와 controller calibration이다.
