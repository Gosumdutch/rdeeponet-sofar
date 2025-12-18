## R-DeepONet 데이터 팩토리 + 모델 기술/물리 통합 리포트 (KOR)

이 문서는 본 프로젝트의 물리적 가정(해양음향)과 엔지니어링 구현(데이터 생성/파서/시각화/훈련)을 한 곳에 정리합니다. 재현성, 디버깅, 확장 학습을 위한 기준 문서입니다.

---

## 목적
- 자동화된 데이터 팩토리: Ray Map(X)과 TL Field(Y)를 안정적으로 대량 생성, 실패/재개 지원.
- 정답(물리) 엔진: BELLHOP(광선), KRAKEN(정상모드), SCOOTER(범위의존; 차후).
- 학습 준비 포맷: HDF5(학습), PNG(시각 QA), 256×256 격자 정합.
- R-DeepONet: Ray Map + 조건(주파수/소스깊이) → TL Field 예측.

---

## 시나리오 의도
- JASA_Benchmark: 심해 평탄 해저, 정답=KRAKEN(정상모드), GEBCO 미사용.
- Defense_Wideband: 실제 GEBCO(제주 남방), 정답=SCOOTER(차후 1000셋).
- Munk_Profile_Benchmark: 디버깅/검증용 표준 Munk SSP 심해 환경.

---

## 좌표계/단위
- 거리 r[m], 깊이 z[m], 주파수 f[Hz].
- TL(dB) = −20·log10(|p|). 훈련 타일은 [40,120] dB → [0,1]로 정규화.

---

## 물리 모델 개요

### BELLHOP (광선추적)
- 기하음향 기반. 소스에서 방사된 빔(각도 분포)이 환경(SSP, 경계면)과 상호작용하며 경로를 형성.
- 출력: `.ray`(각 빔의 경로 좌표). 본 프로젝트에선 Ray density를 256×256 이미지로 가공.
- 주의: GPU bellhopcuda는 특정 환경에서 좌표가 기록되지 않는 사례가 있어 CPU `bellhop.exe` 사용으로 고정.

### KRAKEN (정상모드)
- 수층-해저로 이루어진 파도관(waveguide)의 정상모드 해를 구해 음압장을 합성.
- 모드 고유치(수평 파수)와 고유함수(깊이 모드)를 구한 뒤, 소스·수신기 조건으로 복소 음압 p(r,z,f) 합산.
- 출력: `.mod`(모드), `FIELD` 후처리로 `.shd`(복소 sound field). 본 프로젝트는 `.shd`를 파싱하여 TL로 변환.
- 본 심해 평탄 시나리오는 “유체 반공간(bottom)” 가정으로 충분하고, 전단파는 0으로 두어 탄성효과를 배제.

### SCOOTER (범위의존)
- 거리 방향으로 환경이 변하는 경우(실제 지형/성층)에 유리. 본 프로젝트의 대규모(1000장) 실측형은 차후 단계에서 사용.

---

## 환경(ENV) 선택 및 핵심 물리 파라미터

### SSP (Sound Speed Profile)
- 디버깅/검증: Munk SSP 사용(문헌/AT Matlab 예제와 동일). 수온·압력 기원 음속 최소점(채널)로 인해 도파관 간섭 무늬 재현.
- BELLHOP 입력은 스플라인 계열 옵션('S' 포함) 권장(부드러운 SSP 근사 및 수치 안정성 향상).

### 경계면/해저(Flat, Fluid Half-space)
- KRAKEN 심해 평탄 해저: bottom cp≈1600 m/s, cs=0 m/s(유체 가정), ρ≈1.8 g/cm³, 감쇠 ap≈0.1–0.8 dB/λ.
- Thorp 흡수 비활성화('NVW'), 수치적 과감쇠 방지 위해 ap를 낮춰 TL 포화(120 dB) 현상 해소.
- GEBCO 지형은 KRAKEN 시나리오에서 사용하지 않음(평탄 가정 유지). GEBCO는 SCOOTER에만 적용.

### 모드 대역(cLow/cHigh)
- 수층 내 음속 범위를 감싸도록 설정(예: 1500–1600 m/s).
- 대역이 협소/비현실적이면 모드 미발견, 과다 감쇠 등으로 TL이 단색화될 수 있음.

### 빔(광선) 설정
- 예: −30°~+30°, N=361 등. Ray map 가독성/학습 유용성 균형을 위해 조절.

### 필드 격자(.flp)
- FIELD 입력을 통해 0–100 km × 0–5000 m를 256×256으로 고정 생성(학습 타겟 크기와 일치).

---

## 데이터 팩토리 구현(엔지니어링)

### 주요 모듈
- `env_generator.py`: BELLHOP/KRAKEN/SCOOTER용 ENV/BTY/FLP 생성. KRAKEN은 Munk 템플릿에 맞춤.
- `main_factory.py`: 실행 오케스트레이션. 실행 성공=산출물 존재로 판정. `--resume`, `--start_idx` 지원.
- `output_parser.py`:
  - `.ray` 파싱: 헤더 → 빔별 좌표 블록 신뢰성 있게 파싱.
  - `.shd` 파싱: Matlab `read_shd_bin.m` 방식 모사. Fortran 레코드 마커와 무관하게 절대 오프셋으로 읽기.
  - 단위 정합: `rarray` km→m 변환. NaN/Inf 처리. 보간은 기본 비활성(256×256 직접 사용).

### 시각화
- TL Combined: Matlab `plotshd.m` 스타일 통계 기반 자동 `caxis`.
- TL/Ray Tiles(학습용): TL=[40,120] dB 고정, Ray는 백그라운드 억제+감마 보정.

### 출력 구조
- H5: `R-DeepONet_Data/data/h5/*.h5` → `X`(ray, 256×256), `Y`(TL, 256×256), `metadata`.
- 학습 타일: `R-DeepONet_Data/data/images/*_{ray|tl}.png`
- 통합 QA: `R-DeepONet_Data/data/check/*_combined.png`

### 자주 겪은 이슈 → 해결
- BELLHOP 실패: `'A~'` 오타 → `'A'` 수정, CPU 버전 사용, 성공 판정 로직 변경.
- KRAKEN 모드 실패/포화: Munk 템플릿 엄수, `cLow/cHigh`/bottom 감쇠 수정, GEBCO 비활성화.
- `.shd` 파싱 실패: `FortranFile` 폐기, 수동 이진 파서 도입(절대 오프셋).
- 단위 불일치: km↔m 정합.
- 윈도우 인코딩: 이모지 제거, 로그 조정.

---

## 샘플 생성 시 KRAKEN의 물리적 의미(요약)
- 정상모드 해석은 수직 방향 고유함수(깊이 모드)와 수평 파수(전파 상수)를 계산합니다.
- 음압장은 각 모드의 기여를 전 범위 r에 대해 합성해 산출됩니다(거리 감쇠/간섭 포함).
- 해저는 유체 반공간(전단 0) 가정으로 모드의 경계 조건을 결정하며, 감쇠(ap)가 커지면 높은 거리에서 에너지가 빠르게 소실됩니다.
- `.mod`는 모드 정보, `.shd`는 소스/수신기 격자에서의 복소 음압. TL(dB) 변환 후 256×256 타겟으로 사용합니다.

---

## R-DeepONet v2 (모델/훈련)

### 입력/타겟
- Branch 이미지 입력: Ray Map `X` ∈ [0,1], shape [1,256,256]
- 스칼라 조건: 주파수(log10 정규화), 소스깊이(zs/5000)
- 타겟: TL(dB) → [40,120] dB를 [0,1]로 선형 정규화

### 아키텍처(셰이크다운에서 실제 사용값)
- Branch CNN: ResNet-18(첫 conv 1채널로 교체), GAP → 512 차원
- Branch Cond: MLP(2 → 128 → 64)
- Branch Projection: concat(512,64) → Linear → K(256)
- Trunk: Sinusoidal Positional Encoding(L=6) + MLP(hidden=256, depth=6) → K(256)
- 결합: dot-product(branch K, trunk K) → 256×256 좌표에 대한 TL
- 활성함수: GELU(기본)

### 셰이크다운 훈련 설정(예시)
- 데이터: `R-DeepONet_Data/data/h5_mini`(샘플 60개 복사)
- 모드: coord(좌표 샘플링), `pts_per_map=1024`
- 배치: 8, 에폭: 15(초기 5 → 15), AMP 사용, grad_clip=1.0
- Optim/Scheduler: AdamW(lr=1e-3, wd=1e-2) + CosineAnnealingLR(T_max=epochs)
- 손실: MSE(기본). full 모드에서 SSIM/Gradient 옵션 가능(현재 기본 비활성)
- 체크포인트: `experiments/rdeeponet_v2_shakedown/best.pt`
- 곡선 저장(에폭 축 누적):
  - `loss_curve.png`: epoch vs train/val MSE
  - `mae_db_curve.png`: epoch vs train/val MAE(dB) [coord 모드]

### 셰이크다운 예비 결과(5 에폭 러닝 로그)
- `best_val`(val MSE, [0,1] 스케일): 0.01396
- 대략 RMSE ≈ 0.118 → TL(dB) 범위(80 dB)로 환산 시 ≈ 9.4 dB (거친 추정)
- 에폭/데이터를 늘리면 추가 개선 여지 있음

### 실행/재현
- 최초 1회 ResNet-18 가중치 자동 다운로드(캐시: `~/.cache/torch/hub/checkpoints`).
- 학습(셰이크다운):
  - `python train.py --config config_train_mini.yaml`
- 데이터 팩토리 재개:
  - `python main_factory.py --config config.yaml --resume --start_idx <N>`
- Windows OpenMP 충돌 시(Anaconda):
  - PowerShell: 
    - `$env:KMP_DUPLICATE_LIB_OK='TRUE'`
    - `$env:OMP_NUM_THREADS='1'`

### 추후 계획
- Defense_Wideband: SCOOTER + GEBCO 1000장 생성/검증.
- 학습 고도화: full 이미지 손실(SSIM/Gradient), HPO(Optuna/WandB), 평가 지표 강화.

---

## 문제/원인/해결 요약(Traceability)
- `.ray` 누락: CPU bellhop, ENV 첫 줄(Title) 보장, 빔 설정 동기화
- `.shd` 파싱: Fortran 레코드 무시 수동 파서, 절대 오프셋, float32/복소 분리
- TL 단색: r 단위 km→m, 감쇠(ap) 완화, GEBCO 비활성화(평탄 유지)
- 실패 복구: 산출물 존재 판정, `--resume`/`--start_idx`, temp 보존 옵션

---

## 폴더 구조 참고
- `R-DeepONet_Data/data/h5`: 훈련 H5
- `R-DeepONet_Data/data/images`: 학습 타일(TL/Ray)
- `R-DeepONet_Data/data/check`: 통합 검수 PNG
- `experiments/<run>/`: 체크포인트 및 곡선 PNG

본 문서는 지속적으로 업데이트됩니다. 물리/엔지니어링 변경 사항은 여기에 누적 기록합니다.
HPO 실행 계획 (12h, 단일 GPU, Optuna only)
[공통 설정 확정 요약]
Val=Test 동일 세트(20%) 사용. 목적: full 256x256 TL 맵 MAE(dB) 최소화.
정규화/역정규화: TL 40–120 dB.
시간예산: Study timeout=12h. Trial별 timeout 없음.
학습 길이: max_epochs=200, early_stopping_patience=30.
Pruner: ASHA(reduction_factor=3, grace_period=40, interval=5, min_epochs=40).
Sampler: multivariate TPE, seed=42, n_startup_trials=20.
동시성=1, Optuna storage(sqlite), dashboard 사용.
탐색공간: 넓게(lr, weight_decay, batch_size{4,8,12,16}, pts_per_map{1k,2k,4k,8k}, K{128..512}, trunk_hidden{256..512}, depth{4..10}, L{4..10}, dropout, pretrained, freeze_layers{none,l1,l1-2}, optimizer{Adam,AdamW}, scheduler{Cosine,OneCycle}, grad_clip, accumulate_steps{1,2,4}, num_workers{0,2,4}).
OOM fallback: batch_size→pts_per_map 순으로 단계적 축소 재시도.
Step-by-step
Step 0. 사전 점검 (데이터/환경)
파일 수/경로 확인: R-DeepONet_Data/data/h5/*.h5 개수, 샘플 H5 read 테스트.
CUDA 가용/VRAM 확인, PyTorch/CUDA 버전 로그.
실행 체크리스트
pass 기준: 에러 없이 통과, 파일 ≥1 확인, CUDA True.
산출물: output/logs/env_check.txt
Step 1. 평가 유틸 구현
기능: utils_eval.py
denorm_tl(norm, tl_min=40, tl_max=120)
infer_full_map(model, ray, cond, H=W=256, tile/batch 분할로 메모리 안전 추론)
mae_db_full(pred_map_db, gt_map_db) -> float
테스트: 단일 H5 샘플 로딩→full-map 추론→MAE(dB) 계산.
pass 기준: 추론/역정규화/MAE 정상, 메모리 OOM 없이 완료.
산출물: experiments/check/full_infer_smoke.png, env_check_eval.txt
Step 2. 학습 래퍼 구현
파일: train_runner.py
기능:
DataLoader 구성: split(0.8/0.2/0.0). coord 학습, val 전체는 full-map 평가용 루프 별도.
옵티마이저/스케줄러 생성(탐색공간 반영).
AMP/grad clip/accumulate steps/num_workers/pin_memory/persistent_workers.
Epoch 루프: coord 학습 MSE, 에폭말 val full-map 평가 MAE(dB).
ASHA/early stop 훅 연동을 위해 metric 로깅 콜백 제공.
테스트: 2~3 epoch 스모크(train: 몇 step만), full-map 평가 1회 성공.
산출물: experiments/smoke/loss_curve.png, 
mae_db_curve.png
, 
best.pt
Step 3. Optuna Objective/탐색공간/프루너/샘플러
파일: hpo_optuna.py
기능:
Study(storage=experiments/optuna/rdeeponet.db, study_name=rdeeponet_v2_full_mae_db).
Sampler: multivariate TPE(seed=42, n_startup_trials=20), Pruner: ASHA(r=3, grace=40, interval=5, min_epochs=40).
Objective: 시드 고정, config 샘플링→train_runner.fit_one_trial() 실행→val full-map MAE(dB) 반환.
Timeout=12h, GC/메모리 정리, 실패 trial 재시작 회피 처리.
아티팩트: trial 폴더 experiments/hpo/trial_<id>/best.pt, curves, sample_preds.png, 상위 k=5 유지.
OOM fallback: 예외 캐치→스케일 다운→재시도→포기 시 trial 실패 기록.
테스트: h5_mini로 trials=2, epochs=5 스모크.
산출물: experiments/optuna/rdeeponet.db, experiments/hpo/trial_*/*
Step 4. 대시보드/모니터링
명령:
optuna-dashboard --study-name rdeeponet_v2_full_mae_db --storage sqlite:///experiments/optuna/rdeeponet.db
체크리스트: 실시간 best trial 갱신, pruned/complete 카운트 확인.
Step 5. 본 HPO 실행 (12h)
데이터: 전체 
h5
설정: timeout=12h, concurrency=1
산출물: study DB, 상위 k=5 체크포인트 및 플롯.
Step 6. 결과 정리/앙상블
파일: ensemble_eval.py
상위 k 모델 로드→각각 full-map 예측→단순 평균/가중 평균(by 1/MAE)→MAE(dB) 리포트, 시각화 N=8.
산출물: experiments/ensemble/report.txt, examples/*.png
Step 7. 재현/재개 가이드
중단 후 --skip-finished로 재개, 동일 storage/study_name 사용.
체크포인트 보관: 상위 k=5, 나머지 자동 정리 옵션.
테스트/적용 단위 커맨드 예시
Env check
python -c "import torch,glob;print(torch.cuda.is_available());print(len(glob.glob('R-DeepONet_Data/data/h5/*.h5')))"
Dashboard
optuna-dashboard --study-name rdeeponet_v2_full_mae_db --storage sqlite:///experiments/optuna/rdeeponet.db
TODO 업데이트
[진행중] Optuna HPO 스크립트 생성(Study/Pruner/Sampler/Timeout=12h, Concurrency=1)
[대기] Objective 구현(coord 학습 + full-map MAE dB)
[대기] 탐색공간/스케줄러/Freeze/Accum/num_workers 반영
[대기] OOM Fallback
[대기] 체크포인트/아티팩트 정책(k=5)
[대기] 미니 스모크(2 trials)
[대기] 본 HPO 실행(12h)
[대기] 대시보드 가이드
[대기] 앙상블 평가 플로우
바로 Step 0-1 구현 들어가도 될까요? 승인 주시면 utils_eval.py와 train_runner.py부터 작성하고, 스모크 테스트까지 한번에 올리겠습니다.

Status: Processing... Changes: Plan: Step0→Step7 Bugs: full-mode batch 이슈(학습은 coord-only로 회피)

python hpo_optuna.py --config config_train.yaml --timeout-hours 24.0

optuna-dashboard sqlite:///experiments/optuna/rdeeponet.db

optuna-dashboard sqlite:///experiments/optuna/rdeeponet.db --port 8080
mlflow ui --port 5000

python hpo_optuna.py --config config_train.yaml --storage sqlite:///experiments/optuna/rdeeponet.db --study rdeeponet_v2_full_mae_db --timeout-hours 24

python hpo_optuna.py --config config_train.yaml --storage sqlite:///experiments/optuna/rdeeponet.db --study rdeeponet_v2_full_mae_db --timeout-hours 23

python hpo_optuna.py --config config_train.yaml --storage sqlite:///experiments/optuna/rdeeponet_v2_full_mae_db_v2.db --study rdeeponet_v2_full_mae_db_v2 --timeout-hours 23

mlflow ui --port 5000

optuna-dashboard sqlite:///experiments/optuna/rdeeponet.db --port 8080
Listening on http://127.0.0.1:8080/

python hpo_optuna.py --config config_train.yaml ^
  --storage sqlite:///experiments/optuna/rdeeponet_v2_full_mae_db_v2.db ^
  --study rdeeponet_v2_full_mae_db_v2 ^
  --timeout-hours 23

  SHAP HPO
  python tools\hpo_surrogate_shap.py --storage "sqlite:///experiments/optuna/rdeeponet_v2_full_mae_db_v2.db" --study rdeeponet_v2_full_mae_db_v2 --outdir experiments\interpret\rdeeponet_v2_full_mae_db_v2\hpo
  model해석
  python tools\model_explain.py --config config_train.yaml ^
  --study_dir experiments\optuna\rdeeponet_v2_full_mae_db_v2 ^
  --outdir experiments\interpret\rdeeponet_v2_full_mae_db_v2\model ^
  --use_topk --sample_cases 0 --nsamples_cond 50 --nsamples_ray 100
  
python tools\model_explain.py ^
  --config config_train.yaml ^
  --study_dir experiments\optuna\rdeeponet_v2_full_mae_db_v2 ^
  --outdir experiments\interpret\rdeeponet_v2_full_mae_db_v2\model ^
  --use_topk --sample_cases 0 ^
  --nsamples_cond 50 --nsamples_ray 100 ^
  --block 16 --lime_device cpu
  리포트
  python tools\build_tech_report.py --interpret_root experiments\interpret\rdeeponet_v2_full_mae_db_v2 --study rdeeponet_v2_full_mae_db_v2
  

   optuna-dashboard sqlite:///experiments/optuna/rdeeponet_v2_full_mae_db_v2.db
Listening on http://127.0.0.1:8080/

python tools\model_explain.py --config config_train.yaml --study_dir experiments\optuna\rdeeponet_v2_full_mae_db_v2 --outdir experiments\interpret\rdeeponet_v2_full_mae_db_v2\model --use_topk --sample_cases 4 --device cuda --bg_cond 128 --bg_ray 64 --coords_n 4096 --ray_instances 4 --nsamples_cond 2000 --nsamples_ray 4000 --block 16 --lime_num_samples 10000 --lime_num_features 24 --lime_device cpu

25%
python tools\model_explain.py --config config_train.yaml --study_dir experiments\optuna\rdeeponet_v2_full_mae_db_v2 --outdir experiments\interpret\rdeeponet_v2_full_mae_db_v2\model --use_topk --sample_cases 1 --device cuda --bg_cond 32 --bg_ray 16 --coords_n 1024 --ray_instances 1 --nsamples_cond 500 --nsamples_ray 1000 --block 16 --lime_num_samples 2500 --lime_num_features 6 --lime_device cpu

---

## Physics-Informed Loss (물리 일관성 손실)

### 개요
R-DeepONet 학습에 물리 제약 조건을 반영하는 손실 항을 추가하여 예측의 물리적 일관성을 향상시킵니다.

### 총 손실 공식
```
L = L_value + λ_rec · L_reciprocity + λ_sm · L_smooth
```

### 손실 항 설명

#### 1. Value Loss (L_value)
- 기본 회귀 손실 (Huber/MSE)
- 정규화된 TL 타깃에 대해 계산

#### 2. Reciprocity Loss (L_reciprocity)
- **물리적 의미**: 음원-수신기 상반성 (Source-Receiver Reciprocity)
- **수식**: `TL(r, z_r | z_s) ≈ TL(r, z_s | z_r)`
- **구현**: 배치 내에서 source depth와 query depth를 스왑하여 두 번 forward 후 차이 계산
- **목적**: 비대칭적 예측 패턴 억제

#### 3. Smoothness Loss (L_smooth)
- **물리적 의미**: 공간적 매끄러움 (||∇TL||² penalty)
- **구현**: Finite Difference로 ∂TL/∂r, ∂TL/∂z 근사
- **목적**: 고주파 아티팩트 및 수치 노이즈 억제

### Warmup Schedule
- Physics loss는 `warmup_epochs` 이후부터 선형적으로 활성화
- 초반에는 value loss로 수렴하고, 후반에 physics 제약 적용

### Config 예시
```yaml
loss_weights:
  value: 1.0
  reciprocity: 0.01
  smooth: 0.001

physics_loss:
  enabled: true
  warmup_epochs: 15
  warmup_type: linear
  auto_scale: true
  reciprocity:
    n_samples: 64
  smooth:
    delta: 0.00390625  # 1/256
    n_samples: 128
```

### 비교 실험 실행
```bash
# Value-only vs Loss-guided 비교
python run_comparison.py --config config_stage2_highres.yaml --output_dir experiments/comparison

# Quick test
python run_comparison.py --config config_train.yaml --quick
```

### 평가 메트릭
- **dB MAE/RMSE**: 기본 정확도
- **Reciprocity Violation (dB)**: 상반성 위반 정도
- **Gradient Mean / High-Freq Ratio**: 아티팩트 지표
- **Inference Time**: 추론 속도