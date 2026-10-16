# 소프트웨어 검증 기록

이 디렉터리는 재현 가능한 **소프트웨어 검증 기록**을 보관합니다. 여기에 있는 결과는 VLA 모델이나 실제 로봇의 성능을 입증하는 자료가 아닙니다. 합성 데모의 제어기는 고정된 선형 가중치와 RGB 값으로 인코딩한 상태를 사용하며, 원본 VLAQuantBench의 RTN 양자화기와 폐루프 실행기를 호출합니다.

저장소 루트에서 다음 명령을 실행합니다.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q
CUDA_VISIBLE_DEVICES=6 python -m pytest -q
CUDA_VISIBLE_DEVICES='' vqs demo --out artifacts/verification_cpu --episodes 8 --bootstrap 500
CUDA_VISIBLE_DEVICES=6 vqs demo --out artifacts/verification_gpu6 --device cuda:0 --episodes 4 --bootstrap 500
python scripts/verify_cells.py
```

마지막 명령은 원본 프로젝트의 기존 기록을 점검하며 새 실험을 수행하지 않습니다. 새 데모 실행은 소스 리비전, 실행 환경, 합성 실험 여부를 기록합니다. 이 데모의 시간 측정에는 Python 계측 오버헤드가 포함되므로 모델이나 하드웨어 사이의 성능 비교에 사용할 수 없습니다.

`CUDA_VISIBLE_DEVICES=6`으로 물리 GPU 6만 노출하면, 해당 프로세스 안에서는 그 GPU가 `cuda:0`으로 보입니다. 위 명령의 `--device cuda:0`은 물리 GPU 6을 사용한다는 뜻입니다.

## 수행한 검증 — 2026-10-16

- CPU 테스트: **175개 통과, GPU 전용 테스트 5개 건너뜀**.
- CUDA 6 테스트: **180개 통과**. 두 테스트 실행에서 각각 원본 코드의 multiprocessing/fork 관련 사용 중단 예정 경고 224개가 발생했으며, 전체 요약을 이 디렉터리에 보관했습니다.
- CPU 합성 통합 실행: **실험 조건 9개에서 에피소드 288회**.
- 물리 GPU 6 / NVIDIA B200 합성 통합 실행: **실험 조건 9개에서 에피소드 144회**. CUDA 7은 사용하지 않았습니다.
- 합성 통합 실행 합계: **432 에피소드**. 실제 VLA 모델 실험 횟수가 아닙니다.
- 원본 데이터 점검: **실험 조건 417개 검증**. 이 기록은 원본 VLAQuantBench의 데이터입니다.
- 편집 가능 모드의 패키지 설치와 `vqs`·`vqb` 명령 등록에 성공했습니다. 48개 조건의 파일럿 계획과 24개 조건의 보정 실험 계획은 유효성 검사를 통과했으며, 실제 모델 평가는 시작하지 않았습니다.

측정에 사용한 소스는 커밋 `95dd37a`이며, 전체 SHA는 [summary.json](summary.json)에 있습니다. 보관된 각 실행 기록에는 수집 당시 소스 트리에 미커밋 변경이 없었다는 사실이 기록되어 있습니다. JSONL 파일의 바이트는 수집 당시 그대로입니다. 보관용 실행 계획에는 저장소 기준 상대 경로를 사용했으며, 보고서는 옮겨진 기록을 대상으로 다시 계산했습니다.

```bash
vqs analyze-plan verification/synthetic_cpu/plan.json --out artifacts/recomputed-cpu.json --bootstrap 500
vqs analyze-plan verification/synthetic_gpu6/plan.json --out artifacts/recomputed-gpu6.json --bootstrap 500
```

[CPU 보고서](synthetic_cpu/report.json), [GPU 보고서](synthetic_gpu6/report.json), [기록 파일의 SHA256](summary.json)을 확인할 수 있습니다. 성공률이 바닥에 도달하는 현상을 포함한 데모의 차이는 합성 환경의 특성이며, 사전 학습 VLA에 대한 주장을 뒷받침하지 않습니다.

## 아직 수행하지 않은 작업

실제 LIBERO/VLABench 체크포인트와 시뮬레이터 자산을 준비하고 전체 실험을 수행해야 합니다. LaTeX 초안은 인용 키의 일관성만 확인했으며, TeX 도구가 없어 컴파일하지 않았습니다. GitHub Actions는 별도로 CPU 검증을 수행합니다. 이 디렉터리의 로컬 로그만으로 원격 CI의 성공 여부를 판단할 수는 없습니다.
