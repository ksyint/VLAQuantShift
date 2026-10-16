# 원본 프로젝트와 수정 범위

이 프로젝트는 [VLAQuantBench](https://github.com/jiuyixu25/VLAQuantBench)를 기반으로 개발했습니다.

시작 커밋은 원본 `master` 브랜치의 `4a2cb7c1bf9160ae683fe598ec583d7c9b9c13c9`입니다. 원본 Git 이력을 유지했으며, MIT 라이선스의 저작권 표시와 이용 허락 문구는 [LICENSE](LICENSE)에 그대로 보존했습니다.

VLAQuantBench의 저자는 Jiuyi Xu, Qing Jin, Meida Chen, Song Wang, Yang Sui, Yangming Shi입니다. 원본 논문은 [VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2609.25376)입니다. 원본 README와 인용 정보는 각각 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md), [docs/UPSTREAM_CITATION.cff](docs/UPSTREAM_CITATION.cff)에 보관했습니다.

## 원본에서 가져온 부분

`vlaquantbench/`, 기존 설정·스크립트·테스트·진단 도구·그림과 `results/`는 변경 내역에 별도로 표시된 부분을 제외하면 원본 프로젝트의 작업입니다. 기존 측정값은 원본 VLAQuantBench의 데이터이며, VLAQuantShift에서 새로 수행한 실험 결과가 아닙니다.

보관된 원본 README는 수정하지 않았습니다. 따라서 그 안의 상대 경로 링크는 원본 저장소의 루트 디렉터리를 기준으로 작성되어 있습니다.

## 새로 개발한 부분

`vlaquantshift/`, `configs/shift/`, `tests/test_shift_*.py`, `docs/shift_*.md`, `paper/shift_*`, `verification/` 및 새 CI 워크플로는 확장 기능을 구현하거나 설명합니다. 한국어 브랜치는 이 확장 기능을 이해하기 위한 안내 문서도 제공합니다.

원본의 `vlaquantbench/cli.py`, `benchmarks/base.py`, `results.py`에는 관측 변화 적용 지점, 에피소드 매칭을 확인할 기록, 보정 데이터 제어, 중단된 실행 재개 시 호환성 확인을 추가했습니다. 원본의 양자화 수식과 모델 어댑터는 유지했습니다.

원본과 현재 브랜치의 정확한 차이는 다음 명령으로 확인할 수 있습니다.

```bash
git diff 4a2cb7c1bf9160ae683fe598ec583d7c9b9c13c9..HEAD
```

패키지 메타데이터와 저장소 첫 화면의 README는 확장 프로젝트를 설명합니다. 원본 저자 정보는 이 문서와 Git 이력, 인용 정보, 라이선스에 유지합니다. **출처를 명시하는 것만으로 연구의 독창성이 입증되지는 않습니다.** 새로운 주장을 하려면 통제된 실험과 선행연구 비교가 필요합니다.

이 확장 프로젝트는 사전 학습 체크포인트, 외부 벤치마크 자산, 제3자 논문 본문을 재배포하지 않습니다. 모델과 시뮬레이터에는 각각의 라이선스와 사용 요건이 적용됩니다. 이 프로젝트는 원본 저자들과 독립적으로 개발합니다.
