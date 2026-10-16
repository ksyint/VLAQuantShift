# 선행연구와 중복 검토

2026-10-16 기준으로 논문 원문 페이지와 저자 저장소를 확인해 정리한 내용이다. 아래 설명은 직접 인용이 아닌 요약이다. 제한된 범위의 문헌 검토이며, 독창성을 빠짐없이 검증했다는 뜻은 아니다. 이 저장소는 원본 VLAQuantBench의 출처를 밝히고 확장한다. 가져온 코드와 결과는 이름을 바꾼다고 새로운 기여가 되지 않는다.

## 가장 가까운 선행연구와 차별화 후보

표의 폐루프(closed-loop) 평가는 정책의 행동이 다음 관측과 상태에 영향을 주는 실행 환경에서의 평가를 뜻한다. PTQ(post-training quantization)는 학습 후 양자화다.

| 자료 | 이미 다룬 내용 | VLAQuantShift에서 고려할 점 |
| --- | --- | --- |
| [VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2609.25376) | 폐루프에서의 양자화 형식/적용 범위 비교, 보정 통제, 고정 관측 재생, 행동 충실도, 태스크 군집 단위 불확실성. | 프레임워크를 재사용하고 인용한다. 환경 변화의 네 조건 대비와 에피소드 짝짓기 유지에 초점을 둔다. 최초의 폐루프 양자화 벤치마크라고 주장하지 않는다. |
| [QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2602.20309) | 어텐션 온도와 출력 에너지 보정을 사용하는 선택적 양자화. | 보정의 강건성을 평가할 수 있다. 기존 보정 기법의 이름을 바꾸는 것은 독창성의 근거가 아니다. |
| [QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization](https://arxiv.org/abs/2602.03782) | 행동 민감도에 따른 채널 배분, 혼합 정밀도, 누적 행동 민감도. | 행동 평균제곱오차(MSE) 기반 정밀도 선택기는 비교 기준이다. 누적 행동 오차만으로는 새로운 기여가 되지 않는다. |
| [Beyond In-Distribution Preservation: Recovering Generalization in Quantized VLAs via Vulnerability-Oriented Tuning](https://arxiv.org/abs/2610.05745) (PIVOT-Q) | LIBERO-Plus의 환경 변화에서 양자화 VLA의 강건성을 평가하고, 정책이 직접 방문하는 취약 상태를 선택적으로 보정. | 강건성 주제에서 가장 가까운 연구다. 양자화 정책의 취약성을 처음 발견했다고 주장하기보다, 검증 가능한 상호작용 추정과 평가에 초점을 둔다. 실험적으로 무엇이 다른지는 여전히 근거가 필요하다. |
| [DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models](https://arxiv.org/abs/2604.11572) | 궤적 이탈(drift), 표현 보상, 움직임 기반 혼합 정밀도. | 궤적 이탈 고려와 궤적 기반 정밀도 배분은 기존 아이디어다. 오프라인 분해는 진단 도구이며, 검증 없이 새로운 양자화기로 취급해서는 안 된다. |
| [LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models](https://arxiv.org/abs/2510.13626) | 시각·기하·언어·상태 변화에 따른 VLA 평가. | 연동 후 같은 조건의 벤치마크 비교를 하려면 공식 변화 조건을 사용한다. 자체 관측 손상 변환은 별도로 이름을 붙이고 문서화한다. |
| [Calibrate Where You Deploy](https://github.com/parastoopil/on-policy-calibration-vla) | 정책이 직접 방문한 상태를 사용하는 온폴리시(on-policy) 보정, 짝지은 평가, 상태 분포 진단을 연구하는 공개 저장소. 부정적인 결과를 보고함. | 검증된 동료 심사 논문이 아닌 소프트웨어/연구 산출물로 인용한다. 보고된 측정값을 이 프로젝트에서 독립적으로 재현하지는 않았다. 온폴리시 보정과 짝지은 궤적은 아직 아무도 다루지 않은 아이디어가 아니다. |

원본의 [한계 절](https://arxiv.org/html/2609.25376v1#S7)은 고정 관측 재생과 양자화 정책이 실제로 방문하는 달라진 상태 분포를 명확히 구분한다. 또한 태스크 부트스트랩이 에피소드 짝짓기를 유지하지 않는다고 설명한다. 이는 구체적인 확장 지점이지만, 소프트웨어의 한계를 보완하는 것만으로 출판 가능한 과학적 기여가 확립되는 것은 아니다.

## 추가로 비교해야 할 연구

- [DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models](https://arxiv.org/abs/2603.07904)는 현재 v3에서 동적 활성값 정밀도와 누적 회전 오차 보상을 연구한다. 실행 중 정밀도 전환이나 단계별 정밀도를 추가한다면 비교해야 한다. 이전 버전은 운동학 기반 관점을 사용했으므로 인용 버전을 고정한다.
- [ActQuant: Sub-4-bit Action-Guided Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2605.24011)는 행동 기반 혼합 정밀도와 스케일 최적화를 사용한다. 예산 제약하의 정밀도 배분은 이 방법 및 QVLA와의 차이를 밝혀야 한다.
- [HBVLA: Pushing 1-Bit Post-Training Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2602.13710)는 정책 민감도를 고려한 이진화를 사용한다. 같은 조건으로 비교할 수 있는 구현이 확보되면 비교 대상에 포함한다.
- [Mixed-Precision Quantization for Deep Vision Models with Integer Quadratic Programming](https://arxiv.org/abs/2307.05657)(CLADO, 현재 v2 제목)는 층 사이의 양자화 의존성을 다룬다. 층 간 상호작용은 로보틱스 밖에서도 이미 연구되었다. 이전 v1의 제목은 *Mixed-Precision Quantization with Cross-Layer Dependencies*였다.
- [VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks](https://arxiv.org/abs/2412.18194)와 [공식 코드](https://github.com/OpenMOSS/VLABench)는 더 넓은 태스크 환경을 제공한다. VLAQuantBench가 이미 여기에서 X-VLA를 평가했으므로, 같은 벤치마크를 연결하는 것만으로는 새롭지 않다.

## 주장할 수 있는 범위

제안하는 추가 성능 저하(excess degradation) 통계량은 표준적인 요인 설계의 상호작용 대비(factorial interaction contrast)다. 정책과 상태를 교차한 행동 항등식도 기초적인 대수 분해다. 어느 쪽도 새로운 추정량이나 이론으로 내세워서는 안 된다. 이 프로젝트에서의 가치는 에피소드 짝짓기를 명시적으로 유지하고, 데이터 누수를 검사하며, 정밀도 설정에 관한 질문을 반증할 수 있도록 만드는 재현 가능한 프로토콜에 있다.

향후 근거를 갖춰 주장할 수 있는 실증 기여의 예는, 기존 비교가 확인하지 못했던 상호작용이나 설정 순위의 역전을 독립적인 모델·태스크·보정 데이터셋에서 재현하는 것이다. 이를 주장하려면 실제 네 조건의 측정값과 적절한 불확실성 평가가 필요하다. PIVOT-Q의 강건성 결과와 DA-PTQ의 궤적 이탈 관점과 비교한 뒤에도 차별점이 유지되어야 한다.

초기 구현은 원본 실행기를 통한 관측 개입과 오프라인 행동 배열 분석을 제공한다. 물리적인 조명 변화, 공식 LIBERO-Plus 연동, 시뮬레이터 상태를 분기한 실행, 무작위성을 맞춘 정책 간 교차 재생, 새로운 실제 로봇 측정은 아직 제공하지 않는다. 소형 합성 테스트 환경(toy fixture)는 소프트웨어 연동을 검증할 뿐이며, VLA에서의 효과·속도 향상·출판 수준의 독창성을 입증하지 못한다.

모든 자료는 [참고문헌 파일](../paper/shift_references.bib)에 대응하는 항목이 있다. arXiv 제목과 저자는 버전 사이에 바뀔 수 있으므로 최종 실험 명세(manifest)에서 논문 버전과 코드 리비전을 고정한다.
