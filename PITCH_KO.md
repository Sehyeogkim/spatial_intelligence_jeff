# World2Work — 2분 발표 대본

## 핵심

- **가치제안:** 사용자가 자기 환경 사진과 로봇 task를 주면, 그 장소에 맞는 정책과 학습 경험 데이터를 만든다.
- **차별점:** Marble world를 배경으로만 쓰지 않고 collider를 navigation map으로 변환해, 학습·held-out 평가·Isaac 물리 실행까지 하나의 재현 가능한 파이프라인으로 연결한다.
- **검증 결과:** 94,906개 transition, random 대비 trained 성공률 `5% → 100%`, collision rate `21.4076% → 0%`, Nova Carter 실제 시뮬레이션 성공.

## 발표 대본

### 0:00–0:15 — 문제와 입력

**[화면: Corgi Cafe 원본 사진 + `Deliver coffee to table 7`]**

“로봇을 새로운 장소에 배치하려면 그 장소에서 다시 데이터를 모으고 정책을 검증해야 합니다. World2Work의 입력은 사용자의 환경 사진 한 장과 원하는 로봇 task입니다.”

### 0:15–0:35 — 사진에서 World로

**[화면: 완성된 Marble 카페를 짧게 탐색]**

“World Labs Marble은 이 사진을 탐색 가능한 3D 카페로 만듭니다. 저희는 여기서 멈추지 않고, Marble이 제공한 fused collider의 metric geometry를 로봇이 학습할 수 있는 35센티미터 점유 격자로 변환했습니다.”

### 0:35–0:55 — World에서 정책으로

**[화면: occupancy map에서 start와 주황색 `table_7` goal 표시]**

“이 map 위에서 상태는 로봇 cell, 행동은 네 방향 이동입니다. 4,000번의 Q-learning episode를 실제 실행해 94,906개의 state, action, reward, next-state transition과 최종 정책을 만들었습니다.”

### 0:55–1:25 — 학습 전후 검증

**[화면: learning curve와 Untrained / Trained metric 카드]**

“같은 20개 held-out 시작 조건으로 학습 전후를 비교했습니다. 고정 seed random policy의 성공률은 5퍼센트, collision rate는 21.4076퍼센트였습니다. 학습된 greedy policy는 성공률 100퍼센트, collision rate 0퍼센트, 최단경로 대비 효율 1.0을 기록했습니다.”

“여기서 collision rate는 막힌 cell로 이동하려 한 action의 비율이고, path efficiency는 성공한 episode만 대상으로 최단 step을 실제 step으로 나눈 값입니다. Held-out 조건은 학습 episode의 시작점에서 제외했지만, 학습 중 그 cell을 지나갈 수 있으므로 unseen state라고 과장하지 않습니다.”

### 1:25–1:52 — 정책을 로봇으로

**[화면: RunPod Isaac Sim의 Nova Carter 성공 영상 + telemetry]**

“마지막으로 이 정책의 경로를 Isaac Sim의 공식 Nova Carter에 전달했습니다. 로봇은 커피가 실린 트레이를 운반해 36.4667초 만에 목표에 도착했고, 366개 state를 기록했으며 충돌은 없었습니다.”

### 1:52–2:00 — 마무리

**[화면: `Your space → Learned policy → Robot experience`]**

“World2Work는 사용자의 공간을, 학습된 로봇 정책과 재사용 가능한 경험 데이터로 바꿉니다.”

## 질문이 나오면 정직하게 답할 내용

- `table_7`은 collider에 semantic label이 없어 자동 인식한 실제 테이블이 아니라 reachable delivery-goal proxy다.
- collider는 fused static mesh다. 안정적인 Carter 물리는 flat ground에서 실행했고, Marble은 visual world와 geometry-derived obstacle/planning proxy로 사용했다.
- 커피는 트레이에 부착된 payload다. grasp task가 아니다.
- 현재 정책은 tabular Q-learning이며 VLA/GR00T 학습이 아니다. 향후 생성 episode를 더 복잡한 imitation learning 또는 VLA post-training으로 확장할 수 있다.

## 화면 하단 출처 표기

> Reference world: Corgi Cafe, San Francisco — AI reconstruction generated from a publicly available interior photo by Maksym Kuzhdin via Google Maps (Feb 2026). Not an official digital twin; not affiliated with or endorsed by Corgi Cafe or Google. Source photo may be copyright-protected; attribution is not permission.
