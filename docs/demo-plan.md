# World2Work — 최종 데모 계획

## 한 문장

사용자 환경 사진을 Marble 공간과 내비게이션 맵으로 바꾸고, 그 공간에서 학습한 정책으로 Nova Carter가 커피를 목표 지점까지 배달하며 학습 경험과 평가 결과를 함께 내보낸다.

## 검증된 파이프라인

```text
Corgi Cafe 사진
  → World Labs Marble 3D world + fused collider.glb
  → 35 cm occupancy grid
  → 4-action tabular Q-learning, 4,000 episodes
  → policy + route + 94,906 training transitions
  → 20개 held-out 조건에서 Untrained vs Trained 평가
  → Isaac Sim의 공식 Nova Carter가 학습 경로 실행
```

현재 점유 격자는 Marble collider의 바닥 범위와 높이 구간의 장애물 증거에서 만들어졌다. fused mesh 노이즈로 직접 추출한 free space가 조각나서, 최종 맵은 collider의 robust floor bounds와 rasterized obstacle proxy를 사용하는 명시적 fallback이다. `table_7`은 semantic하게 인식한 실제 테이블이 아니라, 추출된 free-space island 안에서 선택한 reachable delivery-goal proxy다.

## 실제 결과

- **학습:** tabular Q-learning 4,000 episodes, 4개 절대 방향 action, 실제 transition 94,906개
- **맵:** 0.35 m/cell, 24×9, free cell 81개
- **평가:** 학습 episode의 고정 시작점과 다른 20개 start-cell/orientation configuration
- **Untrained random:** 성공률 5%, collision rate 21.4076%, path efficiency 0.351351
- **Trained greedy:** 성공률 100%, collision rate 0%, path efficiency 1.0
- **Isaac 검증:** RunPod의 Isaac Sim 6.0.1에서 공식 Nova Carter 실행, 36.4667초, 366개 logged step, 성공, 충돌 0회

평가의 `collision rate`는 **occupied 또는 out-of-bounds cell로 이동을 시도한 action 수 / 전체 action 수**다. `path efficiency`는 성공 episode에 한해 **shortest-path steps / max(actual steps, shortest-path steps)**를 계산한 뒤 평균했다. Untrained의 효율은 성공한 1개 episode만 대상으로 한 값이다.

Held-out configuration은 학습 episode의 **시작 조건에서만 제외**했다. 학습 rollout이 해당 cell을 지나갈 수 있으므로 unseen-state 결과라고 주장하지 않는다. 초기 orientation은 downstream controller에 전달되는 조건이며, 현재 high-level grid policy는 절대 방향 action을 출력하므로 orientation-invariant다.

## 2분 데모 순서

1. **0:00–0:15 — 사용자 입력**  
   Corgi Cafe 사진과 `Deliver coffee to table 7` 요청을 보여준다.
2. **0:15–0:35 — World 생성**  
   완성된 Marble 카페를 탐색하고, 같은 world에서 받은 fused collider를 표시한다.
3. **0:35–0:55 — World를 작업 공간으로 변환**  
   collider에서 생성한 occupancy map, 시작점, `table_7` goal, 최단경로를 보여준다.
4. **0:55–1:15 — 실제 학습 데이터**  
   4,000-episode learning curve와 94,906-line `training_experience.jsonl`의 state/action/reward transition을 보여준다.
5. **1:15–1:35 — Untrained vs Trained**  
   동일한 20개 held-out 조건에서 `5% → 100%` 성공률, `21.4076% → 0%` collision rate를 나란히 보여준다.
6. **1:35–1:55 — 물리 실행**  
   Nova Carter가 트레이에 커피를 싣고 학습 경로를 따라 목표에 도착하는 RunPod 검증 영상을 재생한다.
7. **1:55–2:00 — 마무리**  
   `Your space → a learned robot policy → reusable experience`로 끝낸다.

라이브 world 생성이나 재학습은 하지 않는다. 완성된 Marble world, 생성된 데이터, 평가 화면, 검증된 Carter 녹화본을 순서대로 사용한다.

## 화면에 보여줄 산출물

- `occupancy.png` — Marble-derived navigation grid
- `value_heatmap.png` — 학습된 Q-value와 최종 route
- `learning_curve.json` — 4,000개 학습 episode
- `training_experience.jsonl` — 94,906개 실제 training transition
- `evaluation_summary.json` — 20개 held-out 조건의 두 정책 비교
- `policy_table_7.json`, `route_table_7.json` — 학습 정책과 Carter waypoint
- Isaac 결과 — 36.4667초, 366 logged step, success, collision 0

## 정직한 경계

- Marble collider는 object semantics와 movable joint가 없는 fused static mesh다.
- 안정적인 Carter 물리 검증은 flat ground에서 수행했으며, Marble은 visual world와 geometry-derived planning/proxy context로 사용했다.
- 커피는 트레이에 부착된 payload이며 grasp/manipulation task가 아니다.
- 현재 학습은 tabular Q-learning이다. VLA 또는 GR00T를 학습했다고 주장하지 않으며, 생성 데이터를 그 방향으로 확장하는 것은 future work다.
