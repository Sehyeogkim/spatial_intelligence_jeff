# Submission Copy

## Project name

**World2Work**

## One-liner

Turn a photo of your space into a learned robot policy and reusable experience data.

## English description

World2Work turns a photo of a user’s environment into a site-conditioned navigation training ground. World Labs Marble reconstructs Corgi Cafe as an explorable world. We convert its fused collider into a 35-centimeter occupancy grid, train a four-action tabular Q-learning policy for 4,000 episodes, and export 94,906 real state-action transitions. On 20 held-out start-cell/orientation configurations, a fixed-seed random policy achieved 5% success, a 21.4076% collision-action rate, and 0.351351 path efficiency; the trained greedy policy achieved 100%, 0%, and 1.0. Held-out configurations were excluded only as episode starts, not guaranteed unseen states. Finally, an official Nova Carter executed the learned route in Isaac Sim on RunPod: 36.4667 seconds, 366 logged steps, success, and zero collisions. Marble supplies the visual world and geometry-derived planning context; stable robot physics currently uses flat ground and collision proxies, so this is not a fully automatic digital twin. The output is a reproducible policy, evaluation, and robot-experience dataset. VLA/GR00T integration is future work.

## 한국어 설명

World2Work는 사용자 환경 사진을 Marble 3D 공간과 점유 격자로 바꾸고, 4,000회 Q-learning으로 Nova Carter의 배달 경로를 학습합니다. 94,906개 transition을 생성했으며, 20개 held-out 시작 조건에서 성공률이 random 5%에서 trained 100%로 향상됐습니다. 학습 경로는 RunPod의 Isaac Sim에서 충돌 없이 검증했습니다.

## 사용 기술

- **World Labs Marble:** 한 장의 장소 사진에서 탐색 가능한 3D world와 fused collider 생성
- **Marble-to-grid pipeline:** collider의 metric geometry에서 0.35 m occupancy grid와 obstacle proxy 생성
- **Tabular Q-learning:** 4,000 episodes, 4방향 action, 정책·경로·94,906개 transition export
- **NVIDIA Isaac Sim 6.0.1 / Nova Carter:** 학습 경로의 물리 실행과 10 Hz telemetry 검증
- **Evaluation dashboard:** 동일한 20개 held-out start-cell/orientation configuration에서 random baseline과 trained policy 비교

## 평가 정의

- **Success rate:** 목표에 도달한 episode 수 / 전체 held-out episode 수
- **Collision rate:** occupied 또는 out-of-bounds cell로 이동을 시도한 action 수 / 전체 action 수
- **Path efficiency:** 성공 episode에 대해 `shortest steps / max(actual steps, shortest steps)`를 계산한 평균. 실패 episode는 제외한다.
- **Held-out 범위:** 20개 configuration은 학습 episode의 시작점에서만 제외됐다. 학습 경로가 해당 cell을 통과할 수 있으므로 unseen-state 일반화라고 주장하지 않는다.

## 정직한 limitation

Marble 결과는 object semantics가 없는 fused static mesh다. 현재 데모는 collider에서 얻은 바닥 범위와 장애물 증거를 단순화한 planning map을 사용하며, 안정적인 Nova Carter 물리 실행은 flat ground와 collision proxy 위에서 검증했다. `table_7`은 자동 인식된 실제 테이블이 아니라 reachable goal proxy이고, 커피는 트레이에 부착된 payload다. 현재 모델은 tabular Q-learning이며 VLA나 GR00T를 학습하지 않았다.

Google Maps 사진은 해커톤용 world-generation 입력으로만 사용했다. 공개 또는 상용 배포 전에는 venue-owned, licensed, 또는 팀이 직접 촬영한 이미지로 교체해야 한다.

## Demo video (final)

- `web-demo/assets/world2work-demo.mp4` (30 s): NVIDIA Nova Carter delivering coffee to table 7 inside the World Labs Marble reconstruction of Corgi Cafe. This is a visual replay of an actual Isaac Sim physics episode (350 logged steps at 10 Hz, 7.24 m driven, target reached, held-out start cell (2,7) heading south) rendered from the panorama capture point.
- What the robot knows: the route was planned by the trained policy on an occupancy grid derived from the Marble collider, so furniture is avoided by construction.
- Honest limits: the panorama backdrop has no depth, so café furniture does not occlude the robot in the video; the replayed episode ran on a flat physics floor. A second run with Marble collider collision enabled (`output/isaac/fit_geom.jsonl`) reached the target over the same 7.24 m route. There is no online perception; this is not a digital twin.
