"use strict";

const ARTIFACT_BASE = "./data";
const ROLLOUT_VIDEO_URL = "./assets/world2work-demo.mp4";
const TARGET_ID = "table_7";
const POLICY_LABEL = "tabular Q-learning high-level policy";
const REPLAY_LABEL = "kinematic browser replay";

const $ = (selector) => document.querySelector(selector);
const elements = {
  artifactStatus: $("#artifactStatus"),
  liveDot: $(".live-dot"),
  clock: $("#clock"),
  sourcePhoto: $("#sourcePhoto"),
  marblePano: $("#marblePano"),
  gridSourcePill: $("#gridSourcePill"),
  navCanvas: $("#navCanvas"),
  curveCanvas: $("#curveCanvas"),
  heatmapCanvas: $("#heatmapCanvas"),
  curveStatus: $("#curveStatus"),
  heatmapStatus: $("#heatmapStatus"),
  rolloutPanel: $("#rolloutPanel"),
  rolloutVideo: $("#rolloutVideo"),
  rolloutStatus: $("#rolloutStatus"),
  rolloutFallbackTitle: $("#rolloutFallbackTitle"),
  rolloutFallbackText: $("#rolloutFallbackText"),
  retryRolloutButton: $("#retryRolloutButton"),
  hudPhase: $("#hudPhase"),
  hudPhaseDot: $("#hudPhaseDot"),
  hudCell: $("#hudCell"),
  hudProgress: $("#hudProgress"),
  successCard: $("#successCard"),
  successDistance: $("#successDistance"),
  runButton: $("#runButton"),
  resetButton: $("#resetButton"),
  replayButton: $("#replayButton"),
  exportButton: $("#exportButton"),
  successMetric: $("#successMetric"),
  collisionMetric: $("#collisionMetric"),
  efficiencyMetric: $("#efficiencyMetric"),
  evaluationStatus: $("#evaluationStatus"),
  evaluationNote: $("#evaluationNote"),
  untrainedBar: $("#untrainedBar"),
  trainedBar: $("#trainedBar"),
  untrainedScore: $("#untrainedScore"),
  trainedScore: $("#trainedScore"),
  eventCount: $("#eventCount"),
  eventLog: $("#eventLog"),
  fieldEpisode: $("#fieldEpisode"),
  fieldPose: $("#fieldPose"),
  fieldWheels: $("#fieldWheels"),
  fieldCell: $("#fieldCell"),
  fieldCommand: $("#fieldCommand"),
  fieldDone: $("#fieldDone"),
  toast: $("#toast"),
};

const phases = [...document.querySelectorAll(".phase")];
const phaseOrder = ["load", "policy", "navigate", "arrive"];

let artifactData = createFallbackArtifacts();
let artifactSources = { grid: false, meta: false, route: false, curve: false, policy: false, evaluation: false };
let playback = {
  running: false,
  token: 0,
  progress: 0,
  animationFrame: 0,
  lastSampleAt: 0,
  lastCellIndex: -1,
};
let currentEpisode = createEpisode();
let toastTimer = 0;
let rolloutLoadToken = 0;

function createFallbackArtifacts() {
  const rows = 24;
  const cols = 9;
  const cells = Array.from({ length: rows }, (_, row) =>
    Array.from({ length: cols }, (_, col) => {
      const border = row === 0 || row === rows - 1 || col === 0 || col === cols - 1;
      const islandA = row >= 5 && row <= 10 && col >= 2 && col <= 4;
      const islandB = row >= 13 && row <= 18 && col >= 6 && col <= 7;
      const islandC = row >= 19 && row <= 21 && col >= 1 && col <= 2;
      return border || islandA || islandB || islandC ? 1 : 0;
    }),
  );
  const routeCells = [
    [1, 7], [2, 7], [3, 7], [3, 6], [4, 6], [5, 6], [6, 6], [7, 6],
    [7, 5], [8, 5], [9, 5], [10, 5], [11, 5], [11, 4], [12, 4], [13, 4],
    [14, 4], [15, 4], [16, 4], [17, 4], [18, 4], [19, 4], [20, 4], [21, 4], [22, 4],
  ];
  routeCells.forEach(([row, col]) => { cells[row][col] = 0; });

  const origin = [2.95, -0.92];
  const cellSize = 0.35;
  const waypoints = routeCells.map(([row, col]) => [
    Number((origin[0] + (col + 0.5) * cellSize).toFixed(5)),
    Number((origin[1] + (row + 0.5) * cellSize).toFixed(5)),
  ]);
  const rolling = Array.from({ length: 80 }, (_, index) => {
    const episode = (index + 1) * 50;
    const learned = 1 - Math.exp(-index / 13);
    return {
      episode,
      mean_reward_fixed_start: -31 + learned * 49 + Math.sin(index * 0.42) * (2.4 * (1 - learned)),
      success_rate_fixed_start: Math.min(1, Math.max(0, learned * 1.08 - 0.07 + Math.sin(index * 0.31) * 0.035)),
    };
  });
  const qValues = {};
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      if (cells[row][col] === 0) {
        const distance = Math.abs(22 - row) + Math.abs(4 - col);
        const value = 18 - distance * 0.43;
        qValues[`${row},${col}`] = [value - 0.4, value - 0.8, value - 1.0, value - 0.6];
      }
    }
  }

  return {
    grid: { schema_version: "preview.occupancy-grid.v1", rows, cols, occupied_value: 1, cells },
    meta: {
      schema_version: "preview.grid-meta.v1",
      cell_size_m: cellSize,
      origin_xy_m: origin,
      rows,
      cols,
      start: { cell: routeCells[0], xy_m: waypoints[0] },
      targets: { table_7: { cell: routeCells.at(-1), xy_m: waypoints.at(-1) } },
    },
    route: {
      schema_version: "preview.nav-route.v1",
      target_id: TARGET_ID,
      success: true,
      cells: routeCells,
      waypoints_xy_m: waypoints,
      steps: routeCells.length - 1,
      path_length_m: (routeCells.length - 1) * cellSize,
    },
    curve: {
      schema_version: "preview.learning-curve.v1",
      algorithm: "tabular_q_learning",
      target_id: TARGET_ID,
      episodes: Array.from({ length: 4000 }, (_, index) => ({ episode: index + 1 })),
      rolling,
      summary: { fixed_start_success_rate_last_100: 1, greedy_success: true, greedy_steps: routeCells.length - 1 },
    },
    policy: {
      schema_version: "preview.tabular-q-policy.v1",
      algorithm: "tabular_q_learning",
      target_id: TARGET_ID,
      q_values: qValues,
    },
    evaluation: null,
  };
}

function createEpisode() {
  return {
    schema_version: "world2work.navigation-episode.v1",
    episode_id: null,
    task_id: "corgi_cafe_deliver_coffee_table_7",
    instruction: "Deliver the coffee to table_7.",
    target_id: TARGET_ID,
    robot: {
      name: "NVIDIA Nova Carter",
      payload: "coffee_tray_fixed",
    },
    environment: {
      world_provider: "World Labs Marble",
      layout_source: "Marble-derived occupancy grid",
      grid_artifact: `${ARTIFACT_BASE}/occupancy_grid.json`,
      cell_size_m: artifactData?.meta?.cell_size_m ?? 0.35,
    },
    policy: {
      algorithm: "tabular_q_learning",
      layer: "high_level_grid_navigation",
      source: `${ARTIFACT_BASE}/policy_table_7.json`,
    },
    replay: {
      mode: "kinematic_browser_replay",
      physics_simulation_claimed: false,
      rigid_body_dynamics_claimed: false,
      note: "Browser animation replays the learned cell route; it is not an Isaac Sim physics result.",
    },
    started_at: null,
    completed_at: null,
    steps: [],
    outcome: null,
  };
}

async function fetchJson(name) {
  const response = await fetch(`${ARTIFACT_BASE}/${name}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response.json();
}

function validGrid(value) {
  return Boolean(
    value &&
    Number.isInteger(value.rows) &&
    Number.isInteger(value.cols) &&
    Array.isArray(value.cells) &&
    value.cells.length === value.rows &&
    value.cells.every((row) => Array.isArray(row) && row.length === value.cols),
  );
}

function validRoute(value, grid) {
  return Boolean(
    value && value.target_id === TARGET_ID && value.success !== false && Array.isArray(value.cells) &&
    value.cells.length > 1 && value.cells.every(([row, col]) =>
      Number.isInteger(row) && Number.isInteger(col) && row >= 0 && col >= 0 && row < grid.rows && col < grid.cols),
  );
}

async function loadArtifacts() {
  const fallback = createFallbackArtifacts();
  const jobs = [
    ["grid", "occupancy_grid.json"],
    ["meta", "grid_meta.json"],
    ["route", "route_table_7.json"],
    ["curve", "learning_curve.json"],
    ["policy", "policy_table_7.json"],
    ["evaluation", "evaluation_summary.json"],
  ];
  const settled = await Promise.allSettled(jobs.map(([, file]) => fetchJson(file)));
  const next = { ...fallback };

  settled.forEach((result, index) => {
    const [key] = jobs[index];
    if (result.status === "fulfilled") {
      next[key] = result.value;
      artifactSources[key] = true;
    }
  });

  if (!validGrid(next.grid)) {
    next.grid = fallback.grid;
    artifactSources.grid = false;
  }
  if (!validRoute(next.route, next.grid)) {
    next.route = fallback.route;
    artifactSources.route = false;
  }
  if (!next.meta || !Number.isFinite(Number(next.meta.cell_size_m))) {
    next.meta = fallback.meta;
    artifactSources.meta = false;
  }
  if (!Array.isArray(next.curve?.rolling) && !Array.isArray(next.curve?.episodes)) {
    next.curve = fallback.curve;
    artifactSources.curve = false;
  }
  if (!next.policy || typeof next.policy.q_values !== "object") {
    next.policy = fallback.policy;
    artifactSources.policy = false;
  }
  if (!validEvaluation(next.evaluation)) {
    next.evaluation = null;
    artifactSources.evaluation = false;
  }

  artifactData = next;
  currentEpisode = createEpisode();
  updateArtifactLabels();
  updateMetrics();
  resetEpisode({ quiet: true });
  drawAll();
}

function updateArtifactLabels() {
  const loaded = Object.values(artifactSources).filter(Boolean).length;
  elements.artifactStatus.textContent = loaded === 6 ? "6/6 LOCAL ARTIFACTS" : `${loaded}/6 ARTIFACTS · PARTIAL`;
  elements.liveDot.classList.toggle("is-live", loaded === 6);

  elements.gridSourcePill.classList.remove("is-loaded", "is-fallback");
  elements.gridSourcePill.classList.add(artifactSources.grid ? "is-loaded" : "is-fallback");
  elements.gridSourcePill.innerHTML = artifactSources.grid
    ? `<i></i> occupancy_grid.json · ${artifactData.grid.rows}×${artifactData.grid.cols}`
    : "<i></i> preview grid · artifact missing";

  setMiniStatus(
    elements.curveStatus,
    artifactSources.curve ? "learning_curve.json" : "preview curve",
    artifactSources.curve,
  );
  setMiniStatus(
    elements.heatmapStatus,
    artifactSources.policy ? "policy q_values" : "preview values",
    artifactSources.policy,
  );
}

function setMiniStatus(element, label, loaded) {
  element.textContent = label;
  element.classList.remove("is-loaded", "is-fallback");
  element.classList.add(loaded ? "is-loaded" : "is-fallback");
}

function updateMetrics() {
  const routeLength = Number(artifactData.route.path_length_m || 0);
  elements.successDistance.textContent = Number.isFinite(routeLength) ? `${routeLength.toFixed(2)} m` : "route complete";

  const evaluation = artifactData.evaluation;
  const untrained = evaluation?.policies?.untrained_random?.metrics;
  const trained = evaluation?.policies?.trained_greedy?.metrics;
  if (!validEvaluation(evaluation)) {
    elements.successMetric.textContent = "—";
    elements.collisionMetric.textContent = "—";
    elements.efficiencyMetric.textContent = "—";
    elements.untrainedScore.textContent = "—";
    elements.trainedScore.textContent = "—";
    elements.untrainedBar.style.width = "0";
    elements.trainedBar.style.width = "0";
    setMiniStatus(elements.evaluationStatus, "evaluation pending", false);
    elements.evaluationNote.textContent = "Waiting for evaluation_summary.json; no placeholder scores are shown.";
    return;
  }

  elements.successMetric.textContent = formatRate(trained.success_rate);
  elements.collisionMetric.textContent = formatRate(trained.collision_rate);
  elements.efficiencyMetric.textContent = formatRate(trained.path_efficiency);
  elements.untrainedScore.textContent = formatRate(untrained.success_rate);
  elements.trainedScore.textContent = formatRate(trained.success_rate);
  elements.untrainedBar.style.width = `${rateToPercent(untrained.success_rate)}%`;
  elements.trainedBar.style.width = `${rateToPercent(trained.success_rate)}%`;
  setMiniStatus(elements.evaluationStatus, "evaluation_summary.json", true);
  const episodes = Number(evaluation.holdout?.episodes ?? trained.total_episodes);
  elements.evaluationNote.textContent = Number.isFinite(episodes)
    ? `${episodes} held-out start/orientation configurations · path efficiency is evaluated over successful episodes only.`
    : "Held-out evaluation metrics loaded from the local artifact.";
}

function validEvaluation(value) {
  const untrained = value?.policies?.untrained_random?.metrics;
  const trained = value?.policies?.trained_greedy?.metrics;
  return Boolean(
    value && untrained && trained &&
    Number.isFinite(Number(untrained.success_rate)) &&
    Number.isFinite(Number(trained.success_rate)) &&
    Number.isFinite(Number(trained.collision_rate)) &&
    (trained.path_efficiency === null || Number.isFinite(Number(trained.path_efficiency))),
  );
}

function rateToPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(100, number <= 1 ? number * 100 : number));
}

function formatRate(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  const percent = rateToPercent(value);
  return `${percent.toFixed(percent > 0 && percent < 10 ? 1 : 0)}%`;
}

function curvePoints() {
  if (Array.isArray(artifactData.curve.rolling) && artifactData.curve.rolling.length) {
    return artifactData.curve.rolling.map((point, index) => ({
      episode: Number(point.episode ?? index + 1),
      reward: Number(point.mean_reward_fixed_start ?? point.mean_reward ?? point.reward ?? 0),
      success: Number(point.success_rate_fixed_start ?? point.success_rate ?? point.success ?? 0),
    }));
  }

  const episodes = Array.isArray(artifactData.curve.episodes) ? artifactData.curve.episodes : [];
  const windowSize = Math.max(1, Math.floor(episodes.length / 80));
  const points = [];
  for (let end = windowSize; end <= episodes.length; end += windowSize) {
    const window = episodes.slice(Math.max(0, end - windowSize), end);
    points.push({
      episode: Number(window.at(-1)?.episode ?? end),
      reward: window.reduce((sum, point) => sum + Number(point.reward || 0), 0) / window.length,
      success: window.filter((point) => Boolean(point.success)).length / window.length,
    });
  }
  return points;
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * dpr);
  const pixelHeight = Math.round(height * dpr);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { context, width, height };
}

function mapLayout(width, height, padding = 48) {
  const { rows, cols } = artifactData.grid;
  const cell = Math.min((width - padding * 2) / rows, (height - padding * 1.35) / cols);
  const mapWidth = rows * cell;
  const mapHeight = cols * cell;
  return {
    cell,
    width: mapWidth,
    height: mapHeight,
    x: (width - mapWidth) / 2,
    y: (height - mapHeight) / 2,
  };
}

function cellPoint(cell, layout) {
  const [row, col] = cell;
  return {
    x: layout.x + (row + 0.5) * layout.cell,
    y: layout.y + (artifactData.grid.cols - col - 0.5) * layout.cell,
  };
}

function interpolatedRoute(progress) {
  const cells = artifactData.route.cells;
  const waypoints = Array.isArray(artifactData.route.waypoints_xy_m) && artifactData.route.waypoints_xy_m.length === cells.length
    ? artifactData.route.waypoints_xy_m
    : cells.map(([row, col]) => [
      Number(artifactData.meta.origin_xy_m?.[0] || 0) + (col + 0.5) * Number(artifactData.meta.cell_size_m || 0.35),
      Number(artifactData.meta.origin_xy_m?.[1] || 0) + (row + 0.5) * Number(artifactData.meta.cell_size_m || 0.35),
    ]);
  const scaled = Math.min(1, Math.max(0, progress)) * (cells.length - 1);
  const index = Math.min(cells.length - 2, Math.floor(scaled));
  const mix = scaled - index;
  const fromCell = cells[index];
  const toCell = cells[index + 1];
  const fromWorld = waypoints[index];
  const toWorld = waypoints[index + 1];
  return {
    index,
    mix,
    cell: [fromCell[0] + (toCell[0] - fromCell[0]) * mix, fromCell[1] + (toCell[1] - fromCell[1]) * mix],
    policyCell: mix < 0.5 ? fromCell : toCell,
    world: [fromWorld[0] + (toWorld[0] - fromWorld[0]) * mix, fromWorld[1] + (toWorld[1] - fromWorld[1]) * mix],
    yaw: Math.atan2(toWorld[1] - fromWorld[1], toWorld[0] - fromWorld[0]),
  };
}

function drawAll() {
  drawNavigation();
  drawLearningCurve();
  drawHeatmap();
}

function drawNavigation() {
  const { context: ctx, width, height } = prepareCanvas(elements.navCanvas);
  const layout = mapLayout(width, height);
  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createRadialGradient(width * 0.55, height * 0.44, 20, width * 0.55, height * 0.44, width * 0.72);
  gradient.addColorStop(0, "#151b1d");
  gradient.addColorStop(1, "#090c0e");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,.55)";
  ctx.shadowBlur = 28;
  ctx.fillStyle = "#111619";
  ctx.fillRect(layout.x - 8, layout.y - 8, layout.width + 16, layout.height + 16);
  ctx.restore();

  for (let row = 0; row < artifactData.grid.rows; row += 1) {
    for (let col = 0; col < artifactData.grid.cols; col += 1) {
      const occupied = Number(artifactData.grid.cells[row][col]) === Number(artifactData.grid.occupied_value ?? 1);
      const x = layout.x + row * layout.cell;
      const y = layout.y + (artifactData.grid.cols - col - 1) * layout.cell;
      ctx.fillStyle = occupied ? "#3a4245" : "#172023";
      ctx.fillRect(x + 0.45, y + 0.45, layout.cell - 0.9, layout.cell - 0.9);
      if (occupied) {
        ctx.strokeStyle = "rgba(255,255,255,.035)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x + layout.cell * 0.2, y + layout.cell * 0.8);
        ctx.lineTo(x + layout.cell * 0.8, y + layout.cell * 0.2);
        ctx.stroke();
      }
    }
  }

  const routePoints = artifactData.route.cells.map((cell) => cellPoint(cell, layout));
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.setLineDash([Math.max(3, layout.cell * 0.2), Math.max(4, layout.cell * 0.24)]);
  ctx.strokeStyle = "rgba(241, 138, 60, .46)";
  ctx.lineWidth = Math.max(2, layout.cell * 0.1);
  drawPolyline(ctx, routePoints);
  ctx.stroke();
  ctx.restore();

  const scaled = playback.progress * (routePoints.length - 1);
  const fullIndex = Math.floor(scaled);
  const partial = scaled - fullIndex;
  const traversed = routePoints.slice(0, fullIndex + 1);
  if (fullIndex < routePoints.length - 1) {
    const a = routePoints[fullIndex];
    const b = routePoints[fullIndex + 1];
    traversed.push({ x: a.x + (b.x - a.x) * partial, y: a.y + (b.y - a.y) * partial });
  }
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#f39a50";
  ctx.shadowColor = "rgba(241, 138, 60, .72)";
  ctx.shadowBlur = 12;
  ctx.lineWidth = Math.max(2.5, layout.cell * 0.12);
  drawPolyline(ctx, traversed);
  ctx.stroke();
  ctx.restore();

  const start = routePoints[0];
  const goal = routePoints.at(-1);
  drawMapMarker(ctx, start, "START", "#73b5da", layout.cell);
  drawMapMarker(ctx, goal, "TABLE 7", "#87eaae", layout.cell, true);

  const pose = interpolatedRoute(playback.progress);
  const robotPoint = cellPoint(pose.cell, layout);
  const nextPoint = routePoints[Math.min(routePoints.length - 1, pose.index + 1)];
  const fromPoint = routePoints[pose.index];
  const displayYaw = Math.atan2(nextPoint.y - fromPoint.y, nextPoint.x - fromPoint.x);
  drawCarter(ctx, robotPoint, displayYaw, layout.cell);

  ctx.save();
  ctx.fillStyle = "rgba(255,255,255,.28)";
  ctx.font = `650 ${Math.max(7, Math.min(9, layout.cell * 0.24))}px ${getComputedStyle(document.documentElement).getPropertyValue("--mono")}`;
  ctx.fillText("+Y", layout.x + layout.width + 11, layout.y + layout.height / 2 + 3);
  ctx.beginPath();
  ctx.moveTo(layout.x + layout.width + 8, layout.y + layout.height / 2 - 8);
  ctx.lineTo(layout.x + layout.width + 8, layout.y + layout.height / 2 + 8);
  ctx.strokeStyle = "rgba(255,255,255,.2)";
  ctx.stroke();
  ctx.restore();
}

function drawPolyline(ctx, points) {
  if (!points.length) return;
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let index = 1; index < points.length; index += 1) ctx.lineTo(points[index].x, points[index].y);
}

function drawMapMarker(ctx, point, label, color, cell, pulse = false) {
  ctx.save();
  if (pulse) {
    const glow = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, cell * 0.85);
    glow.addColorStop(0, color.replace(")", ", .22)").replace("rgb", "rgba"));
    glow.addColorStop(1, "rgba(135,234,174,0)");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(point.x, point.y, cell * 0.85, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = "#0f1516";
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point.x, point.y, Math.max(4, cell * 0.18), 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.font = `750 ${Math.max(7, Math.min(10, cell * 0.27))}px ${getComputedStyle(document.documentElement).getPropertyValue("--mono")}`;
  ctx.textAlign = "center";
  ctx.fillText(label, point.x, point.y - cell * 0.48);
  ctx.restore();
}

function roundedRect(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function drawCarter(ctx, point, yaw, cell) {
  const length = Math.max(20, cell * 0.82);
  const width = Math.max(15, cell * 0.58);
  ctx.save();
  ctx.translate(point.x, point.y);
  ctx.rotate(yaw);

  ctx.fillStyle = "rgba(241, 138, 60, .18)";
  ctx.shadowColor = "rgba(241, 138, 60, .55)";
  ctx.shadowBlur = 14;
  ctx.beginPath();
  ctx.ellipse(0, 0, length * 0.72, width * 0.82, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;

  ctx.fillStyle = "#090b0c";
  for (const side of [-1, 1]) {
    roundedRect(ctx, -length * 0.29, side * width * 0.5 - width * 0.12, length * 0.28, width * 0.24, 2);
    ctx.fill();
    roundedRect(ctx, length * 0.05, side * width * 0.5 - width * 0.12, length * 0.28, width * 0.24, 2);
    ctx.fill();
  }

  const bodyGradient = ctx.createLinearGradient(-length / 2, 0, length / 2, 0);
  bodyGradient.addColorStop(0, "#d8d7d1");
  bodyGradient.addColorStop(0.7, "#f1eee5");
  bodyGradient.addColorStop(1, "#a9aaa6");
  ctx.fillStyle = bodyGradient;
  ctx.strokeStyle = "#090c0d";
  ctx.lineWidth = Math.max(1.4, cell * 0.045);
  roundedRect(ctx, -length / 2, -width / 2, length, width, width * 0.22);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#ef8439";
  roundedRect(ctx, -length * 0.18, -width * 0.34, length * 0.43, width * 0.68, width * 0.12);
  ctx.fill();

  ctx.fillStyle = "#20272a";
  ctx.beginPath();
  ctx.arc(length * 0.28, 0, width * 0.19, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#8ed7f1";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.arc(length * 0.28, 0, width * 0.11, 0, Math.PI * 2);
  ctx.stroke();

  // Tray and coffee payload are kinematically attached for this replay.
  ctx.fillStyle = "#6d4a32";
  ctx.strokeStyle = "#17110d";
  ctx.lineWidth = 1;
  roundedRect(ctx, -length * 0.34, -width * 0.28, length * 0.34, width * 0.56, 3);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f5ece0";
  ctx.beginPath();
  ctx.arc(-length * 0.2, 0, width * 0.13, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#51301f";
  ctx.beginPath();
  ctx.arc(-length * 0.2, 0, width * 0.075, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.fillStyle = "rgba(245,238,228,.72)";
  ctx.font = `750 ${Math.max(7, Math.min(9, cell * 0.24))}px ${getComputedStyle(document.documentElement).getPropertyValue("--mono")}`;
  ctx.textAlign = "center";
  ctx.fillText("NOVA CARTER + COFFEE", point.x, point.y + cell * 0.72);
  ctx.restore();
}

function drawLearningCurve() {
  const { context: ctx, width, height } = prepareCanvas(elements.curveCanvas);
  const points = curvePoints();
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101416";
  ctx.fillRect(0, 0, width, height);

  const margin = { left: 34, right: 16, top: 25, bottom: 23 };
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const rewardValues = points.map((point) => point.reward).filter(Number.isFinite);
  const rewardMin = Math.min(...rewardValues, 0);
  const rewardMax = Math.max(...rewardValues, 1);
  const episodeMax = Math.max(...points.map((point) => point.episode), 1);

  ctx.save();
  ctx.font = `600 7px ${getComputedStyle(document.documentElement).getPropertyValue("--mono")}`;
  ctx.fillStyle = "#616566";
  ctx.strokeStyle = "rgba(255,255,255,.065)";
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const y = margin.top + chartHeight * tick / 4;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(width - margin.right, y);
    ctx.stroke();
    const rewardLabel = rewardMax - (rewardMax - rewardMin) * tick / 4;
    ctx.textAlign = "right";
    ctx.fillText(rewardLabel.toFixed(0), margin.left - 7, y + 3);
  }
  ctx.textAlign = "left";
  ctx.fillText("0", margin.left, height - 7);
  ctx.textAlign = "right";
  ctx.fillText(formatEpisode(episodeMax), width - margin.right, height - 7);
  ctx.restore();

  const toPoint = (point, key) => ({
    x: margin.left + point.episode / episodeMax * chartWidth,
    y: key === "success"
      ? margin.top + (1 - Math.min(1, Math.max(0, point.success))) * chartHeight
      : margin.top + (1 - (point.reward - rewardMin) / Math.max(0.001, rewardMax - rewardMin)) * chartHeight,
  });

  const successArea = points.map((point) => toPoint(point, "success"));
  if (successArea.length) {
    const fill = ctx.createLinearGradient(0, margin.top, 0, margin.top + chartHeight);
    fill.addColorStop(0, "rgba(135,234,174,.16)");
    fill.addColorStop(1, "rgba(135,234,174,0)");
    ctx.beginPath();
    ctx.moveTo(successArea[0].x, margin.top + chartHeight);
    successArea.forEach((point) => ctx.lineTo(point.x, point.y));
    ctx.lineTo(successArea.at(-1).x, margin.top + chartHeight);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  }

  drawChartLine(ctx, points.map((point) => toPoint(point, "reward")), "#f18a3c", 1.7);
  drawChartLine(ctx, successArea, "#87eaae", 1.7);
}

function drawChartLine(ctx, points, color, width) {
  if (!points.length) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  drawPolyline(ctx, points);
  ctx.stroke();
  const end = points.at(-1);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(end.x, end.y, 2.7, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function formatEpisode(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 1)}k eps` : `${value} eps`;
}

function drawHeatmap() {
  const { context: ctx, width, height } = prepareCanvas(elements.heatmapCanvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101416";
  ctx.fillRect(0, 0, width, height);
  const layout = mapLayout(width, height, 17);
  const qValues = artifactData.policy.q_values || {};
  const values = Object.values(qValues)
    .map((actions) => Array.isArray(actions) ? Math.max(...actions.map(Number).filter(Number.isFinite)) : Number(actions))
    .filter(Number.isFinite);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);

  for (let row = 0; row < artifactData.grid.rows; row += 1) {
    for (let col = 0; col < artifactData.grid.cols; col += 1) {
      const occupied = Number(artifactData.grid.cells[row][col]) === Number(artifactData.grid.occupied_value ?? 1);
      const x = layout.x + row * layout.cell;
      const y = layout.y + (artifactData.grid.cols - col - 1) * layout.cell;
      if (occupied) {
        ctx.fillStyle = "#252b2e";
      } else {
        const raw = qValues[`${row},${col}`];
        const value = Array.isArray(raw) ? Math.max(...raw.map(Number).filter(Number.isFinite)) : Number(raw);
        const normalized = Number.isFinite(value) ? (value - min) / Math.max(0.001, max - min) : 0;
        ctx.fillStyle = heatColor(normalized);
      }
      ctx.fillRect(x + 0.35, y + 0.35, layout.cell - 0.7, layout.cell - 0.7);
    }
  }

  const routePoints = artifactData.route.cells.map((cell) => cellPoint(cell, layout));
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,.45)";
  ctx.lineWidth = Math.max(1, layout.cell * 0.07);
  ctx.setLineDash([2, 3]);
  drawPolyline(ctx, routePoints);
  ctx.stroke();
  ctx.restore();

  const goal = routePoints.at(-1);
  ctx.fillStyle = "#eafff2";
  ctx.beginPath();
  ctx.arc(goal.x, goal.y, Math.max(2.5, layout.cell * 0.15), 0, Math.PI * 2);
  ctx.fill();
}

function heatColor(t) {
  const value = Math.max(0, Math.min(1, t));
  if (value < 0.55) {
    const mix = value / 0.55;
    return `rgb(${Math.round(20 + 105 * mix)}, ${Math.round(31 + 42 * mix)}, ${Math.round(39 + 14 * mix)})`;
  }
  const mix = (value - 0.55) / 0.45;
  return `rgb(${Math.round(125 + 22 * mix)}, ${Math.round(73 + 165 * mix)}, ${Math.round(53 + 133 * mix)})`;
}

function setPhase(active) {
  const activeIndex = phaseOrder.indexOf(active);
  phases.forEach((phase) => {
    const index = phaseOrder.indexOf(phase.dataset.phase);
    phase.classList.toggle("is-active", index === activeIndex);
    phase.classList.toggle("is-done", index < activeIndex);
  });
}

function setHud(phase, progress, cell) {
  elements.hudPhase.textContent = phase.toUpperCase();
  elements.hudProgress.textContent = `${Math.round(progress * 100)}%`;
  elements.hudCell.textContent = cell ? `[${Math.round(cell[0])}, ${Math.round(cell[1])}]` : "—";
  elements.hudPhaseDot.style.background = phase === "arrived" ? "#87eaae" : "#f18a3c";
}

function resetEpisode(options = {}) {
  playback.token += 1;
  cancelAnimationFrame(playback.animationFrame);
  window.clearTimeout(playback.animationFrame);
  playback.running = false;
  playback.progress = 0;
  playback.lastSampleAt = 0;
  playback.lastCellIndex = -1;
  currentEpisode = createEpisode();
  elements.runButton.disabled = false;
  elements.exportButton.disabled = true;
  elements.successCard.classList.remove("is-visible");
  elements.eventLog.innerHTML = '<p class="empty-log">Run delivery to stream navigation frames.</p>';
  elements.eventCount.textContent = "0 frames";
  elements.fieldEpisode.textContent = "pending";
  elements.fieldPose.textContent = "[—, —, —]";
  elements.fieldWheels.textContent = "[0.00, 0.00]";
  elements.fieldCell.textContent = "[—, —]";
  elements.fieldCommand.textContent = "[0.00, 0.00]";
  elements.fieldDone.textContent = "false";
  elements.fieldDone.classList.remove("is-success");
  setPhase("load");
  setHud("ready", 0, artifactData.route.cells[0]);
  drawNavigation();
  if (!options.quiet) showToast("Episode reset to the learned route start.");
}

function delay(milliseconds, token) {
  return new Promise((resolve) => {
    window.setTimeout(() => resolve(token === playback.token), milliseconds);
  });
}

async function runEpisode() {
  if (playback.running) return;
  resetEpisode({ quiet: true });
  playback.running = true;
  const token = ++playback.token;
  elements.runButton.disabled = true;
  currentEpisode = createEpisode();
  currentEpisode.episode_id = `w2w_${Date.now().toString(36)}`;
  currentEpisode.started_at = new Date().toISOString();
  elements.fieldEpisode.textContent = currentEpisode.episode_id;

  setPhase("load");
  setHud("loading world", 0, artifactData.route.cells[0]);
  addEventLog("0.00s", "Marble occupancy loaded");
  if (!(await delay(420, token))) return;

  setPhase("policy");
  setHud("reading policy", 0, artifactData.route.cells[0]);
  addEventLog("0.42s", "Greedy table_7 route selected");
  if (!(await delay(460, token))) return;

  setPhase("navigate");
  setHud("navigating", 0, artifactData.route.cells[0]);
  addEventLog("0.88s", "Nova Carter replay started");
  const duration = Math.max(6500, Math.min(10500, Number(artifactData.route.path_length_m || 8) * 1050));
  const start = performance.now();

  function tick(now) {
    if (token !== playback.token) return;
    const elapsed = now - start;
    playback.progress = Math.min(1, elapsed / duration);
    const pose = interpolatedRoute(playback.progress);
    updateLiveFields(pose, elapsed, false);
    recordStep(pose, elapsed, false);

    if (pose.index !== playback.lastCellIndex) {
      playback.lastCellIndex = pose.index;
      if (pose.index > 0 && pose.index % 4 === 0) {
        addEventLog(`${((elapsed + 880) / 1000).toFixed(2)}s`, `Policy cell [${pose.policyCell.join(", ")}]`);
      }
    }
    setHud("navigating", playback.progress, pose.policyCell);
    drawNavigation();

    if (playback.progress < 1) {
      playback.animationFrame = window.setTimeout(() => tick(performance.now()), 32);
    } else {
      finishEpisode(pose, elapsed);
    }
  }

  playback.animationFrame = window.setTimeout(() => tick(performance.now()), 0);
}

function updateLiveFields(pose, elapsed, done) {
  const nextIndex = Math.min(artifactData.route.cells.length - 1, pose.index + 1);
  const nextCell = artifactData.route.cells[nextIndex];
  const currentCell = artifactData.route.cells[pose.index];
  const turn = nextCell[0] !== currentCell[0] && nextCell[1] !== currentCell[1] ? 0.35 : 0;
  const linear = done ? 0 : 0.35;
  const wheel = done ? [0, 0] : [2.5 - turn, 2.5 + turn];
  elements.fieldPose.textContent = `[${pose.world[0].toFixed(2)}, ${pose.world[1].toFixed(2)}, ${pose.yaw.toFixed(2)}]`;
  elements.fieldWheels.textContent = `[${wheel[0].toFixed(2)}, ${wheel[1].toFixed(2)}]`;
  elements.fieldCell.textContent = `[${pose.policyCell.join(", ")}]`;
  elements.fieldCommand.textContent = `[${linear.toFixed(2)}, ${turn.toFixed(2)}]`;
  elements.fieldDone.textContent = String(done);
  elements.eventCount.textContent = `${currentEpisode.steps.length} frames`;
}

function recordStep(pose, elapsed, done, force = false) {
  if (!force && elapsed - playback.lastSampleAt < 100) return;
  playback.lastSampleAt = elapsed;
  const cellSize = Number(artifactData.meta.cell_size_m || 0.35);
  const nextIndex = Math.min(artifactData.route.cells.length - 1, pose.index + 1);
  const a = artifactData.route.cells[pose.index];
  const b = artifactData.route.cells[nextIndex];
  const changedAxis = a[0] !== b[0] || a[1] !== b[1];
  const linear = done ? 0 : 0.35;
  const angular = done || !changedAxis ? 0 : 0;
  const wheels = done ? [0, 0] : [2.5, 2.5];
  currentEpisode.steps.push({
    t: Number((elapsed / 1000).toFixed(3)),
    phase: done ? "arrive" : "navigate",
    base_pose_xyyaw_m: [
      Number(pose.world[0].toFixed(4)),
      Number(pose.world[1].toFixed(4)),
      Number(pose.yaw.toFixed(4)),
    ],
    wheel_velocity_rad_s: wheels,
    command: { linear_mps: linear, angular_rps: angular },
    policy_cell: pose.policyCell.map(Math.round),
    payload_pose_xyz_m: [Number(pose.world[0].toFixed(4)), Number(pose.world[1].toFixed(4)), 0.92],
    target_pose_xy_m: artifactData.route.target?.xy_m || artifactData.meta.targets?.table_7?.xy_m || artifactData.route.waypoints_xy_m?.at(-1),
    reward: done ? 10 : -0.01 * cellSize,
    done,
    provenance: REPLAY_LABEL,
  });
}

function finishEpisode(pose, elapsed) {
  recordStep(pose, elapsed, true, true);
  playback.running = false;
  setPhase("arrive");
  setHud("arrived", 1, artifactData.route.cells.at(-1));
  updateLiveFields(pose, elapsed, true);
  currentEpisode.completed_at = new Date().toISOString();
  currentEpisode.outcome = {
    success: true,
    reason: "learned_route_reached_goal_cell",
    target_id: TARGET_ID,
    final_policy_cell: artifactData.route.cells.at(-1),
    path_length_m: Number(artifactData.route.path_length_m || 0),
    route_steps: Number(artifactData.route.steps || artifactData.route.cells.length - 1),
    browser_replay: true,
    physics_simulation_claimed: false,
  };
  elements.fieldDone.textContent = "true";
  elements.fieldDone.classList.add("is-success");
  elements.eventCount.textContent = `${currentEpisode.steps.length} frames`;
  elements.successCard.classList.add("is-visible");
  elements.runButton.disabled = false;
  elements.exportButton.disabled = false;
  addEventLog(`${((elapsed + 880) / 1000).toFixed(2)}s`, "table_7 reached · success", true);
  drawNavigation();
}

function addEventLog(time, label, success = false) {
  elements.eventLog.querySelector(".empty-log")?.remove();
  const row = document.createElement("div");
  row.className = `log-row${success ? " success" : ""}`;
  const timeNode = document.createElement("time");
  timeNode.textContent = time;
  const dot = document.createElement("i");
  const body = document.createElement("span");
  body.textContent = label;
  row.append(timeNode, dot, body);
  elements.eventLog.append(row);
  elements.eventLog.scrollTop = elements.eventLog.scrollHeight;
}

function exportEpisode() {
  if (!currentEpisode.outcome) return;
  const payload = {
    ...currentEpisode,
    artifact_availability: { ...artifactSources },
    artifact_contract: {
      occupancy: `${ARTIFACT_BASE}/occupancy_grid.json`,
      grid_meta: `${ARTIFACT_BASE}/grid_meta.json`,
      policy: `${ARTIFACT_BASE}/policy_table_7.json`,
      route: `${ARTIFACT_BASE}/route_table_7.json`,
      learning_curve: `${ARTIFACT_BASE}/learning_curve.json`,
      value_heatmap: `${ARTIFACT_BASE}/value_heatmap.png`,
      evaluation: `${ARTIFACT_BASE}/evaluation_summary.json`,
    },
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${currentEpisode.episode_id}.json`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  showToast("Episode JSON exported with replay provenance.");
}

function showToast(message) {
  window.clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  toastTimer = window.setTimeout(() => elements.toast.classList.remove("is-visible"), 2200);
}

function setupImageFallback(image, label) {
  image.addEventListener("error", () => {
    const safeLabel = label.replace(/[<>&]/g, "");
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="500"><rect width="100%" height="100%" fill="#171b1e"/><path d="M0 360L260 180l190 130 240-190 510 300v80H0z" fill="#252b2e"/><text x="50%" y="46%" fill="#a6a19a" font-family="monospace" font-size="26" text-anchor="middle">${safeLabel}</text><text x="50%" y="56%" fill="#686c6d" font-family="monospace" font-size="16" text-anchor="middle">local image artifact unavailable</text></svg>`;
    image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
  }, { once: true });
}

function setRolloutStatus(label, mode = "checking") {
  elements.rolloutStatus.classList.remove("is-loaded", "is-fallback");
  if (mode === "loaded") elements.rolloutStatus.classList.add("is-loaded");
  if (mode === "missing") elements.rolloutStatus.classList.add("is-fallback");
  elements.rolloutStatus.innerHTML = `<i></i> ${label}`;
}

function waitForVideoMetadata(video, token) {
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", loaded);
      video.removeEventListener("error", failed);
      window.clearTimeout(timeout);
    };
    const loaded = () => {
      cleanup();
      if (token === rolloutLoadToken) resolve();
      else reject(new Error("superseded rollout request"));
    };
    const failed = () => {
      cleanup();
      reject(new Error("rollout video could not be decoded"));
    };
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("rollout metadata timed out"));
    }, 8000);
    video.addEventListener("loadedmetadata", loaded);
    video.addEventListener("error", failed);
  });
}

async function loadRolloutVideo() {
  const token = ++rolloutLoadToken;
  elements.rolloutPanel.classList.remove("is-ready");
  elements.rolloutFallbackTitle.textContent = "Checking for Isaac Sim rollout…";
  elements.rolloutFallbackText.innerHTML = "Looking for <code>web-demo/assets/world2work-demo.mp4</code>.";
  setRolloutStatus("checking local MP4");
  elements.retryRolloutButton.disabled = true;

  try {
    const response = await fetch(ROLLOUT_VIDEO_URL, { method: "HEAD", cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    if (token !== rolloutLoadToken) return;
    const metadataReady = waitForVideoMetadata(elements.rolloutVideo, token);
    elements.rolloutVideo.src = `${ROLLOUT_VIDEO_URL}?v=${Date.now()}`;
    elements.rolloutVideo.load();
    await metadataReady;
    if (token !== rolloutLoadToken) return;
    elements.rolloutPanel.classList.add("is-ready");
    setRolloutStatus("world2work-demo.mp4 · ready", "loaded");
  } catch (error) {
    if (token !== rolloutLoadToken) return;
    elements.rolloutVideo.pause();
    elements.rolloutVideo.removeAttribute("src");
    elements.rolloutVideo.load();
    elements.rolloutFallbackTitle.textContent = "Isaac Sim rollout is not available yet";
    elements.rolloutFallbackText.innerHTML = "Place the rendered video at <code>web-demo/assets/world2work-demo.mp4</code>, then retry. The kinematic browser replay above remains available.";
    setRolloutStatus("local MP4 pending", "missing");
    console.info("Actual Isaac Sim rollout card is using its local-file fallback.", error);
  } finally {
    if (token === rolloutLoadToken) elements.retryRolloutButton.disabled = false;
  }
}

function tickClock() {
  elements.clock.textContent = new Date().toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

elements.runButton.addEventListener("click", runEpisode);
elements.replayButton.addEventListener("click", runEpisode);
elements.resetButton.addEventListener("click", () => resetEpisode());
elements.exportButton.addEventListener("click", exportEpisode);
elements.retryRolloutButton.addEventListener("click", loadRolloutVideo);
setupImageFallback(elements.sourcePhoto, "SOURCE PHOTO");
setupImageFallback(elements.marblePano, "MARBLE PANORAMA");

const resizeObserver = new ResizeObserver(() => drawAll());
resizeObserver.observe(elements.navCanvas);
resizeObserver.observe(elements.curveCanvas);
resizeObserver.observe(elements.heatmapCanvas);

tickClock();
window.setInterval(tickClock, 1000);
updateArtifactLabels();
updateMetrics();
resetEpisode({ quiet: true });
drawAll();
loadRolloutVideo();
loadArtifacts().catch((error) => {
  console.warn("Artifact loader retained its explicit preview fallback.", error);
  updateArtifactLabels();
  drawAll();
});

window.world2WorkDemo = {
  run: runEpisode,
  reset: resetEpisode,
  getEpisode: () => JSON.parse(JSON.stringify(currentEpisode)),
  getArtifacts: () => JSON.parse(JSON.stringify(artifactData)),
  claims: { policy: POLICY_LABEL, replay: REPLAY_LABEL },
};
