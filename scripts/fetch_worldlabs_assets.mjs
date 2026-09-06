#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { pipeline } from "node:stream/promises";
import { Readable } from "node:stream";

const args = process.argv.slice(2);
const root = args.find((arg) => !arg.startsWith("--")) ?? "artifacts/worldlabs-audit";
const watch = args.includes("--watch");
const pollIntervalMs = 30_000;
const env = fs.readFileSync(".env", "utf8");
const keyLine = env.split(/\r?\n/).find((line) => /^WORLDLAB_API_KEY\s*=/.test(line));
let apiKey = keyLine?.replace(/^WORLDLAB_API_KEY\s*=\s*/, "").trim() ?? "";
if ((apiKey.startsWith('"') && apiKey.endsWith('"')) || (apiKey.startsWith("'") && apiKey.endsWith("'"))) {
  apiKey = apiKey.slice(1, -1);
}
if (!apiKey) throw new Error("WORLDLAB_API_KEY is missing or empty");

const start = JSON.parse(fs.readFileSync(path.join(root, "start.json"), "utf8"));
const operationId = start.response.operation_id;

async function apiJson(url) {
  const response = await fetch(url, { headers: { "WLT-Api-Key": apiKey } });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${await response.text()}`);
  return response.json();
}

async function download(url, filename) {
  if (!url) return { filename, skipped: true };
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Download failed for ${filename}: ${response.status}`);
  const output = path.join(root, filename);
  await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(output));
  const bytes = fs.statSync(output).size;
  return { filename, bytes };
}

fs.mkdirSync(root, { recursive: true });
let operation;
while (true) {
  operation = await apiJson(`https://api.worldlabs.ai/marble/v1/operations/${operationId}`);
  fs.writeFileSync(path.join(root, "operation.json"), JSON.stringify(operation, null, 2));
  if (operation.done || operation.error) break;

  console.log(JSON.stringify({
    checkedAt: new Date().toISOString(),
    done: operation.done,
    progress: operation.metadata?.progress,
  }));
  if (!watch) process.exit(2);
  await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
}

if (operation.error) {
  console.log(JSON.stringify({ done: operation.done, error: operation.error }, null, 2));
  process.exit(1);
}

const worldId = operation.metadata?.world_id ?? operation.response?.id ?? operation.response?.world_id;
const worldResponse = await apiJson(`https://api.worldlabs.ai/marble/v1/worlds/${worldId}`);
fs.writeFileSync(path.join(root, "world.json"), JSON.stringify(worldResponse, null, 2));
const world = worldResponse.world ?? worldResponse;
const assets = world.assets ?? {};
const spz = assets.splats?.spz_urls ?? {};

const downloads = await Promise.all([
  download(spz["100k"], "splats-100k.spz"),
  download(spz["500k"], "splats-500k.spz"),
  download(spz.full_res, "splats-full-res.spz"),
  download(assets.mesh?.collider_mesh_url, "collider.glb"),
  download(assets.mesh?.hq_mesh_url, "hq-mesh.glb"),
  download(assets.mesh?.full_res_mesh_url, "full-res-mesh.glb"),
  download(assets.imagery?.pano_url, "pano.png"),
]);

console.log(JSON.stringify({
  done: true,
  worldId,
  marbleUrl: world.world_marble_url,
  cost: operation.cost,
  downloads,
}, null, 2));
