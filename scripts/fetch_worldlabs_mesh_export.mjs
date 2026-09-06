#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const root = process.argv[2] ?? "artifacts/worldlabs-audit";
const env = fs.readFileSync(".env", "utf8");
const keyLine = env.split(/\r?\n/).find((line) => /^WORLDLAB_API_KEY\s*=/.test(line));
let apiKey = keyLine?.replace(/^WORLDLAB_API_KEY\s*=\s*/, "").trim() ?? "";
if ((apiKey.startsWith('"') && apiKey.endsWith('"')) || (apiKey.startsWith("'") && apiKey.endsWith("'"))) apiKey = apiKey.slice(1, -1);
if (!apiKey) throw new Error("WORLDLAB_API_KEY is missing or empty");

const start = JSON.parse(fs.readFileSync(path.join(root, "export-start.json"), "utf8"));
const operationId = start.response.operation_id;
const worldResponse = JSON.parse(fs.readFileSync(path.join(root, "world.json"), "utf8"));
const priorWorld = worldResponse.world ?? worldResponse;
const worldId = priorWorld.id ?? priorWorld.world_id;

async function apiJson(url) {
  const response = await fetch(url, { headers: { "WLT-Api-Key": apiKey } });
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

async function download(url, filename) {
  if (!url) return { filename, skipped: true };
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Download failed for ${filename}: ${response.status}`);
  const output = path.join(root, filename);
  await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(output));
  return { filename, bytes: fs.statSync(output).size };
}

const operation = await apiJson(`https://api.worldlabs.ai/marble/v1/operations/${operationId}`);
fs.writeFileSync(path.join(root, "export-operation.json"), JSON.stringify(operation, null, 2));
if (!operation.done || operation.error) {
  console.log(JSON.stringify({ done: operation.done, progress: operation.metadata?.progress, error: operation.error }, null, 2));
  process.exit(operation.error ? 1 : 2);
}

const latestResponse = await apiJson(`https://api.worldlabs.ai/marble/v1/worlds/${worldId}`);
fs.writeFileSync(path.join(root, "world-after-export.json"), JSON.stringify(latestResponse, null, 2));
const world = latestResponse.world ?? latestResponse;
const downloads = await Promise.all([
  download(world.assets?.mesh?.hq_mesh_url, "hq-mesh.glb"),
  download(world.assets?.mesh?.full_res_mesh_url, "full-res-mesh.glb"),
]);
console.log(JSON.stringify({ done: true, cost: operation.cost, downloads }, null, 2));
