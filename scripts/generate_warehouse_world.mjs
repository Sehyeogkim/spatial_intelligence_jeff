#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const API_ROOT = "https://api.worldlabs.ai/marble/v1";
const imagePath = process.argv[2] ?? "artifacts/warehouse/source-empty-storage.jpg";
const outputRoot = process.argv[3] ?? "artifacts/warehouse/world";
const startPath = path.join(outputRoot, "start.json");

if (!fs.existsSync(imagePath)) throw new Error(`Source image not found: ${imagePath}`);
if (fs.existsSync(startPath)) {
  throw new Error(`${startPath} already exists. Refusing to start a duplicate paid generation.`);
}

const env = fs.readFileSync(".env", "utf8");
const keyLine = env.split(/\r?\n/).find((line) => /^WORLDLAB_API_KEY\s*=/.test(line));
let apiKey = keyLine?.replace(/^WORLDLAB_API_KEY\s*=\s*/, "").trim() ?? "";
if (
  (apiKey.startsWith('"') && apiKey.endsWith('"')) ||
  (apiKey.startsWith("'") && apiKey.endsWith("'"))
) {
  apiKey = apiKey.slice(1, -1);
}
if (!apiKey) throw new Error("WORLDLAB_API_KEY is missing or empty");

async function apiPost(endpoint, payload) {
  const response = await fetch(`${API_ROOT}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "WLT-Api-Key": apiKey },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(
      `${endpoint}: ${response.status} ${response.statusText}: ${await response.text()}`,
    );
  }
  return { status: response.status, body: await response.json() };
}

fs.mkdirSync(outputRoot, { recursive: true });

const preparePayload = {
  file_name: path.basename(imagePath).slice(0, 64),
  kind: "image",
  extension: "jpg",
  metadata: {
    project: "dead-data-living-worlds",
    source: "PxHere CC0 photo by Michel Toulouse",
    source_url: "https://pxhere.com/en/photo/1598891",
  },
};
const prepared = await apiPost("/media-assets:prepare_upload", preparePayload);
const mediaAssetId =
  prepared.body.media_asset.media_asset_id ?? prepared.body.media_asset.id;
const uploadInfo = prepared.body.upload_info;
if (!mediaAssetId || !uploadInfo?.upload_url) {
  throw new Error("Prepare-upload response did not include the expected IDs or URL");
}

const uploadResponse = await fetch(uploadInfo.upload_url, {
  method: uploadInfo.upload_method ?? "PUT",
  headers: uploadInfo.required_headers ?? {},
  body: fs.readFileSync(imagePath),
});
if (!uploadResponse.ok) {
  throw new Error(
    `Image upload failed: ${uploadResponse.status} ${uploadResponse.statusText}: ${await uploadResponse.text()}`,
  );
}

const textPrompt = [
  "Reconstruct this existing empty indoor storage warehouse as one coherent, photorealistic 3D space.",
  "Preserve the symmetrical blue steel rack structure, orange horizontal beams, concrete floor, high metal ceiling, bright industrial lighting, aisle-number signs, and long unobstructed passages.",
  "Maintain the original aisle widths, rack spacing, level floor, consistent metric scale, and strong central depth.",
  "Extend the warehouse architecture naturally beyond the frame while keeping the foreground floor open and preserving clear connected routes through the left and right aisles.",
].join(" ");

const generationRequest = {
  display_name: "Empty Rack Warehouse",
  model: "marble-1.1",
  seed: 20260905,
  permission: { public: false },
  tags: ["hackathon", "physical-ai", "warehouse", "navigation"],
  world_prompt: {
    type: "image",
    image_prompt: { source: "media_asset", media_asset_id: mediaAssetId },
    is_pano: false,
    text_prompt: textPrompt,
  },
};

fs.writeFileSync(
  path.join(outputRoot, "media-asset.json"),
  JSON.stringify(
    {
      request: preparePayload,
      response: prepared.body.media_asset,
      uploaded_at: new Date().toISOString(),
    },
    null,
    2,
  ),
);
fs.writeFileSync(
  path.join(outputRoot, "generation-request.json"),
  JSON.stringify(generationRequest, null, 2),
);

const generated = await apiPost("/worlds:generate", generationRequest);
const start = {
  request: generationRequest,
  http_status: generated.status,
  response: generated.body,
  started_at: new Date().toISOString(),
};
fs.writeFileSync(startPath, JSON.stringify(start, null, 2));

console.log(
  JSON.stringify(
    {
      started: true,
      operationId: generated.body.operation_id,
      mediaAssetId,
      sourceImage: imagePath,
      outputRoot,
    },
    null,
    2,
  ),
);
