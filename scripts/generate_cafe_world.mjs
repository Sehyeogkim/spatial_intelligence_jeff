#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const API_ROOT = "https://api.worldlabs.ai/marble/v1";
const imagePath = process.argv[2] ?? "artifacts/corgi-cafe/source-google-maps.jpg";
const outputRoot = process.argv[3] ?? "artifacts/corgi-cafe";
const startPath = path.join(outputRoot, "start.json");

if (!fs.existsSync(imagePath)) {
  throw new Error(`Source image not found: ${imagePath}`);
}
if (fs.existsSync(startPath)) {
  throw new Error(
    `${startPath} already exists. Refusing to start a duplicate paid generation.`,
  );
}

const env = fs.readFileSync(".env", "utf8");
const keyLine = env
  .split(/\r?\n/)
  .find((line) => /^WORLDLAB_API_KEY\s*=/.test(line));
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
    headers: {
      "Content-Type": "application/json",
      "WLT-Api-Key": apiKey,
    },
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

const fileName = path.basename(imagePath);
const preparePayload = {
  file_name: fileName.slice(0, 64),
  kind: "image",
  extension: "jpg",
  metadata: {
    project: "corgi-cafe-robot-data-demo",
    source: "Google Maps user photo",
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
  "Reconstruct this bright modern San Francisco Corgi Cafe interior as a clean, static robot-training environment.",
  "Preserve the rectangular room, light wood floor, large windows on the left, orange ceiling accents, right-side service counter, shelves, columns, and the visual style of the orange and wood tables and chairs.",
  "Remove every person, laptop, bag, loose cup, trash bin, QR code, legible sign, and personal belonging.",
  "Keep the main aisle from the service counter to the front-center area wide, level, and unobstructed.",
  "Create two clearly isolated orange serving zones on a clean counter-height work surface near the front-center, both with empty tops and clear space for a Franka robot arm.",
  "Use realistic human metric scale, a perfectly level continuous floor, solid non-floating furniture geometry, stable bright lighting, and no robot or movable coffee cup baked into the scene.",
  "The environment is static; the robot, collision-safe work surface, target markers, and coffee cup will be added separately in the simulator.",
].join(" ");

const generationRequest = {
  display_name: "Corgi Cafe Robot Training World",
  model: "marble-1.1",
  seed: 20260905,
  permission: { public: false },
  tags: ["hackathon", "robotics", "cafe", "corgi-cafe"],
  world_prompt: {
    type: "image",
    image_prompt: {
      source: "media_asset",
      media_asset_id: mediaAssetId,
    },
    is_pano: false,
    disable_recaption: true,
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
