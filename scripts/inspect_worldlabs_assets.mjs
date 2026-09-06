#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = process.argv[2] ?? "artifacts/worldlabs-audit";

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function fileInfo(file) {
  const stat = fs.statSync(file);
  return { file: path.basename(file), bytes: stat.size, mebibytes: +(stat.size / 1048576).toFixed(3) };
}

function inspectPng(file) {
  const data = fs.readFileSync(file);
  const signature = data.subarray(0, 8).toString("hex");
  if (signature !== "89504e470d0a1a0a") return { ...fileInfo(file), format: "unknown" };
  return {
    ...fileInfo(file),
    format: "png",
    width: data.readUInt32BE(16),
    height: data.readUInt32BE(20),
    aspectRatio: +(data.readUInt32BE(16) / data.readUInt32BE(20)).toFixed(4),
  };
}

function inspectSpz(file) {
  const data = fs.readFileSync(file);
  const base = fileInfo(file);
  if (data.subarray(0, 4).toString("ascii") === "NGSP") {
    const header = {
      magic: "NGSP",
      version: data.readUInt32LE(4),
      numPoints: data.readUInt32LE(8),
      shDegree: data.readUInt8(12),
      fractionalBits: data.readUInt8(13),
      flags: data.readUInt8(14),
      numStreams: data.readUInt8(15),
      tocByteOffset: data.readUInt32LE(16),
    };
    const streamNames = ["positions", "alphas", "colors", "scales", "rotations", "spherical_harmonics"];
    const streams = [];
    for (let i = 0; i < header.numStreams; i += 1) {
      const offset = header.tocByteOffset + i * 16;
      streams.push({
        index: i,
        name: streamNames[i] ?? `stream_${i}`,
        compressedBytes: Number(data.readBigUInt64LE(offset)),
        uncompressedBytes: Number(data.readBigUInt64LE(offset + 8)),
      });
    }
    const extensionBytes = data.subarray(32, header.tocByteOffset);
    return {
      ...base,
      format: "spz-v4-zstd",
      header,
      antialiased: Boolean(header.flags & 0x1),
      hasExtensions: Boolean(header.flags & 0x2),
      extensionZoneBytes: extensionBytes.length,
      extensionAscii: extensionBytes.toString("ascii").replace(/[^\x20-\x7e]+/g, " ").trim() || null,
      streams,
    };
  }
  if (data[0] === 0x1f && data[1] === 0x8b) return { ...base, format: "spz-v1-v3-gzip" };
  return { ...base, format: "unknown", first16Hex: data.subarray(0, 16).toString("hex") };
}

function componentReader(type) {
  switch (type) {
    case 5120: return { bytes: 1, read: (b, o) => b.readInt8(o) };
    case 5121: return { bytes: 1, read: (b, o) => b.readUInt8(o) };
    case 5122: return { bytes: 2, read: (b, o) => b.readInt16LE(o) };
    case 5123: return { bytes: 2, read: (b, o) => b.readUInt16LE(o) };
    case 5125: return { bytes: 4, read: (b, o) => b.readUInt32LE(o) };
    case 5126: return { bytes: 4, read: (b, o) => b.readFloatLE(o) };
    default: throw new Error(`Unsupported glTF componentType ${type}`);
  }
}

const typeSize = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };

function decodeAccessor(gltf, bin, index) {
  const accessor = gltf.accessors[index];
  if (accessor.sparse) throw new Error(`Sparse accessor ${index} is not supported by this audit script`);
  const view = gltf.bufferViews[accessor.bufferView];
  const component = componentReader(accessor.componentType);
  const width = typeSize[accessor.type];
  const stride = view.byteStride ?? width * component.bytes;
  const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const values = new Array(accessor.count);
  for (let i = 0; i < accessor.count; i += 1) {
    const row = new Array(width);
    for (let j = 0; j < width; j += 1) row[j] = component.read(bin, start + i * stride + j * component.bytes);
    values[i] = width === 1 ? row[0] : row;
  }
  return values;
}

function identity() {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      for (let k = 0; k < 4; k += 1) out[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
    }
  }
  return out;
}

function nodeMatrix(node) {
  if (node.matrix) return node.matrix;
  const [x, y, z, w] = node.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = node.scale ?? [1, 1, 1];
  const [tx, ty, tz] = node.translation ?? [0, 0, 0];
  const x2 = x + x; const y2 = y + y; const z2 = z + z;
  const xx = x * x2; const xy = x * y2; const xz = x * z2;
  const yy = y * y2; const yz = y * z2; const zz = z * z2;
  const wx = w * x2; const wy = w * y2; const wz = w * z2;
  return [
    (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
    (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
    (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

function transformPoint(m, p) {
  const [x, y, z] = p;
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

function raycastGrid(triangles, bounds, gridSize = 11) {
  if (!triangles.length) return { supported: false, reason: "no triangle primitives" };
  const hits = [];
  const eps = 1e-8;
  const inset = 0.02;
  for (let iz = 0; iz < gridSize; iz += 1) {
    for (let ix = 0; ix < gridSize; ix += 1) {
      const fx = inset + (1 - inset * 2) * ix / (gridSize - 1);
      const fz = inset + (1 - inset * 2) * iz / (gridSize - 1);
      const x = bounds.min[0] + (bounds.max[0] - bounds.min[0]) * fx;
      const z = bounds.min[2] + (bounds.max[2] - bounds.min[2]) * fz;
      let bestY = -Infinity;
      for (let t = 0; t < triangles.length; t += 9) {
        const x1 = triangles[t]; const y1 = triangles[t + 1]; const z1 = triangles[t + 2];
        const x2 = triangles[t + 3]; const y2 = triangles[t + 4]; const z2 = triangles[t + 5];
        const x3 = triangles[t + 6]; const y3 = triangles[t + 7]; const z3 = triangles[t + 8];
        const den = (z2 - z3) * (x1 - x3) + (x3 - x2) * (z1 - z3);
        if (Math.abs(den) < eps) continue;
        const a = ((z2 - z3) * (x - x3) + (x3 - x2) * (z - z3)) / den;
        const b = ((z3 - z1) * (x - x3) + (x1 - x3) * (z - z3)) / den;
        const c = 1 - a - b;
        if (a >= -eps && b >= -eps && c >= -eps) bestY = Math.max(bestY, a * y1 + b * y2 + c * y3);
      }
      if (Number.isFinite(bestY)) hits.push(bestY);
    }
  }
  hits.sort((a, b) => a - b);
  return {
    supported: true,
    rays: gridSize * gridSize,
    hits: hits.length,
    misses: gridSize * gridSize - hits.length,
    hitRate: +(hits.length / (gridSize * gridSize)).toFixed(4),
    surfaceY: hits.length ? { min: hits[0], median: hits[Math.floor(hits.length / 2)], max: hits.at(-1) } : null,
  };
}

function inspectGlb(file) {
  const data = fs.readFileSync(file);
  if (data.subarray(0, 4).toString("ascii") !== "glTF") throw new Error("Not a binary glTF file");
  const version = data.readUInt32LE(4);
  const declaredLength = data.readUInt32LE(8);
  let cursor = 12;
  let gltf;
  let bin;
  while (cursor < declaredLength) {
    const length = data.readUInt32LE(cursor);
    const type = data.readUInt32LE(cursor + 4);
    const chunk = data.subarray(cursor + 8, cursor + 8 + length);
    if (type === 0x4e4f534a) gltf = JSON.parse(chunk.toString("utf8").replace(/\0+$/g, "").trim());
    if (type === 0x004e4942) bin = chunk;
    cursor += 8 + length;
  }
  if (!gltf || !bin) throw new Error("GLB is missing JSON or BIN chunk");

  const nodeNames = (gltf.nodes ?? []).map((node, index) => ({ index, name: node.name ?? null, mesh: node.mesh ?? null }));
  const extras = [];
  for (const [kind, values] of Object.entries({ asset: [gltf.asset], scene: gltf.scenes, node: gltf.nodes, mesh: gltf.meshes, primitive: (gltf.meshes ?? []).flatMap((m) => m.primitives ?? []), material: gltf.materials })) {
    (values ?? []).forEach((value, index) => { if (value?.extras != null) extras.push({ kind, index, extras: value.extras }); });
  }

  const triangles = [];
  const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  let totalVertices = 0;
  let totalTriangles = 0;
  let totalPrimitives = 0;
  const visited = new Set();
  const scene = gltf.scenes?.[gltf.scene ?? 0];
  const roots = scene?.nodes ?? (gltf.nodes ?? []).map((_, i) => i);
  function walk(nodeIndex, parent) {
    const visitKey = `${nodeIndex}:${parent.join(",")}`;
    if (visited.has(visitKey)) return;
    visited.add(visitKey);
    const node = gltf.nodes[nodeIndex];
    const world = multiply(parent, nodeMatrix(node));
    if (node.mesh != null) {
      const mesh = gltf.meshes[node.mesh];
      for (const primitive of mesh.primitives ?? []) {
        totalPrimitives += 1;
        if (primitive.attributes?.POSITION == null) continue;
        const positions = decodeAccessor(gltf, bin, primitive.attributes.POSITION).map((p) => transformPoint(world, p));
        totalVertices += positions.length;
        for (const p of positions) {
          for (let k = 0; k < 3; k += 1) {
            bounds.min[k] = Math.min(bounds.min[k], p[k]);
            bounds.max[k] = Math.max(bounds.max[k], p[k]);
          }
        }
        if ((primitive.mode ?? 4) !== 4) continue;
        const indices = primitive.indices == null ? positions.map((_, i) => i) : decodeAccessor(gltf, bin, primitive.indices);
        totalTriangles += Math.floor(indices.length / 3);
        for (let i = 0; i + 2 < indices.length; i += 3) {
          triangles.push(...positions[indices[i]], ...positions[indices[i + 1]], ...positions[indices[i + 2]]);
        }
      }
    }
    for (const child of node.children ?? []) walk(child, world);
  }
  for (const rootNode of roots) walk(rootNode, identity());

  const semanticKeyHits = [];
  const semanticPattern = /(semantic|class|instance|object.?id|joint|affordance|movable)/i;
  function scan(value, location) {
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      if (semanticPattern.test(key)) semanticKeyHits.push(`${location}.${key}`);
      if (child && typeof child === "object") scan(child, `${location}.${key}`);
    }
  }
  scan(gltf, "gltf");

  return {
    ...fileInfo(file),
    format: "glb",
    version,
    generator: gltf.asset?.generator ?? null,
    sceneCount: gltf.scenes?.length ?? 0,
    nodeCount: gltf.nodes?.length ?? 0,
    namedNodes: nodeNames.filter((node) => node.name),
    meshCount: gltf.meshes?.length ?? 0,
    primitiveCount: totalPrimitives,
    decodedVertexReferences: totalVertices,
    triangleCount: totalTriangles,
    materialCount: gltf.materials?.length ?? 0,
    textureCount: gltf.textures?.length ?? 0,
    imageCount: gltf.images?.length ?? 0,
    cameraCount: gltf.cameras?.length ?? 0,
    skinCount: gltf.skins?.length ?? 0,
    animationCount: gltf.animations?.length ?? 0,
    extras,
    semanticMetadataKeyHits: [...new Set(semanticKeyHits)],
    bounds,
    raycast: raycastGrid(triangles, bounds),
  };
}

const result = { generatedAt: new Date().toISOString(), root };
const worldPath = path.join(root, "world.json");
if (fs.existsSync(worldPath)) {
  const response = readJson(worldPath);
  const world = response.world ?? response;
  result.world = {
    id: world.id ?? world.world_id,
    displayName: world.display_name,
    model: world.model,
    worldMarbleUrl: world.world_marble_url,
    caption: world.assets?.caption,
    metricScaleFactor: world.assets?.splats?.semantics_metadata?.metric_scale_factor,
    groundPlaneOffset: world.assets?.splats?.semantics_metadata?.ground_plane_offset,
    assetKeys: Object.keys(world.assets ?? {}),
  };
}

for (const [key, name, inspect] of [
  ["spz100k", "splats-100k.spz", inspectSpz],
  ["spz500k", "splats-500k.spz", inspectSpz],
  ["spzFull", "splats-full-res.spz", inspectSpz],
  ["collider", "collider.glb", inspectGlb],
  ["hqMesh", "hq-mesh.glb", inspectGlb],
  ["fullResMesh", "full-res-mesh.glb", inspectGlb],
  ["pano", "pano.png", inspectPng],
]) {
  const file = path.join(root, name);
  if (fs.existsSync(file)) result[key] = inspect(file);
}

const output = path.join(root, "inspection.json");
fs.writeFileSync(output, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
