#!/usr/bin/env node

import fs from "node:fs";
import http from "node:http";
import path from "node:path";

const projectRoot = process.cwd();
const port = Number.parseInt(process.env.DEMO_PORT ?? "4173", 10);
const host = process.env.DEMO_HOST ?? "127.0.0.1";

const publicRoots = new Map([
  ["/web-demo/", path.join(projectRoot, "web-demo")],
  ["/world-viewer/", path.join(projectRoot, "world-viewer")],
]);
const publicArtifacts = new Set([
  "/artifacts/corgi-cafe/source-google-maps.jpg",
  "/artifacts/corgi-cafe/pano.png",
  "/artifacts/corgi-cafe/splats-100k.spz",
  "/artifacts/corgi-cafe/collider.glb",
  "/artifacts/corgi-cafe/world.json",
  "/artifacts/worldlabs-audit/splats-100k.spz",
]);
const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".glb": "model/gltf-binary",
  ".html": "text/html; charset=utf-8",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".spz": "application/octet-stream",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
};

function resolveRequest(urlPath) {
  if (urlPath === "/") return { redirect: "/web-demo/" };

  if (publicArtifacts.has(urlPath)) {
    return { filePath: path.join(projectRoot, urlPath.slice(1)) };
  }

  for (const [prefix, root] of publicRoots) {
    if (!urlPath.startsWith(prefix)) continue;
    let relativePath = decodeURIComponent(urlPath.slice(prefix.length));
    if (!relativePath || relativePath.endsWith("/")) relativePath += "index.html";
    const filePath = path.resolve(root, relativePath);
    if (filePath !== root && !filePath.startsWith(`${root}${path.sep}`)) return null;
    return { filePath };
  }

  return null;
}

function sendText(response, statusCode, message) {
  response.writeHead(statusCode, {
    "Content-Type": "text/plain; charset=utf-8",
    "Cache-Control": "no-store",
  });
  response.end(message);
}

const server = http.createServer((request, response) => {
  if (request.method !== "GET" && request.method !== "HEAD") {
    sendText(response, 405, "Method not allowed");
    return;
  }

  const urlPath = new URL(request.url ?? "/", `http://${host}`).pathname;
  const resolved = resolveRequest(urlPath);
  if (!resolved) {
    sendText(response, 404, "Not found");
    return;
  }
  if (resolved.redirect) {
    response.writeHead(302, { Location: resolved.redirect, "Cache-Control": "no-store" });
    response.end();
    return;
  }

  fs.stat(resolved.filePath, (statError, stat) => {
    if (statError || !stat.isFile()) {
      sendText(response, 404, "Not found");
      return;
    }

    response.writeHead(200, {
      "Content-Type": mimeTypes[path.extname(resolved.filePath).toLowerCase()] ?? "application/octet-stream",
      "Content-Length": stat.size,
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    });
    if (request.method === "HEAD") {
      response.end();
      return;
    }
    fs.createReadStream(resolved.filePath).pipe(response);
  });
});

server.listen(port, host, () => {
  console.log(`World2Work demo: http://${host}:${port}/web-demo/`);
  console.log(`Only demo files and allow-listed generated assets are being served.`);
});
