import { chromium } from '/Users/jeff/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
import { pathToFileURL } from 'node:url';

const input = '/Users/jeff/project/spatial_intelligence_jeff/artifacts/slide_assets/demo_scenario.html';
const output = '/Users/jeff/project/spatial_intelligence_jeff/artifacts/slide_assets/demo_scenario.png';

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(input).href, { waitUntil: 'networkidle' });
await page.screenshot({ path: output, type: 'png' });
await browser.close();

console.log(output);
