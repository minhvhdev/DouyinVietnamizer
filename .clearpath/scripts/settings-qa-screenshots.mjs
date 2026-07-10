import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(__dirname, "../docs/changes/settings-ui-redesign/evidence");
const tabs = [
  { label: "Tải video", slug: "download" },
  { label: "Dịch thuật", slug: "translation" },
  { label: "Âm thanh", slug: "audio" },
  { label: "Lồng tiếng", slug: "tts" },
  { label: "Phụ đề", slug: "subtitles" },
];
const viewports = [
  { name: "desktop", width: 1440, height: 900 },
  { name: "narrow", width: 390, height: 844 },
];

await mkdir(outDir, { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

async function openSettings() {
  await page.goto("http://localhost:5173/", { waitUntil: "domcontentloaded", timeout: 60000 });
  await page.waitForSelector("nav button", { state: "attached", timeout: 60000 });
  await page.evaluate(() => {
    const buttons = document.querySelectorAll("nav button");
    buttons[3]?.click();
  });
  await page.waitForSelector('[role="tab"]', { timeout: 15000 });
  await page.waitForTimeout(600);
}

async function captureViewport(vp) {
  await page.setViewportSize({ width: vp.width, height: vp.height });
  await page.waitForTimeout(500);

  for (const tab of tabs) {
    await page.getByRole("tab", { name: tab.label }).click();
    await page.waitForTimeout(500);
    const file = path.join(outDir, `${vp.name}-settings-${tab.slug}.png`);
    await page.screenshot({ path: file, fullPage: true });
    console.log("saved", file);
  }
}

await page.setViewportSize({ width: viewports[0].width, height: viewports[0].height });
await openSettings();

for (const vp of viewports) {
  await captureViewport(vp);
}

await browser.close();
console.log("done");
