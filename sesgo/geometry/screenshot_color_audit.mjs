// Visual color-audit harness for the geometry UI.
//
// Drives the running geometry_viz_server, switches to the richest model
// (Qwen3-0.6B, 16 axes), then screenshots the 2D scatter coloured by each of a
// representative set of axis types (2/3/4/high-cardinality categorical + two
// continuous) plus the per-sample DETAIL panel. Every screenshot is written to
// out/_coloraudit/*.png for human (image-token) review, and JS console errors
// are collected and printed so the audit can demand ZERO errors.
//
// Run: NODE_PATH="$(npm root -g)" node sesgo/geometry/screenshot_color_audit.mjs
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const BASE = process.env.BASE_URL || "http://127.0.0.1:8081";
const MODEL = process.env.MODEL || "Qwen3-0.6B";
const OUT = process.env.OUT_DIR || "out/_coloraudit";

// (axis key, file label). High-cardinality + continuous + low-cardinality mix.
const AXES = [
  ["scaffold", "2class_scaffold"],
  ["selected_role", "3class_selected_role"],
  ["bias_category", "4class_bias_category"],
  ["target_identity", "highcard_target_identity"],
  ["top_choice_prob", "cont_top_choice_prob"],
  ["vocab_entropy", "cont_vocab_entropy"],
];

const consoleErrors = [];

async function pickColor(page, key) {
  await page.selectOption("#sel-color", key);
  await page.waitForTimeout(450); // let Plotly.react settle
}

// Read live trace summary (names + colors) straight off the Plotly graph div.
const traceSummary = (page) =>
  page.evaluate(() => {
    const gd = document.getElementById("plot");
    const data = gd.data || [];
    return data.map((t) => ({
      name: t.name,
      markerColor: t.marker && !Array.isArray(t.marker.color) ? t.marker.color : null,
      colorIsArray: !!(t.marker && Array.isArray(t.marker.color)),
      hasColorbar: !!(t.marker && t.marker.showscale),
      colorscale: t.marker && t.marker.colorscale ? "set" : null,
    }));
  });

// Read the rendered legend swatch fills so we can compare to marker colors.
const legendSwatches = (page) =>
  page.evaluate(() => {
    const out = [];
    document.querySelectorAll("g.traces").forEach((g) => {
      const label = g.querySelector(".legendtext");
      const swatch = g.querySelector(".legendpoints path.scatterpts, .legendpoints path");
      if (label) {
        out.push({
          text: label.textContent,
          fill: swatch ? swatch.getAttribute("style") || swatch.getAttribute("fill") : null,
        });
      }
    });
    return out;
  });

async function main() {
  // scattergl/scatter3d need WebGL; headless Chromium has none by default, so
  // force the SwiftShader software-GL path (otherwise the plot area is blank).
  const browser = await chromium.launch({
    headless: true,
    args: [
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--ignore-gpu-blocklist",
      "--enable-webgl",
    ],
  });
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
  page.on("pageerror", (e) => consoleErrors.push(String(e)));

  await page.goto(BASE, { waitUntil: "networkidle" });
  await page.selectOption("#sel-model", MODEL);
  await page.waitForTimeout(1000);
  await page.waitForFunction(
    () => document.querySelectorAll("#sel-color option").length >= 15,
    { timeout: 10000 }
  );

  const report = {};
  for (const [key, file] of AXES) {
    await pickColor(page, key);
    const ts = await traceSummary(page);
    const sw = await legendSwatches(page);
    report[key] = { traces: ts.length, summary: ts, swatches: sw };
    // Full page screenshot (scatter + legend + side panels).
    await page.screenshot({ path: `${OUT}/2d_${file}.png`, fullPage: false });
    console.log(`shot 2d_${file}.png  (${ts.length} traces)`);
    // For a continuous axis, also crop just the colorbar region (right edge of
    // the plot) so its tick/title contrast can be judged up close.
    if (ts.some((t) => t.hasColorbar)) {
      const plot = await page.$("#plot");
      const bb = await plot.boundingBox();
      await page.screenshot({
        path: `${OUT}/colorbar_${file}.png`,
        clip: { x: bb.x + bb.width - 230, y: bb.y, width: 230, height: bb.height },
      });
      console.log(`shot colorbar_${file}.png`);
    }
  }

  // High-cardinality identity in 3D too (folded legend should hold up there).
  await page.click('#view-toggle .pill[data-view="3d"]');
  await pickColor(page, "target_identity");
  await page.waitForTimeout(700);
  await page.screenshot({ path: `${OUT}/3d_highcard_target_identity.png` });
  console.log("shot 3d_highcard_target_identity.png");
  await page.click('#view-toggle .pill[data-view="2d"]');
  await page.waitForTimeout(400);

  // DETAIL panel: color by selected_role, then load a sample whose non-thinking
  // role probabilities are non-degenerate so ALL THREE role bars (target/other/
  // unknown) are visibly coloured -- that is what we need to judge the role hues.
  await pickColor(page, "selected_role");
  await page.evaluate(async () => {
    const gd = document.getElementById("plot");
    const ids = [];
    (gd.data || []).forEach((t) => (t.customdata || []).forEach((c) => ids.push(c)));
    const model = new URLSearchParams(location.search).get("model") || "Qwen3-0.6B";
    // Scan candidates for one whose 3-way prob vector is most spread out.
    let best = null, bestSpread = -1;
    for (const idx of ids.slice(0, 120)) {
      try {
        const d = await fetch(`/api/sample/${idx}?model=${encodeURIComponent(model)}`).then((r) => r.json());
        const p = d.non_thinking && d.non_thinking.prob;
        if (!p) continue;
        // Favor the sample with the largest minimum role prob -> all 3 bars show.
        const minProb = Math.min(...p);
        if (minProb > bestSpread) { bestSpread = minProb; best = idx; }
      } catch (e) {}
    }
    if (best != null && window.loadDetail) window.loadDetail(best);
  });
  await page.waitForTimeout(900);
  await page.screenshot({ path: `${OUT}/detail_panel_full.png`, fullPage: false });
  // Crop just the side panel (detail card).
  const side = await page.$(".side");
  if (side) await side.screenshot({ path: `${OUT}/detail_panel.png` });
  console.log("shot detail_panel.png");

  await browser.close();

  console.log("\n=== CONSOLE ERRORS (" + consoleErrors.length + ") ===");
  consoleErrors.forEach((e) => console.log("  ERR:", e));

  console.log("\n=== TRACE/SWATCH REPORT ===");
  for (const [key, r] of Object.entries(report)) {
    console.log(`\n[${key}] traces=${r.traces}`);
    r.summary.slice(0, 20).forEach((t) =>
      console.log(`   name=${JSON.stringify(t.name)} color=${t.markerColor} cbar=${t.hasColorbar} arr=${t.colorIsArray}`)
    );
    if (r.swatches.length) {
      console.log("   legend swatches:");
      r.swatches.slice(0, 20).forEach((s) => console.log(`     ${JSON.stringify(s.text)} -> ${s.fill}`));
    }
  }
  process.exit(consoleErrors.length ? 1 : 0);
}

main().catch((e) => { console.error("harness crashed:", e); process.exit(2); });
