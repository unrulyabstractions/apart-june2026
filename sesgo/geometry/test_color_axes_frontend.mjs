// Headless frontend test for the geometry COLOR BY dropdown (full axis set).
//
// Launches against an already-running geometry_viz_server (BASE_URL env or
// http://127.0.0.1:8079), then asserts, with ZERO JS console errors throughout:
//   1. The COLOR BY dropdown exposes >= 15 colour-by axes (full registry, not 4).
//   2. Selecting a CATEGORICAL axis (accuracy, then selected_role) recolours the
//      scatter into a discrete per-category legend (multiple legend entries).
//   3. Selecting a CONTINUOUS axis (top_choice_prob, then vocab_entropy)
//      recolours into a single Viridis-coloured trace with a visible colorbar.
//
// Run (Node playwright via the global install):
//   NODE_PATH="$(npm root -g)" node sesgo/geometry/test_color_axes_frontend.mjs
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const BASE = process.env.BASE_URL || "http://127.0.0.1:8079";
// The default boot model may be axis-poor (e.g. Llama-70B has 13 axes); switch to
// a model that carries the full registry (>= 15 axes) to exercise the dropdown.
const FULL_AXIS_MODEL = process.env.FULL_AXIS_MODEL || "Qwen3-0.6B";

const fails = [];
const check = (cond, msg) => { if (!cond) fails.push(msg); else console.log("  ok:", msg); };

// Read the Plotly graph div's live traces (the source of truth for recolouring).
const traces = (page) =>
  page.evaluate(() => (document.getElementById("plot").data || []).map((t) => ({
    name: t.name,
    hasColorbar: !!(t.marker && t.marker.showscale && t.marker.colorbar),
    colorIsArray: !!(t.marker && Array.isArray(t.marker.color)),
  })));

// Set the COLOR BY <select> to an axis KEY and dispatch change (drives render()).
async function pickColor(page, key) {
  await page.selectOption("#sel-color", key);
  await page.waitForTimeout(350); // let Plotly.react settle
}

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const consoleErrors = [];
  page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
  page.on("pageerror", (e) => consoleErrors.push(String(e)));

  await page.goto(BASE, { waitUntil: "networkidle" });
  // Switch to the full-axis model and wait for its config/projection to load.
  await page.selectOption("#sel-model", FULL_AXIS_MODEL);
  await page.waitForTimeout(900);
  await page.waitForFunction(
    () => document.querySelectorAll("#sel-color option").length >= 15,
    { timeout: 8000 }
  );

  // 1) Dropdown exposes the full axis set.
  const axisKeys = await page.$$eval("#sel-color option", (os) => os.map((o) => o.value));
  const axisLabels = await page.$$eval("#sel-color option", (os) => os.map((o) => o.textContent));
  console.log("\nCOLOR BY axes (" + axisKeys.length + "):");
  axisKeys.forEach((k, i) => console.log(`  ${k}  ->  ${axisLabels[i]}`));
  check(axisKeys.length >= 15, `dropdown lists >= 15 axes (got ${axisKeys.length})`);
  // Labels (not raw keys) should be shown.
  check(axisLabels.some((l, i) => l !== axisKeys[i]),
    "dropdown shows human axis LABELS, not raw keys");

  // 2) Categorical axes -> discrete per-category legend (multiple traces).
  for (const key of ["accuracy", "selected_role"]) {
    await pickColor(page, key);
    const ts = await traces(page);
    const catTraces = ts.filter((t) => !t.hasColorbar && !t.colorIsArray);
    const anyColorbar = ts.some((t) => t.hasColorbar);
    check(catTraces.length >= 2,
      `categorical '${key}': discrete legend has >= 2 category traces (got ${catTraces.length})`);
    check(!anyColorbar, `categorical '${key}': NO colorbar present`);
  }

  // 3) Continuous axes -> single Viridis trace with a visible colorbar.
  for (const key of ["top_choice_prob", "vocab_entropy"]) {
    await pickColor(page, key);
    const ts = await traces(page);
    const withBar = ts.filter((t) => t.hasColorbar && t.colorIsArray);
    check(withBar.length === 1,
      `continuous '${key}': single colour-mapped trace with colorbar (got ${withBar.length})`);
  }

  // 4) Zero JS console errors throughout.
  check(consoleErrors.length === 0,
    `zero JS console errors (got ${consoleErrors.length}${consoleErrors.length ? ": " + consoleErrors.join(" | ") : ""})`);

  await browser.close();

  console.log("\n" + (fails.length ? "FAILED:\n  - " + fails.join("\n  - ") : "ALL CHECKS PASSED"));
  process.exit(fails.length ? 1 : 0);
}

main().catch((e) => { console.error("test crashed:", e); process.exit(2); });
