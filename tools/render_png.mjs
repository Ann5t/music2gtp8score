// Render a .gp file to PNG via alphaTab (alphaSkia backend).
// Usage: node render_png.mjs score.gp out.png [width]
import * as alphaTab from '@coderline/alphatab';
import * as alphaSkia from '@coderline/alphaskia';
import fs from 'node:fs';
import path from 'node:path';

const [gpPath, outPng, widthArg] = process.argv.slice(2);
if (!gpPath || !outPng) {
  console.error('usage: node render_png.mjs score.gp out.png [width]');
  process.exit(2);
}

const fontBuf = fs.readFileSync(
  path.resolve('node_modules/@coderline/alphatab/dist/font/Bravura.otf'));
const fontData = fontBuf.buffer.slice(fontBuf.byteOffset, fontBuf.byteOffset + fontBuf.byteLength);
alphaTab.Environment.enableAlphaSkia(fontData, alphaSkia);

const settings = new alphaTab.Settings();
settings.core.engine = 'skia';
settings.core.useWorkers = false;
settings.core.enableLazyLoading = false; // render all chunks eagerly
const score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(
  new Uint8Array(fs.readFileSync(gpPath)), settings);

const renderer = new alphaTab.rendering.ScoreRenderer(settings);
renderer.width = Number(widthArg) || 1400;
const chunks = [];
let totalWidth = 0;
let totalHeight = 0;
renderer.partialRenderFinished.on(r => chunks.push(r));
renderer.renderFinished.on(r => { totalWidth = r.totalWidth; totalHeight = r.totalHeight; });
renderer.renderScore(score, [0]);

if (chunks.length === 0) {
  console.error('render produced no chunks');
  process.exit(1);
}
const w = Math.max(...chunks.map(c => c.x + c.width));
const h = Math.max(...chunks.map(c => c.y + c.height));
const canvas = new alphaSkia.AlphaSkiaCanvas();
canvas.beginRender(Math.ceil(w), Math.ceil(h), 1);
canvas.color = alphaSkia.AlphaSkiaCanvas.rgbaToColor(255, 255, 255, 255);
canvas.fillRect(0, 0, w, h);
for (const c of chunks) canvas.drawImage(c.renderResult, c.x, c.y, c.width, c.height);
const img = canvas.endRender();
fs.writeFileSync(outPng, Buffer.from(img.toPng()));
console.log(JSON.stringify({ out: outPng, width: w, height: h, chunks: chunks.length }));
