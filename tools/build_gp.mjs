// Build a Guitar Pro 7/8 (.gp) file from an alphaTab score-model JSON.
// Usage: node build_gp.mjs model.json out.gp
// Prints a one-line JSON summary of the re-imported file for verification.
import * as alphaTab from '@coderline/alphatab';
import fs from 'node:fs';

const [modelPath, outPath] = process.argv.slice(2);
if (!modelPath || !outPath) {
  console.error('usage: node build_gp.mjs model.json out.gp');
  process.exit(2);
}

const modelText = fs.readFileSync(modelPath, 'utf8');
const score = alphaTab.model.JsonConverter.jsonToScore(modelText); // expects JSON text
const bytes = new alphaTab.exporter.Gp7Exporter().export(score, null);
fs.writeFileSync(outPath, Buffer.from(bytes));

// verify round-trip
const reloaded = alphaTab.importer.ScoreLoader.loadScoreFromBytes(
  new Uint8Array(fs.readFileSync(outPath)), new alphaTab.Settings());
let notes = 0;
for (const t of reloaded.tracks)
  for (const st of t.staves)
    for (const b of st.bars)
      for (const v of b.voices)
        for (const bt of v.beats) notes += bt.notes.length;
console.log(JSON.stringify({
  title: reloaded.title, masterBars: reloaded.masterBars.length, notes,
  bytes: fs.statSync(outPath).size,
}));
