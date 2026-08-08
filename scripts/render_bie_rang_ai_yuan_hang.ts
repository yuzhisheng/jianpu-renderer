import { createCanvas, GlobalFonts, Path2D } from '@napi-rs/canvas';
import { bieRangAiYuanHang } from '../src/data/bie-rang-ai-yuan-hang';
import { calculateLayout, DEFAULT_CONFIG, LIGHT_THEME, render } from '../src/engine';
import { writeFileSync } from 'node:fs';

// symbols.ts uses the browser global. @napi-rs/canvas provides the same API.
(globalThis as typeof globalThis & { Path2D: typeof Path2D }).Path2D = Path2D;
GlobalFonts.registerFromPath('/System/Library/Fonts/Supplemental/Songti.ttc', 'STSong');

const config = {
  ...DEFAULT_CONFIG,
  canvasWidth: 1200,
  paddingHorizontal: 42,
  paddingVertical: 34,
  noteFontSize: 28,
  noteWidth: 38,
  noteHeight: 32,
  measureGap: 18,
  rowGap: 26,
  dotRadius: 2.2,
  dotGap: 3,
  accentDotRadius: 2.2,
  underlineOffset: 3.5,
  underlineGap: 4,
  underlineThickness: 1.8,
  barlineWidth: 2,
  techniqueFontSize: 13,
  titleFontSize: 28,
  metaFontSize: 18,
  lyricFontSize: 18,
  lyricOffset: 22,
  tieCurveHeight: 9,
};

const layout = calculateLayout(bieRangAiYuanHang, config);
const canvas = createCanvas(layout.width, layout.height);
const context = canvas.getContext('2d');
render(context as unknown as CanvasRenderingContext2D, layout, bieRangAiYuanHang, config, LIGHT_THEME);
writeFileSync('public/别让爱远航_jianpu_rendered.png', canvas.toBuffer('image/png'));
console.log(JSON.stringify({ width: layout.width, height: layout.height, rows: layout.rows.length, measures: bieRangAiYuanHang.measures.length }));
