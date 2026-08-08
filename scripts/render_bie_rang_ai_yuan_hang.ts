import { createCanvas, GlobalFonts, Path2D } from '@napi-rs/canvas';
import { bieRangAiYuanHang } from '../src/data/bie-rang-ai-yuan-hang';
import { calculateLayout, PRINT_CONFIG, PRINT_THEME, render } from '../src/engine';
import { writeFileSync } from 'node:fs';

// symbols.ts uses the browser global. @napi-rs/canvas provides the same API.
(globalThis as typeof globalThis & { Path2D: typeof Path2D }).Path2D = Path2D;
GlobalFonts.registerFromPath('/System/Library/Fonts/Supplemental/Songti.ttc', 'STSong');

const config = { ...PRINT_CONFIG };

const layout = calculateLayout(bieRangAiYuanHang, config);
const canvas = createCanvas(layout.width, layout.height);
const context = canvas.getContext('2d');
render(context as unknown as CanvasRenderingContext2D, layout, bieRangAiYuanHang, config, PRINT_THEME);
writeFileSync('public/别让爱远航_jianpu_rendered.png', canvas.toBuffer('image/png'));
console.log(JSON.stringify({ width: layout.width, height: layout.height, rows: layout.rows.length, measures: bieRangAiYuanHang.measures.length }));
