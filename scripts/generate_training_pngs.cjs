#!/usr/bin/env node
/**
 * 将 public/training/ 下的 JSON 简谱渲染为白底黑字 PNG
 * 用法: node scripts/generate_training_pngs.cjs
 */
const fs = require('fs');
const path = require('path');
const { createCanvas, Path2D } = require('@napi-rs/canvas');

// 注入全局 Path2D 和 document（引擎依赖）
globalThis.Path2D = Path2D;
globalThis.document = { createElement: () => createCanvas(1, 1) };

const trainDir = 'public/training';
const files = fs.readdirSync(trainDir).filter(f => f.endsWith('.json') && f !== 'index.json');

const bwTheme = {
  backgroundColor: '#ffffff',
  noteColor: '#000000',
  symbolColor: '#000000',
  dashColor: '#000000',
  dotColor: '#000000',
  underlineColor: '#000000',
  barlineColor: '#000000',
  tieColor: '#000000',
  lyricColor: '#000000',
  techniqueColor: '#000000',
  repeatDotColor: '#000000',
  paddingHorizontal: 30,
  paddingVertical: 15,
  canvasWidth: 800,
};

let count = 0;
for (const file of files) {
  const data = JSON.parse(fs.readFileSync(`${trainDir}/${file}`, 'utf-8'));
  delete data.title; delete data.key; delete data.tempo; delete data.tempoText; delete data.introMeasureCount;

  // 每次动态加载引擎模块（绕过 ESM/CJS 互操作问题）
  delete require.cache[require.resolve('../src/engine/index')];
  delete require.cache[require.resolve('../src/engine/layout')];
  delete require.cache[require.resolve('../src/engine/renderer')];
  delete require.cache[require.resolve('../src/engine/symbols')];

  const { calculateLayout, DEFAULT_CONFIG, render } = require('../src/engine/index');

  const config = { ...DEFAULT_CONFIG, ...bwTheme, paddingVertical: 10 };
  const layout = calculateLayout(data, config);

  // 去除标题/调号/拍号的 meta 位置，只保留乐句
  layout.titlePosition = undefined;
  layout.keyPosition = undefined;
  layout.timeSignaturePosition = undefined;

  const canvas = createCanvas(layout.width, layout.height);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, layout.width, layout.height);

  render(ctx, layout, data, config, bwTheme);

  const pngFile = file.replace('.json', '.png');
  fs.writeFileSync(`${trainDir}/${pngFile}`, canvas.toBuffer('image/png'));
  console.log(`✓ ${pngFile}`);
  count++;
}
console.log(`\nDone! ${count} PNGs → ${trainDir}/`);
