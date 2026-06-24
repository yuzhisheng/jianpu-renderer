import type {
  ScoreLayout, RowLayout, MeasureLayout, NoteLayout, SymbolPosition,
  Note, Dash, DiziTechnique,
} from '../types';
import type { LayoutConfig } from './layout';
import { drawSymbol as drawSvgSymbol } from './symbols';

/** 用 SVG 路径符号替换 Canvas 本地绘制 */
function drawSymbol(ctx: CanvasRenderingContext2D, num: number, x: number, y: number, size: number, color: string) {
  drawSvgSymbol(ctx, num, x, y, size, color);
}

/** 渲染配色方案 */
export interface RenderTheme {
  noteColor: string;
  symbolColor: string;
  barlineColor: string;
  titleColor: string;
  metaColor: string;
  lyricColor: string;
  techniqueColor: string;
  backgroundColor: string;
  dashColor: string;
  dotColor: string;
  underlineColor: string;
  tieColor: string;
  repeatDotColor: string;
}

export const DEFAULT_THEME: RenderTheme = {
  noteColor: '#569cd6',
  symbolColor: '#569cd6',
  barlineColor: '#5a5a5a',
  titleColor: '#569cd6',
  metaColor: '#9cdcfe',
  lyricColor: '#8a8a8a',
  techniqueColor: '#569cd6',
  backgroundColor: '#1e1e1e',
  dashColor: '#569cd6',
  dotColor: '#569cd6',
  underlineColor: '#569cd6',
  tieColor: '#569cd6',
  repeatDotColor: '#569cd6',
};

export const LIGHT_THEME: RenderTheme = {
  noteColor: '#333333',
  symbolColor: '#333333',
  barlineColor: '#999999',
  titleColor: '#222222',
  metaColor: '#555555',
  lyricColor: '#888888',
  techniqueColor: '#333333',
  backgroundColor: '#ffffff',
  dashColor: '#333333',
  dotColor: '#333333',
  underlineColor: '#333333',
  tieColor: '#333333',
  repeatDotColor: '#333333',
};

function isNoteType(data: Note | Dash): data is Note {
  return 'pitch' in data;
}

/** 绘制音符数字 */
function drawNoteNumber(ctx: CanvasRenderingContext2D, note: Note, pos: SymbolPosition, config: LayoutConfig, theme: RenderTheme) {
  ctx.fillStyle = theme.noteColor;
  ctx.font = `bold ${config.noteFontSize}px "Noto Sans", "SimSun", serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const text = note.pitch === 0 ? '0' : String(note.pitch);
  ctx.fillText(text, pos.x + pos.width / 2, pos.y + pos.height / 2);
}

/** 绘制增时线 */
function drawDash(ctx: CanvasRenderingContext2D, pos: SymbolPosition, config: LayoutConfig, theme: RenderTheme) {
  ctx.strokeStyle = theme.dashColor;
  ctx.lineWidth = config.dashThickness;
  const y = pos.y + pos.height / 2;
  ctx.beginPath();
  ctx.moveTo(pos.x + 2, y);
  ctx.lineTo(pos.x + pos.width - 2, y);
  ctx.stroke();
}

/** 绘制八度点 */
function drawOctaveDots(ctx: CanvasRenderingContext2D, positions: SymbolPosition[], theme: RenderTheme) {
  ctx.fillStyle = theme.dotColor;
  positions.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x + p.width / 2, p.y + p.height / 2, p.width / 2, 0, Math.PI * 2);
    ctx.fill();
  });
}

/** 绘制附点（用短横线代替圆点） */
function drawDots(ctx: CanvasRenderingContext2D, positions: SymbolPosition[], config: LayoutConfig, theme: RenderTheme) {
  ctx.strokeStyle = theme.dotColor;
  ctx.lineWidth = config.dashThickness;
  positions.forEach(p => {
    ctx.beginPath();
    ctx.moveTo(p.x, p.y + p.height / 2);
    ctx.lineTo(p.x + config.noteWidth * 0.4, p.y + p.height / 2);
    ctx.stroke();
  });
}

/** 绘制升降号 */
function drawAccidental(ctx: CanvasRenderingContext2D, accidental: string, pos: SymbolPosition, config: LayoutConfig, theme: RenderTheme) {
  const symbolMap: Record<string, number> = { sharp: 1, flat: 2, natural: 3 };
  const num = symbolMap[accidental];
  if (num) {
    drawSymbol(ctx, num, pos.x, pos.y, pos.height, theme.symbolColor);
    return;
  }
  ctx.fillStyle = theme.symbolColor;
  ctx.font = `bold ${config.noteFontSize - 4}px "Noto Sans", serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const symbols: Record<string, string> = { sharp: '♯', flat: '♭', natural: '♮' };
  ctx.fillText(symbols[accidental] || '', pos.x + pos.width / 2, pos.y + pos.height / 2);
}

/** 绘制减时线 */
function drawUnderlines(ctx: CanvasRenderingContext2D, noteLayout: NoteLayout, config: LayoutConfig, theme: RenderTheme) {
  ctx.strokeStyle = theme.underlineColor;
  ctx.lineWidth = config.underlineThickness;
  noteLayout.underlines.forEach(ul => {
    const startX = noteLayout.position.x + (ul.xOffset || 0);
    const endX = startX + ul.width;
    ctx.beginPath();
    ctx.moveTo(startX, ul.y);
    ctx.lineTo(endX, ul.y);
    ctx.stroke();
  });
}

/** 绘制小节线 */
function drawBarline(ctx: CanvasRenderingContext2D, type: string, pos: SymbolPosition, theme: RenderTheme) {
  ctx.strokeStyle = theme.barlineColor;
  const x = pos.x + pos.width / 2;
  const y1 = pos.y;
  const y2 = pos.y + pos.height;

  ctx.lineWidth = 1.5;
  switch (type) {
    case 'single':
      ctx.beginPath();
      ctx.moveTo(x, y1);
      ctx.lineTo(x, y2);
      ctx.stroke();
      break;
    case 'double':
      ctx.beginPath();
      ctx.moveTo(x - 3, y1);
      ctx.lineTo(x - 3, y2);
      ctx.stroke();
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(x + 3, y1);
      ctx.lineTo(x + 3, y2);
      ctx.stroke();
      break;
    case 'end':
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x - 4, y1);
      ctx.lineTo(x - 4, y2);
      ctx.stroke();
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(x + 3, y1);
      ctx.lineTo(x + 3, y2);
      ctx.stroke();
      break;
    case 'repeat-start':
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x - 4, y1);
      ctx.lineTo(x - 4, y2);
      ctx.stroke();
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x + 2, y1);
      ctx.lineTo(x + 2, y2);
      ctx.stroke();
      // 两个圆点
      ctx.fillStyle = theme.repeatDotColor;
      const dotY1 = y1 + (y2 - y1) * 0.3;
      const dotY2 = y1 + (y2 - y1) * 0.7;
      ctx.beginPath();
      ctx.arc(x + 8, dotY1, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x + 8, dotY2, 2.5, 0, Math.PI * 2);
      ctx.fill();
      break;
    case 'repeat-end':
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x - 5, y1);
      ctx.lineTo(x - 5, y2);
      ctx.stroke();
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + 1, y1);
      ctx.lineTo(x + 1, y2);
      ctx.stroke();
      // 两个圆点
      ctx.fillStyle = theme.repeatDotColor;
      const dotY3 = y1 + (y2 - y1) * 0.3;
      const dotY4 = y1 + (y2 - y1) * 0.7;
      ctx.beginPath();
      ctx.arc(x - 10, dotY3, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x - 10, dotY4, 2.5, 0, Math.PI * 2);
      ctx.fill();
      break;
  }
}

/** 绘制连音线（Tie） */
function drawTie(ctx: CanvasRenderingContext2D, startNote: NoteLayout, endNote: NoteLayout, config: LayoutConfig, theme: RenderTheme) {
  const x1 = startNote.position.x + startNote.position.width / 2;
  const x2 = endNote.position.x + endNote.position.width / 2;
  const y = startNote.position.y - 4;
  const curveH = config.tieCurveHeight;

  ctx.strokeStyle = theme.tieColor;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.quadraticCurveTo((x1 + x2) / 2, y - curveH, x2, y);
  ctx.stroke();
}

/** 绘制圆滑线（Slur） */
function drawSlur(ctx: CanvasRenderingContext2D, startNote: NoteLayout, endNote: NoteLayout, config: LayoutConfig, theme: RenderTheme) {
  const x1 = startNote.position.x + startNote.position.width / 2;
  const x2 = endNote.position.x + endNote.position.width / 2;
  const y = startNote.position.y - 2;
  const curveH = config.tieCurveHeight + 4;

  ctx.strokeStyle = theme.tieColor;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.quadraticCurveTo((x1 + x2) / 2, y - curveH, x2, y);
  ctx.stroke();
}

/** 绘制三连音标记 */
function drawTriplet(ctx: CanvasRenderingContext2D, startNote: NoteLayout, endNote: NoteLayout, config: LayoutConfig, theme: RenderTheme) {
  const x1 = startNote.position.x;
  const x2 = endNote.position.x + endNote.position.width;
  const y = startNote.position.y - config.tieCurveHeight - 10;

  ctx.strokeStyle = theme.symbolColor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x1, y + 4);
  ctx.lineTo(x1, y);
  ctx.lineTo(x2, y);
  ctx.lineTo(x2, y + 4);
  ctx.stroke();

  ctx.fillStyle = theme.symbolColor;
  ctx.font = `${config.techniqueFontSize}px "Noto Sans", sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  ctx.fillText('3', (x1 + x2) / 2, y - 1);
}

/** 绘制技巧符号 */
function drawTechnique(ctx: CanvasRenderingContext2D, tech: DiziTechnique, pos: SymbolPosition, config: LayoutConfig, theme: RenderTheme, mainNotePos?: SymbolPosition, nextNotePos?: SymbolPosition) {
  ctx.fillStyle = theme.techniqueColor;
  ctx.font = `bold ${config.techniqueFontSize}px "Noto Sans", "SimSun", serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';

  // 滑音 — 两个音符之间使用 SVG 图标
  if (tech.type === 'huayin') {
    const symNum = tech.slideDirection === 'up' ? 13 : 14;
    if (nextNotePos && mainNotePos) {
      const currCenterX = mainNotePos.x;
      const nextCenterX = nextNotePos.x + nextNotePos.width / 2;
      const currCenterY = mainNotePos.y;
      const nextCenterY = nextNotePos.y + nextNotePos.height / 2;
      const midX = (currCenterX + nextCenterX) / 2;
      const midY = (currCenterY + nextCenterY) / 2;
      const dist = nextCenterX - currCenterX;
      const size = Math.max(dist * 0.7, 20);
      drawSymbol(ctx, symNum, midX - size * 0.5, midY - size * 0.4 - 10, size, theme.symbolColor);
    } else {
      drawSymbol(ctx, symNum, pos.x, pos.y - 2, pos.height + 6, theme.symbolColor);
    }
    return;
  }
  // 颤音
  if (tech.type === 'chanyin') {
    drawSymbol(ctx, 19, pos.x, pos.y - 2, pos.height + 8, theme.symbolColor);
    return;
  }
  // 历音 — 完全按赠音样式 + 斜波浪线
  if (tech.type === 'liyin') {
    const isDown = tech.liyinDirection === 'down';
    const gFontSize = config.techniqueFontSize + 2;
    const cx = pos.x + pos.width / 2;
    const textBottom = pos.y + pos.height - 4;
    const graceText = tech.graceNotes?.length ? tech.graceNotes.join('') : (isDown ? '下' : '上');

    // 小字
    ctx.fillStyle = theme.noteColor;
    ctx.font = `bold ${gFontSize}px "Noto Sans", serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(graceText, cx, textBottom - 1);

    // 高音点
    const oct = tech.graceOctave || 0;
    if (oct > 0) {
      ctx.fillStyle = theme.dotColor;
      for (let i = 0; i < oct; i++) {
        const dotR = config.dotRadius - 0.5;
        const dy = pos.y - dotR - 2 - (i > 0 ? config.dotGap * i : 0);
        ctx.beginPath();
        ctx.arc(cx, dy + dotR, dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 双横线
    ctx.strokeStyle = theme.underlineColor;
    ctx.lineWidth = 0.8;
    const lineY1 = textBottom;
    const lineY2 = textBottom + 2.5;
    ctx.beginPath();
    ctx.moveTo(pos.x, lineY1);
    ctx.lineTo(pos.x + pos.width, lineY1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, lineY2);
    ctx.lineTo(pos.x + pos.width, lineY2);
    ctx.stroke();

    // SVG 弧线 (与赠音完全一致)
    {
      const scale = pos.width * 0.55 / 5;
      const arcSize = 6 * scale;
      const arcY = lineY2 + 1;
      if (isDown) {
        drawSymbol(ctx, 100, cx - 0.16 * scale, arcY, arcSize, theme.symbolColor);
      } else {
        ctx.save();
        ctx.translate(cx, 0);
        ctx.scale(-1, 1);
        ctx.translate(-cx, 0);
        drawSymbol(ctx, 100, cx - 0.16 * scale, arcY, arcSize, theme.symbolColor);
        ctx.restore();
      }
    }

    // 斜波浪线
    if (mainNotePos) {
      ctx.strokeStyle = theme.symbolColor;
      ctx.lineWidth = 1;
      ctx.lineCap = 'round';
      ctx.beginPath();
      const sx = mainNotePos.x;
      const sy = mainNotePos.y - 1;
      const ex = isDown ? pos.x + pos.width : pos.x;
      const ey = pos.y + 2;
      const dx = ex - sx;
      const dy = ey - sy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const cycles = Math.max(6, Math.round(dist / 4));
      const steps = Math.max(60, Math.round(dist * 4));
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const px = sx + dx * t;
        const py = sy + dy * t;
        const wave = Math.sin(t * Math.PI * cycles) * 1;
        if (i === 0) ctx.moveTo(px, py + wave);
        else ctx.lineTo(px, py + wave);
      }
      ctx.stroke();
    }
    return;
  }
  // 气震音
  if (tech.type === 'qizhenyin') {
    drawSymbol(ctx, 18, pos.x, pos.y - 2, pos.height + 6, theme.symbolColor);
    return;
  }

  const labels: Record<string, string> = {
    dieyin: '又',
    dayin: '扌',
    yinyin: '倚',
    tuyin: tech.articulation === 'double' ? 'TK' : tech.articulation === 'triple' ? 'TKT' : 'T',
    huashe: '✱',
    xunhuan: '↻',
    fanyin: '○',
  };

  const label = labels[tech.type] || tech.type;

  // 倚音 — 双横线 + 数字小字 + 弧线连主音
  if (tech.type === 'yinyin' && tech.graceNotes?.length) {
    const gFontSize = config.techniqueFontSize + 2;
    const graceText = tech.graceNotes.join('');
    const cx = pos.x + pos.width / 2;
    const textBottom = pos.y + pos.height - 4;

    // 小字
    ctx.fillStyle = theme.noteColor;
    ctx.font = `bold ${gFontSize}px "Noto Sans", serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(graceText, cx, textBottom - 1);

    // 高音点（倚音音符的八度）
    const graceOct = tech.graceOctave || 0;
    if (graceOct > 0) {
      ctx.fillStyle = theme.dotColor;
      for (let i = 0; i < graceOct; i++) {
        const dotR = config.dotRadius - 0.5;
        const dy = pos.y - dotR - 2 - (i > 0 ? config.dotGap * i : 0);
        ctx.beginPath();
        ctx.arc(cx, dy + dotR, dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 双横线（减时线）
    ctx.strokeStyle = theme.underlineColor;
    ctx.lineWidth = 0.8;
    const lineY1 = textBottom;
    const lineY2 = textBottom + 2.5;
    ctx.beginPath();
    ctx.moveTo(pos.x, lineY1);
    ctx.lineTo(pos.x + pos.width, lineY1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, lineY2);
    ctx.lineTo(pos.x + pos.width, lineY2);
    ctx.stroke();

    // SVG 弧线从横线中间下方开始连接
    {
      const scale = pos.width * 0.55 / 5;
      const arcSize = 6 * scale;
      const arcX = cx - 0.16 * scale;
      const arcY = lineY2 + 1;
      drawSymbol(ctx, 100, arcX, arcY, arcSize, theme.symbolColor);
    }
    return;
  }

  // 赠音 — 在音符右侧，与倚音镜像对称
  if (tech.type === 'zengyin') {
    const gFontSize = config.techniqueFontSize + 2;
    const labelText = tech.giftPitch !== undefined ? String(tech.giftPitch) : '赠';
    const cx = pos.x + pos.width / 2;
    const textBottom = pos.y + pos.height - 4;

    // 小字
    ctx.fillStyle = theme.noteColor;
    ctx.font = `bold ${gFontSize}px "Noto Sans", serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(labelText, cx, textBottom - 1);

    // 高音点（赠音八度）
    const giftOct = tech.giftOctave || 0;
    if (giftOct > 0) {
      ctx.fillStyle = theme.dotColor;
      for (let i = 0; i < giftOct; i++) {
        const dotR = config.dotRadius - 0.5;
        const dy = pos.y - dotR - 2 - (i > 0 ? config.dotGap * i : 0);
        ctx.beginPath();
        ctx.arc(cx, dy + dotR, dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 双横线
    ctx.strokeStyle = theme.underlineColor;
    ctx.lineWidth = 0.8;
    const lineY1 = textBottom;
    const lineY2 = textBottom + 2.5;
    ctx.beginPath();
    ctx.moveTo(pos.x, lineY1);
    ctx.lineTo(pos.x + pos.width, lineY1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, lineY2);
    ctx.lineTo(pos.x + pos.width, lineY2);
    ctx.stroke();

    // 向左的弧线（倚音弧线镜像，大小一致）
    {
      const scale = pos.width * 0.55 / 5;
      const arcSize = 6 * scale;
      const arcY = lineY2 + 1;
      ctx.save();
      ctx.translate(cx, 0);
      ctx.scale(-1, 1);
      ctx.translate(-cx, 0);
      drawSymbol(ctx, 100, cx - 0.16 * scale, arcY, arcSize, theme.symbolColor);
      ctx.restore();
    }
    return;
  }

  ctx.fillText(label, pos.x, pos.y + pos.height);
}

/** 绘制重音标记 */
function drawAccent(ctx: CanvasRenderingContext2D, pos: SymbolPosition, _config: LayoutConfig, theme: RenderTheme) {
  drawSymbol(ctx, 10, pos.x - 2, pos.y - 2, pos.width + 6, theme.symbolColor);
}

/** 绘制保持音标记 */
function drawTenuto(ctx: CanvasRenderingContext2D, pos: SymbolPosition, theme: RenderTheme) {
  ctx.strokeStyle = theme.symbolColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y + pos.height / 2);
  ctx.lineTo(pos.x + pos.width, pos.y + pos.height / 2);
  ctx.stroke();
}

/** 绘制延长记号（Fermata） */
function drawFermata(ctx: CanvasRenderingContext2D, pos: SymbolPosition, theme: RenderTheme) {
  const size = pos.width * 0.6;
  const cx = pos.x + pos.width / 2;
  drawSymbol(ctx, 8, cx - size * 0.5, pos.y + 2, size, theme.symbolColor);
}

/** 绘制反复跳跃记号（小房子） */
function drawRepeatEnding(ctx: CanvasRenderingContext2D, numbers: number[], pos: SymbolPosition, config: LayoutConfig, theme: RenderTheme, isLast: boolean = false) {
  const text = numbers.join('.');
  ctx.strokeStyle = theme.symbolColor;
  ctx.lineWidth = 1;

  const bracketTop = pos.y;
  const bracketBottom = pos.y + pos.height;

  // 左端竖线
  ctx.beginPath();
  ctx.moveTo(pos.x, bracketTop);
  ctx.lineTo(pos.x, bracketBottom);
  ctx.stroke();

  // 横线（与竖线顶部对齐）
  ctx.beginPath();
  ctx.moveTo(pos.x, bracketTop);
  ctx.lineTo(pos.x + pos.width, bracketTop);
  ctx.stroke();

  // 若是最后一小节，画右端竖线
  if (isLast) {
    ctx.beginPath();
    ctx.moveTo(pos.x + pos.width, bracketTop);
    ctx.lineTo(pos.x + pos.width, bracketBottom);
    ctx.stroke();
  }

  // 数字（横线下方）
  ctx.fillStyle = theme.symbolColor;
  ctx.font = `${config.techniqueFontSize}px "Noto Sans", sans-serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(text, pos.x + 6, bracketTop + 3);
}

/** 绘制标题和元信息 */
function drawMeta(ctx: CanvasRenderingContext2D, layout: ScoreLayout, score: { title?: string; key: string; timeSignature: { numerator: number; denominator: number }; tempo?: number; tempoText?: string }, config: LayoutConfig, theme: RenderTheme) {
  if (layout.titlePosition && score.title) {
    ctx.fillStyle = theme.titleColor;
    ctx.font = `bold ${config.titleFontSize}px "Noto Sans", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(score.title, layout.titlePosition.x + layout.titlePosition.width / 2, layout.titlePosition.y);
  }

  if (layout.keyPosition) {
    ctx.fillStyle = theme.metaColor;
    ctx.font = `${config.metaFontSize}px "Noto Sans", serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`1=${score.key}`, layout.keyPosition.x, layout.keyPosition.y);
  }

  if (layout.timeSignaturePosition) {
    ctx.fillStyle = theme.metaColor;
    ctx.font = `bold ${config.metaFontSize}px "Noto Sans", serif`;
    ctx.textAlign = 'center';

    const ts = score.timeSignature;
    const cx = layout.timeSignaturePosition.x + layout.timeSignaturePosition.width / 2;
    const topY = layout.timeSignaturePosition.y;
    const half = Math.round(config.metaFontSize * 1.1);
    const numeratorY = topY + 2;
    const lineY = topY + half;
    const denominatorY = lineY + 4;

    // 分子
    ctx.textBaseline = 'top';
    ctx.fillText(`${ts.numerator}`, cx, numeratorY);
    // 分母
    ctx.fillText(`${ts.denominator}`, cx, denominatorY);

    // 中间横线
    const lineStartX = cx - config.metaFontSize * 0.32;
    const lineEndX = cx + config.metaFontSize * 0.32;
    ctx.strokeStyle = theme.metaColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(lineStartX, lineY);
    ctx.lineTo(lineEndX, lineY);
    ctx.stroke();
  }

  if (layout.tempoPosition && score.tempo) {
    ctx.fillStyle = theme.metaColor;
    ctx.font = `${config.metaFontSize - 2}px "Noto Sans", serif`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(`♩=${score.tempo}`, layout.tempoPosition.x, layout.tempoPosition.y);
  }
}

/** 主渲染函数 */
export function render(
  ctx: CanvasRenderingContext2D,
  layout: ScoreLayout,
  score: { title?: string; key: string; timeSignature: { numerator: number; denominator: number }; tempo?: number; tempoText?: string },
  config: LayoutConfig,
  theme: RenderTheme = DEFAULT_THEME,
) {
  // 清空画布
  ctx.clearRect(0, 0, layout.width, layout.height);
  ctx.fillStyle = theme.backgroundColor;
  ctx.fillRect(0, 0, layout.width, layout.height);

  // 绘制对齐网格（横向每 10px）
  ctx.strokeStyle = 'rgba(80, 140, 200, 0.15)';
  ctx.lineWidth = 0.5;
  for (let y = 0; y < layout.height; y += 10) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(layout.width, y);
    ctx.stroke();
  }

  // 绘制元信息
  drawMeta(ctx, layout, score, config, theme);

  // 收集连音线/圆滑线/三连音数据
  const tieStarts = new Map<string, NoteLayout>();
  const slurStarts = new Map<string, NoteLayout>();
  const tripletStarts = new Map<string, NoteLayout>();

  // 遍历行
  layout.rows.forEach(row => {
    row.measures.forEach((measure, mi) => {
      const nextMeasure = mi + 1 < row.measures.length ? row.measures[mi + 1] : undefined;

      // 绘制小节线（先画，避免被音符遮挡）
      if (measure.barlinePosition) {
        drawBarline(ctx, measure.data.barline || 'single', measure.barlinePosition, theme);
      }

      // 反复跳跃记号
      if (measure.repeatEndingPosition && measure.data.repeatEnding) {
        const isLast = !nextMeasure?.data.repeatEnding ||
          JSON.stringify(nextMeasure.data.repeatEnding.numbers) !== JSON.stringify(measure.data.repeatEnding.numbers);
        drawRepeatEnding(ctx, measure.data.repeatEnding.numbers, measure.repeatEndingPosition, config, theme, isLast);
      }

      // 绘制音符 (用索引遍历，以便获取前后音符位置)
      for (let ni = 0; ni < measure.notes.length; ni++) {
        const noteLayout = measure.notes[ni];
        const nextNoteLayout = ni + 1 < measure.notes.length ? measure.notes[ni + 1] : undefined;
        const data = noteLayout.data;
        if (isNoteType(data)) {
          drawNoteNumber(ctx, data, noteLayout.position, config, theme);
          drawOctaveDots(ctx, noteLayout.upperDotPositions, theme);
          drawOctaveDots(ctx, noteLayout.lowerDotPositions, theme);
          drawDots(ctx, noteLayout.dotPositions, config, theme);
          if (data.accidental && noteLayout.accidentalPosition) {
            drawAccidental(ctx, data.accidental, noteLayout.accidentalPosition, config, theme);
          }
        } else {
          drawDash(ctx, noteLayout.position, config, theme);
        }
        drawUnderlines(ctx, noteLayout, config, theme);

        // 技巧符号 (传入主音位置和后一个音符位置用于滑音)
        const noteCenterPos = isNoteType(data) ? { x: noteLayout.position.x + noteLayout.position.width / 2, y: noteLayout.position.y + noteLayout.position.height / 2, width: 0, height: 0 } : undefined;
        noteLayout.techniquePositions.forEach(tp => {
          drawTechnique(ctx, tp.technique, tp.position, config, theme, tp.mainNotePos || noteCenterPos, nextNoteLayout?.position);
        });

        // 重音/保持音/延长记号/顿音/换气
        if (noteLayout.accentPosition) drawAccent(ctx, noteLayout.accentPosition, config, theme);
        if (noteLayout.tenutoPosition) drawTenuto(ctx, noteLayout.tenutoPosition, theme);
        if (noteLayout.fermataPosition) drawFermata(ctx, noteLayout.fermataPosition, theme);
        if (noteLayout.staccatoPosition) drawSymbol(ctx, 11, noteLayout.staccatoPosition.x, noteLayout.staccatoPosition.y, 10, theme.symbolColor);
        if (noteLayout.breathPosition) drawSymbol(ctx, 12, noteLayout.breathPosition.x, noteLayout.breathPosition.y, 10, theme.symbolColor);
        // 力度突变 (sf/sfp/fp)
        if (isNoteType(data) && data.forceAccent) {
          ctx.fillStyle = theme.symbolColor;
          ctx.font = `bold ${config.techniqueFontSize}px "Noto Sans", serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(data.forceAccent, noteLayout.position.x + noteLayout.position.width / 2, noteLayout.position.y - 2);
        }

        // 歌词
        if (noteLayout.lyricPosition && isNoteType(data) && data.lyric) {
          ctx.fillStyle = theme.lyricColor;
          ctx.font = `${config.lyricFontSize}px "Noto Sans", sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          ctx.fillText(data.lyric, noteLayout.lyricPosition.x + noteLayout.lyricPosition.width / 2, noteLayout.lyricPosition.y);
        }

        // 收集连音线起点
        if (noteLayout.tieId) {
          if (!tieStarts.has(noteLayout.tieId)) {
            tieStarts.set(noteLayout.tieId, noteLayout);
          } else {
            drawTie(ctx, tieStarts.get(noteLayout.tieId)!, noteLayout, config, theme);
            tieStarts.delete(noteLayout.tieId);
          }
        }
        if (noteLayout.slurId) {
          if (!slurStarts.has(noteLayout.slurId)) {
            slurStarts.set(noteLayout.slurId, noteLayout);
          } else {
            drawSlur(ctx, slurStarts.get(noteLayout.slurId)!, noteLayout, config, theme);
            slurStarts.delete(noteLayout.slurId);
          }
        }
        if (noteLayout.tripletId) {
          if (!tripletStarts.has(noteLayout.tripletId)) {
            tripletStarts.set(noteLayout.tripletId, noteLayout);
          } else {
            drawTriplet(ctx, tripletStarts.get(noteLayout.tripletId)!, noteLayout, config, theme);
            tripletStarts.delete(noteLayout.tripletId);
          }
        }
      }
    });
  });
}
