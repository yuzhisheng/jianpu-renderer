import type {
  ScoreLayout, RowLayout, MeasureLayout, NoteLayout, SymbolPosition,
  Note, Dash, DiziTechnique,
} from '../types';
import type { LayoutConfig } from './layout';

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
  noteColor: '#1a1a1a',
  symbolColor: '#1a1a1a',
  barlineColor: '#1a1a1a',
  titleColor: '#1a1a1a',
  metaColor: '#333333',
  lyricColor: '#444444',
  techniqueColor: '#c0392b',
  backgroundColor: '#ffffff',
  dashColor: '#1a1a1a',
  dotColor: '#1a1a1a',
  underlineColor: '#1a1a1a',
  tieColor: '#1a1a1a',
  repeatDotColor: '#1a1a1a',
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

/** 绘制附点 */
function drawDots(ctx: CanvasRenderingContext2D, positions: SymbolPosition[], theme: RenderTheme) {
  ctx.fillStyle = theme.dotColor;
  positions.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x + p.width / 2, p.y + p.height / 2, p.width / 2, 0, Math.PI * 2);
    ctx.fill();
  });
}

/** 绘制升降号 */
function drawAccidental(ctx: CanvasRenderingContext2D, accidental: string, pos: SymbolPosition, config: LayoutConfig, theme: RenderTheme) {
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
function drawTechnique(ctx: CanvasRenderingContext2D, tech: DiziTechnique, pos: SymbolPosition, config: LayoutConfig, theme: RenderTheme, mainNotePos?: SymbolPosition) {
  ctx.fillStyle = theme.techniqueColor;
  ctx.font = `bold ${config.techniqueFontSize}px "Noto Sans", "SimSun", serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';

  const labels: Record<string, string> = {
    zengyin: '赠',
    dieyin: '又',
    liyin: tech.liyinDirection === 'up' ? '历↑' : tech.liyinDirection === 'down' ? '历↓' : '历',
    huayin: tech.slideDirection === 'up' ? '↗' : '↘',
    dayin: '扌',
    yinyin: '倚',
    chanyin: 'tr',
    qizhenyin: '﹏',
    tuyin: tech.articulation === 'double' ? 'TK' : tech.articulation === 'triple' ? 'TKT' : 'T',
    huashe: '✱',
    xunhuan: '↻',
    fanyin: '○',
  };

  const label = labels[tech.type] || tech.type;

  // 颤音 - 只显示 tr
  if (tech.type === 'chanyin') {
    ctx.fillText(label, pos.x, pos.y + pos.height);
    return;
  }

  // 气震音波浪线
  if (tech.type === 'qizhenyin') {
    const startX = pos.x;
    const endX = pos.x + pos.width + 8;
    const y = pos.y + pos.height / 2;
    ctx.strokeStyle = theme.techniqueColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let x = startX; x <= endX; x += 2) {
      const dy = Math.sin((x - startX) * 1.2) * 4;
      if (x === startX) ctx.moveTo(x, y + dy);
      else ctx.lineTo(x, y + dy);
    }
    ctx.stroke();
    return;
  }

  // 倚音 - 小字标注在左上，减时线，斜线连主音
  if (tech.type === 'yinyin' && tech.graceNotes?.length) {
    const gFontSize = config.techniqueFontSize + 1;
    const graceText = tech.graceNotes.join('');
    const textBottom = pos.y + pos.height - 6; // 留空给减时线

    // 小字
    ctx.fillStyle = theme.noteColor;
    ctx.font = `bold ${gFontSize}px "Noto Sans", serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(graceText, pos.x + pos.width / 2, textBottom);

    // 减时线（双线）
    ctx.strokeStyle = theme.underlineColor;
    ctx.lineWidth = 1;
    const lineStartX = pos.x + 1;
    const lineEndX = pos.x + pos.width - 1;
    const lineY1 = textBottom + 2;
    const lineY2 = textBottom + 4.5;
    ctx.beginPath();
    ctx.moveTo(lineStartX, lineY1);
    ctx.lineTo(lineEndX, lineY1);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(lineStartX, lineY2);
    ctx.lineTo(lineEndX, lineY2);
    ctx.stroke();

    // 弧线从减时线下方向下再向右弯到主音
    if (mainNotePos) {
      const graceX = pos.x + 2;
      const graceY = textBottom + 6.5; // 双横线下方
      const mainX = mainNotePos.x - config.noteWidth * 0.2;
      const mainY = mainNotePos.y + 2;

      ctx.strokeStyle = theme.tieColor;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(graceX, graceY);
      ctx.quadraticCurveTo(graceX, mainY + 4, mainX, mainY);
      ctx.stroke();
    }
    return;
  }

  ctx.fillText(label, pos.x, pos.y + pos.height);
}

/** 绘制重音标记 */
function drawAccent(ctx: CanvasRenderingContext2D, pos: SymbolPosition, theme: RenderTheme) {
  ctx.fillStyle = theme.symbolColor;
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y + pos.height / 2);
  ctx.lineTo(pos.x + pos.width / 2, pos.y);
  ctx.lineTo(pos.x + pos.width, pos.y + pos.height / 2);
  ctx.closePath();
  ctx.fill();
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
  ctx.strokeStyle = theme.symbolColor;
  ctx.lineWidth = 1.5;
  const cx = pos.x + pos.width / 2;
  const topY = pos.y;
  const bottomY = pos.y + pos.height - 4;

  // 弧线
  ctx.beginPath();
  ctx.moveTo(pos.x, bottomY);
  ctx.quadraticCurveTo(cx, topY - 2, pos.x + pos.width, bottomY);
  ctx.stroke();

  // 中心点
  ctx.fillStyle = theme.symbolColor;
  ctx.beginPath();
  ctx.arc(cx, bottomY + 2, 2, 0, Math.PI * 2);
  ctx.fill();
}

/** 绘制反复跳跃记号 */
function drawRepeatEnding(ctx: CanvasRenderingContext2D, numbers: number[], pos: SymbolPosition, config: LayoutConfig, theme: RenderTheme) {
  const text = numbers.join('.');
  ctx.strokeStyle = theme.symbolColor;
  ctx.lineWidth = 1;

  // 上方横线
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y + pos.height / 2);
  ctx.lineTo(pos.x + pos.width, pos.y + pos.height / 2);
  ctx.stroke();
  // 左端竖线
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
  ctx.lineTo(pos.x, pos.y + pos.height);
  ctx.stroke();

  ctx.fillStyle = theme.symbolColor;
  ctx.font = `${config.techniqueFontSize}px "Noto Sans", sans-serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'bottom';
  ctx.fillText(text, pos.x + 4, pos.y + pos.height / 2 - 2);
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

  // 绘制元信息
  drawMeta(ctx, layout, score, config, theme);

  // 收集连音线/圆滑线/三连音数据
  const tieStarts = new Map<string, NoteLayout>();
  const slurStarts = new Map<string, NoteLayout>();
  const tripletStarts = new Map<string, NoteLayout>();

  // 遍历行
  layout.rows.forEach(row => {
    row.measures.forEach(measure => {
      // 绘制小节线（先画，避免被音符遮挡）
      if (measure.barlinePosition) {
        drawBarline(ctx, measure.data.barline || 'single', measure.barlinePosition, theme);
      }

      // 反复跳跃记号
      if (measure.repeatEndingPosition && measure.data.repeatEnding) {
        drawRepeatEnding(ctx, measure.data.repeatEnding.numbers, measure.repeatEndingPosition, config, theme);
      }

      // 绘制音符
      measure.notes.forEach(noteLayout => {
        const data = noteLayout.data;
        if (isNoteType(data)) {
          drawNoteNumber(ctx, data, noteLayout.position, config, theme);
          drawOctaveDots(ctx, noteLayout.upperDotPositions, theme);
          drawOctaveDots(ctx, noteLayout.lowerDotPositions, theme);
          drawDots(ctx, noteLayout.dotPositions, theme);
          if (data.accidental && noteLayout.accidentalPosition) {
            drawAccidental(ctx, data.accidental, noteLayout.accidentalPosition, config, theme);
          }
        } else {
          drawDash(ctx, noteLayout.position, config, theme);
        }
        drawUnderlines(ctx, noteLayout, config, theme);

        // 技巧符号
        noteLayout.techniquePositions.forEach(tp => {
          drawTechnique(ctx, tp.technique, tp.position, config, theme, tp.mainNotePos);
        });

        // 重音/保持音/延长记号
        if (noteLayout.accentPosition) drawAccent(ctx, noteLayout.accentPosition, theme);
        if (noteLayout.tenutoPosition) drawTenuto(ctx, noteLayout.tenutoPosition, theme);
        if (noteLayout.fermataPosition) drawFermata(ctx, noteLayout.fermataPosition, theme);

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
      });
    });
  });
}
