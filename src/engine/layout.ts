import type {
  Score, Measure, Note as NoteType, Dash,
  ScoreLayout, RowLayout, MeasureLayout, NoteLayout, SymbolPosition,
  DiziTechnique,
} from '../types';

/** 布局配置参数 */
export interface LayoutConfig {
  /** 画布宽度 */
  canvasWidth: number;
  /** 左右边距 */
  paddingHorizontal: number;
  /** 上下边距 */
  paddingVertical: number;
  /** 音符主体字体大小 */
  noteFontSize: number;
  /** 音符宽度（占位宽度） */
  noteWidth: number;
  /** 音符高度 */
  noteHeight: number;
  /** 小节间距 */
  measureGap: number;
  /** 行间距 */
  rowGap: number;
  /** 八度点直径 */
  dotRadius: number;
  /** 八度点间距 */
  dotGap: number;
  /** 附点直径 */
  accentDotRadius: number;
  /** 减时线偏移 */
  underlineOffset: number;
  /** 减时线间距 */
  underlineGap: number;
  /** 减时线粗细 */
  underlineThickness: number;
  /** 增时线粗细 */
  dashThickness: number;
  /** 小节线宽度 */
  barlineWidth: number;
  /** 技巧符号字体大小 */
  techniqueFontSize: number;
  /** 技巧符号偏移 */
  techniqueOffset: number;
  /** 标题字体大小 */
  titleFontSize: number;
  /** 元信息字体大小 */
  metaFontSize: number;
  /** 连音线弧度 */
  tieCurveHeight: number;
  /** 歌词字体大小 */
  lyricFontSize: number;
  /** 歌词偏移 */
  lyricOffset: number;
}

export const DEFAULT_CONFIG: LayoutConfig = {
  canvasWidth: 800,
  paddingHorizontal: 40,
  paddingVertical: 30,
  noteFontSize: 28,
  noteWidth: 36,
  noteHeight: 36,
  measureGap: 10,
  rowGap: 24,
  dotRadius: 2,
  dotGap: 3,
  accentDotRadius: 2,
  underlineOffset: 1,
  underlineGap: 2.5,
  underlineThickness: 1.5,
  dashThickness: 2,
  barlineWidth: 1.5,
  techniqueFontSize: 11,
  techniqueOffset: 0,
  titleFontSize: 22,
  metaFontSize: 16,
  tieCurveHeight: 8,
  lyricFontSize: 11,
  lyricOffset: 2,
};

function isNote(item: NoteType | Dash): item is NoteType {
  return 'pitch' in item;
}

/** 计算单个音符的布局 */
function layoutNote(
  item: NoteType | Dash,
  x: number,
  y: number,
  config: LayoutConfig,
  groupInfo?: { underlineLevel: number; groupStartX: number; groupWidth: number },
): NoteLayout {
  const pos: SymbolPosition = { x, y, width: config.noteWidth, height: config.noteHeight };
  const upperDots: SymbolPosition[] = [];
  const lowerDots: SymbolPosition[] = [];
  const dotPositions: SymbolPosition[] = [];

  if (isNote(item)) {
    const note = item;
    // 八度点
    const octave = note.octave || 0;
    if (octave > 0) {
      for (let i = 0; i < octave; i++) {
        upperDots.push({
          x: x + config.noteWidth / 2 - config.dotRadius,
          y: y - config.dotRadius + 3 - (i > 0 ? config.dotGap * i : 0),
          width: config.dotRadius * 2,
          height: config.dotRadius * 2,
        });
      }
    }
    if (octave < 0) {
      for (let i = 0; i < -octave; i++) {
        lowerDots.push({
          x: x + config.noteWidth / 2 - config.dotRadius,
          y: y + config.noteHeight + config.dotGap * i,
          width: config.dotRadius * 2,
          height: config.dotRadius * 2,
        });
      }
    }

    // 附点
    const dotCount = note.dot || 0;
    for (let i = 0; i < dotCount; i++) {
      dotPositions.push({
        x: x + config.noteWidth + 4 + i * (config.accentDotRadius * 2 + 3),
        y: y + config.noteHeight / 2 - config.accentDotRadius,
        width: config.accentDotRadius * 2,
        height: config.accentDotRadius * 2,
      });
    }

    // 升降号位置
    let accidentalPos: SymbolPosition | undefined;
    if (note.accidental) {
      accidentalPos = {
        x: x - 14,
        y: y,
        width: 14,
        height: config.noteHeight,
      };
    }

    // 减时线
    const underlines: { y: number; width: number; xOffset: number }[] = [];
    if (groupInfo && groupInfo.underlineLevel > 0) {
      for (let i = 0; i < groupInfo.underlineLevel; i++) {
        underlines.push({
          y: y + config.noteHeight + config.underlineOffset + config.underlineGap * i,
          width: groupInfo.groupWidth,
          xOffset: groupInfo.groupStartX - x,
        });
      }
    } else if (!groupInfo) {
      // 独立音符（不在分组中）
      const levels = getUnderlineLevel(note.duration);
      for (let i = 0; i < levels; i++) {
        underlines.push({
          y: y + config.noteHeight + config.underlineOffset + config.underlineGap * i,
          width: config.noteWidth,
          xOffset: 0,
        });
      }
    }

    // 技巧符号位置（统一高度，倚音特殊处理）
    const techniquePositions: { technique: DiziTechnique; position: SymbolPosition }[] = [];
    const techYBase = y - (config.techniqueOffset + config.techniqueFontSize);
    if (note.techniques) {
      note.techniques.forEach((tech, idx) => {
        if (tech.type === 'yinyin') {
          const notes = tech.graceNotes || [];
          const label = notes.join('');
          const gFontSize = config.techniqueFontSize + 1;
          const width = label.length * gFontSize * 0.6 + 2;
          techniquePositions.push({
            technique: tech,
            position: {
              x: x - width + 4,
              y: y - gFontSize - 2,
              width,
              height: gFontSize + 6,
            },
            mainNotePos: { x: x + config.noteWidth / 2, y: y + 2, width: 0, height: 0 },
          });
          return;
        }
        const label = getTechniqueLabel(tech);
        const width = label.length * config.techniqueFontSize * 0.7;
        techniquePositions.push({
          technique: tech,
          position: {
            x: x + config.noteWidth / 2 - width / 2,
            y: techYBase,
            width,
            height: config.techniqueFontSize,
          },
        });
      });
    }

    // 重音、保持音、延长记号（统一在技法符号上方）
    const hasTechniques = (note.techniques?.length || 0) > 0;
    const accentY = hasTechniques
      ? techYBase - config.techniqueFontSize - 2
      : y - 2;

    const accentPosition = note.accent ? {
      x: x + config.noteWidth / 2 - 5,
      y: accentY,
      width: 10, height: 8,
    } : undefined;

    const tenutoPosition = note.tenuto ? {
      x: x + config.noteWidth / 2 - 8,
      y: y - 2,
      width: 16, height: 3,
    } : undefined;

    const fermataPosition = note.fermata ? {
      x: x + config.noteWidth / 2 - 10,
      y: hasTechniques ? accentY - 10 : y - 12,
      width: 20, height: 14,
    } : undefined;

    // 歌词
    const lyricPosition = note.lyric ? {
      x: x + config.noteWidth / 2 - note.lyric.length * config.lyricFontSize * 0.5,
      y: y + config.noteHeight + config.lyricOffset + (groupInfo?.underlineLevel || 0) * config.underlineGap + config.underlineOffset + config.lyricFontSize,
      width: note.lyric.length * config.lyricFontSize,
      height: config.lyricFontSize,
    } : undefined;

    return {
      type: 'note',
      data: note,
      position: pos,
      upperDotPositions: upperDots,
      lowerDotPositions: lowerDots,
      dotPositions,
      accidentalPosition: accidentalPos,
      underlines,
      techniquePositions,
      tieStart: !!note.tieId,
      tieEnd: !!note.tieId,
      tieId: note.tieId,
      slurStart: !!note.slurId,
      slurEnd: !!note.slurId,
      slurId: note.slurId,
      tripletId: note.tripletId,
      tripletStart: !!note.tripletId,
      tripletEnd: !!note.tripletId,
      accentPosition,
      tenutoPosition,
      fermataPosition,
      lyricPosition,
    };
  }

  // Dash (增时线)
  return {
    type: 'dash',
    data: item,
    position: pos,
    upperDotPositions: upperDots,
    lowerDotPositions: lowerDots,
    dotPositions,
    underlines: [],
    techniquePositions: [],
    tieId: item.tieId,
  };
}

/** 获取技巧符号显示文本 */
function getTechniqueLabel(tech: DiziTechnique): string {
  const labels: Record<string, string> = {
    zengyin: '赠',
    dieyin: '又',
    liyin: tech.liyinDirection === 'up' ? '历↑' : '历↓',
    huayin: tech.slideDirection === 'up' ? '↗' : '↘',
    dayin: '扌',
    yinyin: '倚',
    chanyin: 'tr',
    qizhenyin: '〰',
    tuyin: tech.articulation === 'double' ? 'TK' : tech.articulation === 'triple' ? 'TKT' : 'T',
    huashe: '✱',
    xunhuan: '↻',
    fanyin: 'o',
  };
  return labels[tech.type] || tech.type;
}

/** 分析小节中的减时线分组 */
interface UnderlineGroup {
  startIndex: number;
  endIndex: number;
  level: number;
}

function analyzeUnderlineGroups(notes: (NoteType | Dash)[]): UnderlineGroup[] {
  const groups: UnderlineGroup[] = [];
  // 将连续的短时值音符（duration <= 0.5）分为一组
  // 每组的 underline level 取组内最低 level（即最大时值对应的 level）
  let i = 0;
  while (i < notes.length) {
    const item = notes[i];
    if (!isNote(item)) { i++; continue; }
    const level = getUnderlineLevel(item.duration);
    if (level > 0) {
      const start = i;
      let end = i;
      let minLevel = level;
      while (end + 1 < notes.length && isNote(notes[end + 1])) {
        const nextLevel = getUnderlineLevel((notes[end + 1] as NoteType).duration);
        if (nextLevel > 0) {
          minLevel = Math.min(minLevel, nextLevel);
          end++;
        } else {
          break;
        }
      }
      groups.push({ startIndex: start, endIndex: end, level: minLevel });
      i = end + 1;
    } else {
      i++;
    }
  }
  return groups;
}

function getUnderlineLevel(duration: number): number {
  if (duration <= 0.125) return 3;
  if (duration <= 0.25) return 2;
  if (duration <= 0.5) return 1;
  return 0;
}

/** 计算一行中能容纳的小节数 */
function fitMeasuresInRow(
  measures: Measure[],
  startIdx: number,
  availWidth: number,
  config: LayoutConfig,
): number {
  let width = 0;
  let count = 0;
  for (let i = startIdx; i < measures.length; i++) {
    const m = measures[i];
    const noteCount = m.notes.length;
    const dashCount = m.notes.filter(n => !isNote(n)).length;
    let mWidth = noteCount * config.noteWidth;
    // 附点额外宽度
    m.notes.forEach(n => {
      if (isNote(n) && (n.dot || 0) > 0) {
        mWidth += (n.dot || 0) * (config.accentDotRadius * 2 + 6);
      }
      if (isNote(n) && n.accidental) {
        mWidth += 14;
      }
    });
    mWidth += config.barlineWidth;
    if (count > 0) mWidth += config.measureGap;
    if (width + mWidth > availWidth && count > 0) break;
    width += mWidth;
    count++;
  }
  return Math.max(count, 1);
}

/** 主布局计算函数 */
export function calculateLayout(score: Score, config: LayoutConfig = DEFAULT_CONFIG): ScoreLayout {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  const contentWidth = cfg.canvasWidth - cfg.paddingHorizontal * 2;
  let currentY = cfg.paddingVertical;

  // 标题
  const titlePosition: SymbolPosition | undefined = score.title ? {
    x: cfg.paddingHorizontal,
    y: currentY,
    width: contentWidth,
    height: cfg.titleFontSize + 8,
  } : undefined;
  if (titlePosition) currentY += titlePosition.height + 8;

  // 调号、拍号、速度标记
  const metaY = currentY;
  const keyText = `1=${score.key}`;
  const tempoText = score.tempo ? `♩=${score.tempo}` : '';

  const keyPosition: SymbolPosition = {
    x: cfg.paddingHorizontal,
    y: metaY,
    width: keyText.length * cfg.metaFontSize * 0.7,
    height: cfg.metaFontSize + 4,
  };
  // 拍号放在调号右边，垂直居中对齐
  const tsDigitWidth = cfg.metaFontSize * 0.65;
  const tsLineWidth = tsDigitWidth + 4;
  const tsTotalHeight = cfg.metaFontSize * 2 + 10; // 两个数字 + 横线 + 间距
  const keyCenterY = metaY + (cfg.metaFontSize + 4) / 2;
  const timeSignaturePosition: SymbolPosition = {
    x: keyPosition.x + keyPosition.width + 6,
    y: keyCenterY - tsTotalHeight / 2,
    width: tsLineWidth,
    height: tsTotalHeight,
  };
  // 速度标记放在调号下方，左对齐
  let tempoPosition: SymbolPosition | undefined;
  if (tempoText) {
    tempoPosition = {
      x: cfg.paddingHorizontal,
      y: metaY + (cfg.metaFontSize + 4) + 6,
      width: tempoText.length * cfg.metaFontSize * 0.6,
      height: cfg.metaFontSize + 4,
    };
  }
  currentY = tempoPosition
    ? tempoPosition.y + tempoPosition.height + 4
    : timeSignaturePosition.y + tsTotalHeight + 4;

  // 计算行布局
  const rows: RowLayout[] = [];
  let measureIdx = 0;

  while (measureIdx < score.measures.length) {
    const fitCount = fitMeasuresInRow(score.measures, measureIdx, contentWidth, cfg);
    const rowMeasures = score.measures.slice(measureIdx, measureIdx + fitCount);

    const measureLayouts: MeasureLayout[] = [];
    let currentX = cfg.paddingHorizontal;

    for (let mi = 0; mi < rowMeasures.length; mi++) {
      const m = rowMeasures[mi];
      const mStartX = currentX;

      // 分析减时线分组
      const groups = analyzeUnderlineGroups(m.notes);

      // 计算小节宽度
      let mWidth = m.notes.length * cfg.noteWidth;
      m.notes.forEach(n => {
        if (isNote(n) && (n.dot || 0) > 0) {
          mWidth += (n.dot || 0) * (cfg.accentDotRadius * 2 + 6);
        }
        if (isNote(n) && n.accidental) {
          mWidth += 14;
        }
      });
      mWidth += cfg.barlineWidth;

      // 音符布局
      const noteLayouts: NoteLayout[] = [];
      let noteX = mStartX;

      for (let ni = 0; ni < m.notes.length; ni++) {
        const item = m.notes[ni];

        // 查找是否属于某个减时线分组
        const group = groups.find(g => ni >= g.startIndex && ni <= g.endIndex);
        let groupInfo: { underlineLevel: number; groupStartX: number; groupWidth: number } | undefined;
        if (group) {
          const gNoteStartX = mStartX + group.startIndex * cfg.noteWidth;
          const gWidth = (group.endIndex - group.startIndex + 1) * cfg.noteWidth;
          groupInfo = { underlineLevel: group.level, groupStartX: gNoteStartX, groupWidth: gWidth };
        }

        const nl = layoutNote(item, noteX, currentY, cfg, groupInfo);
        noteLayouts.push(nl);

        let advanceX = cfg.noteWidth;
        if (isNote(item) && (item.dot || 0) > 0) {
          advanceX += (item.dot || 0) * (cfg.accentDotRadius * 2 + 6);
        }
        if (isNote(item) && item.accidental) {
          advanceX += 14;
        }
        noteX += advanceX;
      }

      // 小节线
      const barlinePos: SymbolPosition = {
        x: noteX,
        y: currentY - 4,
        width: cfg.barlineWidth,
        height: cfg.noteHeight + 8,
      };

      // 反复跳跃记号
      const repeatEndingPos = m.repeatEnding ? {
        x: mStartX,
        y: currentY - 24,
        width: mWidth - cfg.barlineWidth,
        height: 14,
      } as SymbolPosition : undefined;

      measureLayouts.push({
        data: m,
        position: { x: mStartX, y: currentY, width: mWidth, height: cfg.noteHeight },
        barlinePosition: barlinePos,
        notes: noteLayouts,
        repeatEndingPosition: repeatEndingPos,
      });

      currentX = noteX + cfg.barlineWidth + (mi < rowMeasures.length - 1 ? cfg.measureGap : 0);
    }

    // 计算本行最大高度
    let maxBottom = currentY + cfg.noteHeight;
    measureLayouts.forEach(ml => {
      ml.notes.forEach(nl => {
        // 考虑下加点和减时线
        if (nl.lowerDotPositions.length > 0) {
          const lastDot = nl.lowerDotPositions[nl.lowerDotPositions.length - 1];
          maxBottom = Math.max(maxBottom, lastDot.y + lastDot.height + 4);
        }
        if (nl.underlines.length > 0) {
          const lastLine = nl.underlines[nl.underlines.length - 1];
          maxBottom = Math.max(maxBottom, lastLine.y + cfg.underlineThickness + 2);
        }
        if (nl.lyricPosition) {
          maxBottom = Math.max(maxBottom, nl.lyricPosition.y + nl.lyricPosition.height + 2);
        }
      });
    });

    const rowHeight = maxBottom - currentY + 8;
    rows.push({ y: currentY, height: rowHeight, measures: measureLayouts });
    currentY += rowHeight + cfg.rowGap;
    measureIdx += fitCount;
  }

  return {
    width: cfg.canvasWidth,
    height: currentY + cfg.paddingVertical,
    titlePosition,
    keyPosition,
    timeSignaturePosition,
    tempoPosition,
    rows,
  };
}
