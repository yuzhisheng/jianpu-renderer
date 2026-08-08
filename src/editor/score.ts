import type {
  Accidental,
  BarlineType,
  Dash,
  Measure,
  Note,
  Pitch,
  Score,
  TimeSignature,
} from '../types';

export interface NoteAddress {
  measureIndex: number;
  noteIndex: number;
}

export interface ValidationIssue {
  measureIndex?: number;
  noteIndex?: number;
  message: string;
  severity: 'error' | 'warning';
}

export type NotePatch = Partial<Note>;

export const DURATIONS = [0.25, 0.5, 1, 2, 4] as const;

export function isNote(item: Note | Dash): item is Note {
  return 'pitch' in item;
}

export function cloneScore(score: Score): Score {
  return JSON.parse(JSON.stringify(score)) as Score;
}

export function createNote(pitch: Pitch = 1, duration = 1): Note {
  return { pitch, duration };
}

function getMeasure(score: Score, address: NoteAddress): Measure | undefined {
  return score.measures[address.measureIndex];
}

export function getSelectedNote(score: Score, address: NoteAddress | null): Note | null {
  if (!address) return null;
  const item = getMeasure(score, address)?.notes[address.noteIndex];
  return item && isNote(item) ? item : null;
}

export function updateNote(score: Score, address: NoteAddress, patch: NotePatch): Score {
  const next = cloneScore(score);
  const measure = next.measures[address.measureIndex];
  const item = measure?.notes[address.noteIndex];
  if (!measure || !item || !isNote(item)) return score;
  measure.notes[address.noteIndex] = { ...item, ...patch };
  return next;
}

export function updateScoreMeta(score: Score, patch: Partial<Score>): Score {
  return { ...cloneScore(score), ...patch };
}

export function updateTimeSignature(score: Score, patch: Partial<TimeSignature>): Score {
  return updateScoreMeta(score, {
    timeSignature: { ...score.timeSignature, ...patch },
  });
}

export function updateMeasure(score: Score, measureIndex: number, patch: Partial<Measure>): Score {
  const next = cloneScore(score);
  const measure = next.measures[measureIndex];
  if (!measure) return score;
  next.measures[measureIndex] = { ...measure, ...patch };
  return next;
}

export function insertNoteAfter(score: Score, address: NoteAddress | null, note: Note = createNote()): { score: Score; address: NoteAddress } {
  const next = cloneScore(score);
  const measureIndex = address?.measureIndex ?? 0;
  const measure = next.measures[measureIndex] || { notes: [], barline: 'single' as BarlineType };
  if (!next.measures[measureIndex]) next.measures.push(measure);
  const insertAt = address && address.measureIndex === measureIndex
    ? Math.min(address.noteIndex + 1, measure.notes.length)
    : measure.notes.length;
  measure.notes.splice(insertAt, 0, note);
  return { score: next, address: { measureIndex, noteIndex: insertAt } };
}

export function deleteNote(score: Score, address: NoteAddress): { score: Score; address: NoteAddress | null } {
  const next = cloneScore(score);
  const measure = next.measures[address.measureIndex];
  if (!measure || !measure.notes[address.noteIndex]) return { score, address };
  measure.notes.splice(address.noteIndex, 1);

  if (measure.notes.length === 0 && next.measures.length > 1) {
    next.measures.splice(address.measureIndex, 1);
    const nextMeasureIndex = Math.max(0, Math.min(address.measureIndex, next.measures.length - 1));
    const nextMeasure = next.measures[nextMeasureIndex];
    return {
      score: next,
      address: nextMeasure?.notes.length ? { measureIndex: nextMeasureIndex, noteIndex: Math.max(0, nextMeasure.notes.length - 1) } : null,
    };
  }

  if (!measure.notes.length) return { score: next, address: null };
  return {
    score: next,
    address: { measureIndex: address.measureIndex, noteIndex: Math.min(address.noteIndex, measure.notes.length - 1) },
  };
}

export function addMeasure(score: Score): { score: Score; address: NoteAddress } {
  const next = cloneScore(score);
  const measureIndex = next.measures.length;
  next.measures.push({ notes: [createNote()], barline: 'end' });
  if (measureIndex > 0) next.measures[measureIndex - 1].barline = 'single';
  return { score: next, address: { measureIndex, noteIndex: 0 } };
}

export function cycleAccidental(accidental?: Accidental): Accidental | undefined {
  if (!accidental) return 'sharp';
  if (accidental === 'sharp') return 'flat';
  if (accidental === 'flat') return 'natural';
  return undefined;
}

function durationOf(item: Note | Dash): number {
  return item.duration;
}

export function expectedMeasureDuration(timeSignature: TimeSignature): number {
  return timeSignature.numerator * (4 / timeSignature.denominator);
}

export function measureDuration(measure: Measure): number {
  return measure.notes.reduce((total, item) => total + durationOf(item), 0);
}

export function validateScore(score: Score): ValidationIssue[] {
  const issues: ValidationIssue[] = [];
  const defaultExpected = expectedMeasureDuration(score.timeSignature);

  score.measures.forEach((measure, measureIndex) => {
    const expected = expectedMeasureDuration(measure.timeSignature || score.timeSignature);
    const actual = measureDuration(measure);
    const delta = Math.abs(actual - expected);
    if (delta > 0.001) {
      issues.push({
        measureIndex,
        message: `第 ${measureIndex + 1} 小节时值为 ${actual}，应为 ${expected}`,
        severity: actual > expected ? 'error' : 'warning',
      });
    }

    if (measure.notes.some(item => isNote(item) && (item.lyrics?.length || 0) > 1 && item.lyrics!.some(lyric => !lyric))) {
      issues.push({ measureIndex, message: '歌词中存在空歌词项', severity: 'warning' });
    }
    if (measureIndex < score.measures.length - 1 && measure.barline === 'none') {
      issues.push({ measureIndex, message: '该小节被设置为不显示小节线', severity: 'warning' });
    }
  });

  if (score.measures.length === 0) {
    issues.push({ message: '乐谱还没有小节', severity: 'warning' });
  }
  if (defaultExpected <= 0) {
    issues.push({ message: '默认拍号无效', severity: 'error' });
  }
  return issues;
}
