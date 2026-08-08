import { useMemo } from 'react';
import { ChevronDown, ChevronUp, CirclePlus, Copy, Minus, Plus, Redo2, Trash2, Undo2 } from 'lucide-react';
import type { Accidental, BarlineType, Note, Score } from '../types';
import {
  DURATIONS,
  getSelectedNote,
  measureDuration,
  validateScore,
  type NoteAddress,
} from '../editor/score';

interface ScoreEditorPanelProps {
  score: Score;
  selected: NoteAddress | null;
  isDarkTheme: boolean;
  canUndo: boolean;
  canRedo: boolean;
  onSelect: (address: NoteAddress | null) => void;
  onUpdateNote: (patch: Partial<Note>) => void;
  onUpdateScore: (patch: Partial<Score>) => void;
  onUpdateMeasure: (patch: { barline?: BarlineType }) => void;
  onInsert: () => void;
  onDelete: () => void;
  onAddMeasure: () => void;
  onUndo: () => void;
  onRedo: () => void;
}

const pitches = ['0', '1', '2', '3', '4', '5', '6', '7'];
const durationLabels: Record<number, string> = { 0.25: '16分', 0.5: '8分', 1: '4分', 2: '2分', 4: '全音' };

export default function ScoreEditorPanel({
  score,
  selected,
  isDarkTheme,
  canUndo,
  canRedo,
  onSelect,
  onUpdateNote,
  onUpdateScore,
  onUpdateMeasure,
  onInsert,
  onDelete,
  onAddMeasure,
  onUndo,
  onRedo,
}: ScoreEditorPanelProps) {
  const note = getSelectedNote(score, selected);
  const issues = useMemo(() => validateScore(score), [score]);
  const measure = selected ? score.measures[selected.measureIndex] : undefined;
  const field = isDarkTheme
    ? 'bg-gray-800 border-gray-700 text-gray-200'
    : 'bg-white border-gray-300 text-gray-800';
  const muted = isDarkTheme ? 'text-gray-500' : 'text-gray-400';
  const panel = isDarkTheme ? 'text-gray-200' : 'text-gray-800';
  const button = isDarkTheme
    ? 'border-gray-700 bg-gray-800 text-gray-300 hover:bg-gray-700'
    : 'border-gray-300 bg-white text-gray-600 hover:bg-gray-100';
  const active = isDarkTheme
    ? 'border-primary-500 bg-primary-500/20 text-primary-300'
    : 'border-primary-500 bg-primary-50 text-primary-700';

  return (
    <div className={`h-full overflow-y-auto p-3 space-y-4 text-xs ${panel}`}>
      <div className="flex items-center justify-between">
        <div>
          <div className="font-semibold">可视化编辑</div>
          <div className={`mt-0.5 ${muted}`}>点击右侧谱面选择音符</div>
        </div>
        <div className="flex gap-1">
          <button className={`p-1.5 rounded border disabled:opacity-30 ${button}`} disabled={!canUndo} onClick={onUndo} title="撤销"><Undo2 size={14} /></button>
          <button className={`p-1.5 rounded border disabled:opacity-30 ${button}`} disabled={!canRedo} onClick={onRedo} title="重做"><Redo2 size={14} /></button>
        </div>
      </div>

      <section className="space-y-2">
        <div className={`text-[10px] uppercase tracking-wider ${muted}`}>谱头</div>
        <input className={`w-full rounded border px-2 py-1.5 outline-none focus:border-primary-500 ${field}`} value={score.title || ''} onChange={e => onUpdateScore({ title: e.target.value })} placeholder="乐谱标题" />
        <div className="grid grid-cols-2 gap-2">
          <label className="space-y-1"><span className={muted}>调号</span><input className={`w-full rounded border px-2 py-1.5 outline-none focus:border-primary-500 ${field}`} value={score.key} onChange={e => onUpdateScore({ key: e.target.value })} /></label>
          <label className="space-y-1"><span className={muted}>速度 BPM</span><input type="number" min="20" max="300" className={`w-full rounded border px-2 py-1.5 outline-none focus:border-primary-500 ${field}`} value={score.tempo || ''} onChange={e => onUpdateScore({ tempo: e.target.value ? Number(e.target.value) : undefined })} /></label>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <label className="space-y-1"><span className={muted}>拍号分子</span><input type="number" min="1" max="16" className={`w-full rounded border px-2 py-1.5 outline-none focus:border-primary-500 ${field}`} value={score.timeSignature.numerator} onChange={e => onUpdateScore({ timeSignature: { ...score.timeSignature, numerator: Number(e.target.value) || 1 } })} /></label>
          <label className="space-y-1"><span className={muted}>拍号分母</span><select className={`w-full rounded border px-2 py-1.5 outline-none focus:border-primary-500 ${field}`} value={score.timeSignature.denominator} onChange={e => onUpdateScore({ timeSignature: { ...score.timeSignature, denominator: Number(e.target.value) } })}><option value={2}>2</option><option value={4}>4</option><option value={8}>8</option><option value={16}>16</option></select></label>
        </div>
      </section>

      <section className="space-y-2">
        <div className={`flex items-center justify-between text-[10px] uppercase tracking-wider ${muted}`}>
          <span>音符编辑</span>
          {selected && <span>第 {selected.measureIndex + 1} 小节 · 第 {selected.noteIndex + 1} 音</span>}
        </div>
        {!note ? (
          <div className={`rounded border border-dashed p-3 text-center ${muted}`}>请选择一个音符开始编辑</div>
        ) : (
          <>
            <div className="grid grid-cols-8 gap-1">
              {pitches.map(pitch => <button key={pitch} className={`h-7 rounded border font-medium ${String(note.pitch) === pitch ? active : button}`} onClick={() => onUpdateNote({ pitch: Number(pitch) as Note['pitch'] })}>{pitch}</button>)}
            </div>
            <div className="grid grid-cols-5 gap-1">
              {DURATIONS.map(duration => <button key={duration} className={`rounded border px-1 py-1.5 ${note.duration === duration ? active : button}`} onClick={() => onUpdateNote({ duration })}>{durationLabels[duration]}</button>)}
            </div>
            <div className="grid grid-cols-4 gap-1">
              {([undefined, 'sharp', 'flat', 'natural'] as const).map((accidental: Accidental | undefined) => <button key={accidental || 'none'} className={`rounded border px-1 py-1.5 ${note.accidental === accidental ? active : button}`} onClick={() => onUpdateNote({ accidental })}>{accidental === 'sharp' ? '♯ 升' : accidental === 'flat' ? '♭ 降' : accidental === 'natural' ? '♮ 还原' : '无变音'}</button>)}
            </div>
            <div className="flex gap-1">
              <button className={`flex-1 rounded border py-1.5 ${button}`} onClick={() => onUpdateNote({ octave: Math.max(-2, (note.octave || 0) - 1) as Note['octave'] })}><ChevronDown size={14} className="mx-auto" /><span className="sr-only">降低八度</span></button>
              <div className={`flex flex-1 items-center justify-center rounded border ${field}`}>八度 {note.octave || 0}</div>
              <button className={`flex-1 rounded border py-1.5 ${button}`} onClick={() => onUpdateNote({ octave: Math.min(2, (note.octave || 0) + 1) as Note['octave'] })}><ChevronUp size={14} className="mx-auto" /><span className="sr-only">升高八度</span></button>
              <button className={`rounded border px-2 ${note.dot ? active : button}`} onClick={() => onUpdateNote({ dot: note.dot ? 0 : 1 })}>· 附点</button>
            </div>
            <input className={`w-full rounded border px-2 py-1.5 outline-none focus:border-primary-500 ${field}`} value={note.lyric || ''} onChange={e => onUpdateNote({ lyric: e.target.value || undefined })} placeholder="当前音符歌词" />
            <div className="flex gap-1">
              <button className={`flex flex-1 items-center justify-center gap-1 rounded border py-1.5 ${button}`} onClick={onInsert}><Copy size={13} />后插入音符</button>
              <button className={`flex items-center justify-center gap-1 rounded border px-3 py-1.5 text-red-400 ${button}`} onClick={onDelete}><Trash2 size={13} />删除</button>
            </div>
          </>
        )}
      </section>

      <section className="space-y-2">
        <div className={`text-[10px] uppercase tracking-wider ${muted}`}>小节</div>
        {measure && selected && <div className="flex items-center gap-2"><span className={muted}>第 {selected.measureIndex + 1} 小节</span><select className={`flex-1 rounded border px-2 py-1.5 ${field}`} value={measure.barline || 'single'} onChange={e => onUpdateMeasure({ barline: e.target.value as BarlineType })}><option value="single">普通小节线</option><option value="double">双小节线</option><option value="end">终止线</option><option value="repeat-start">反复开始</option><option value="repeat-end">反复结束</option><option value="none">不显示</option></select></div>}
        <button className={`flex w-full items-center justify-center gap-1 rounded border py-1.5 ${button}`} onClick={onAddMeasure}><CirclePlus size={14} />添加小节</button>
      </section>

      <section className="space-y-2">
        <div className="flex items-center justify-between"><div className={`text-[10px] uppercase tracking-wider ${muted}`}>规则检查</div><div className={issues.length ? 'text-amber-400' : 'text-green-400'}>{issues.length ? `${issues.length} 项提示` : '✓ 正常'}</div></div>
        {selected && <div className={muted}>当前小节时值：{measure ? measureDuration(measure) : 0}</div>}
        {issues.slice(0, 4).map((issue, index) => <div key={`${issue.measureIndex}-${index}`} className={`rounded border px-2 py-1.5 ${issue.severity === 'error' ? 'border-red-500/40 text-red-400' : 'border-amber-500/40 text-amber-400'}`} onClick={() => issue.measureIndex !== undefined && onSelect({ measureIndex: issue.measureIndex, noteIndex: 0 })}>{issue.message}</div>)}
      </section>

      <div className={`flex items-center gap-2 border-t pt-3 ${isDarkTheme ? 'border-gray-800' : 'border-gray-200'}`}>
        <Plus size={12} className={muted} /> <span className={muted}>点击音符即可定位并编辑</span>
        <Minus size={12} className={`${muted} ml-auto`} />
      </div>
    </div>
  );
}
