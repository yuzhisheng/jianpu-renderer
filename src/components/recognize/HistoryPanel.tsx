import { AlertCircle, CheckCircle2, Clock3, History, Loader2, X } from 'lucide-react';
import type { RecognitionHistoryItem } from '../../api/recognize';

interface HistoryPanelProps {
  isDarkTheme: boolean;
  items: RecognitionHistoryItem[];
  loading: boolean;
  error: string | null;
  onSelect: (item: RecognitionHistoryItem) => void;
  onClose: () => void;
}

function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  });
}

function statusLabel(status: RecognitionHistoryItem['status']): string {
  if (status === 'succeeded') return '已完成';
  if (status === 'failed') return '失败';
  if (status === 'running') return '识别中';
  return status;
}

export default function HistoryPanel({
  isDarkTheme, items, loading, error, onSelect, onClose,
}: HistoryPanelProps) {
  const panel = isDarkTheme ? '#252526' : '#ffffff';
  const border = isDarkTheme ? '#3e3e42' : '#e5e7eb';
  const muted = isDarkTheme ? '#9ca3af' : '#6b7280';
  const foreground = isDarkTheme ? '#f3f4f6' : '#111827';

  return (
    <aside
      className="absolute right-0 top-10 bottom-0 z-30 flex w-[min(92vw,390px)] flex-col border-l shadow-2xl"
      style={{ backgroundColor: panel, borderColor: border }}
      aria-label="识别历史记录"
    >
      <div className="flex h-11 shrink-0 items-center justify-between border-b px-4" style={{ borderColor: border }}>
        <div className="flex items-center gap-2 text-sm font-semibold" style={{ color: foreground }}>
          <History size={16} />
          识别历史
        </div>
        <button onClick={onClose} className="rounded p-1 hover:bg-black/10" title="关闭">
          <X size={16} style={{ color: muted }} />
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {loading && (
          <div className="flex items-center gap-2 px-3 py-4 text-xs" style={{ color: muted }}>
            <Loader2 size={14} className="animate-spin" /> 正在读取历史记录…
          </div>
        )}
        {error && (
          <div className="m-2 flex items-start gap-2 rounded border border-red-200 bg-red-50 p-3 text-xs text-red-700">
            <AlertCircle size={14} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
        {!loading && !error && items.length === 0 && (
          <div className="px-3 py-8 text-center text-xs" style={{ color: muted }}>
            还没有识别记录。上传一张简谱后，结果会自动保存到这里。
          </div>
        )}
        <div className="space-y-1">
          {items.map(item => (
            <button
              key={item.id}
              onClick={() => onSelect(item)}
              className="w-full rounded border px-3 py-2 text-left transition hover:bg-blue-500/10"
              style={{ borderColor: border }}
            >
              <div className="flex items-start justify-between gap-2">
                <span className="truncate text-xs font-medium" style={{ color: foreground }} title={item.title || item.original_filename}>
                  {item.title || item.original_filename || '未命名图片'}
                </span>
                <span
                  className={`flex shrink-0 items-center gap-1 text-[10px] ${item.status === 'succeeded' ? 'text-emerald-500' : item.status === 'failed' ? 'text-red-500' : 'text-amber-500'}`}
                >
                  {item.status === 'succeeded' ? <CheckCircle2 size={12} /> : item.status === 'failed' ? <AlertCircle size={12} /> : <Clock3 size={12} />}
                  {statusLabel(item.status)}
                </span>
              </div>
              <div className="mt-1 flex items-center justify-between gap-2 text-[10px]" style={{ color: muted }}>
                <span>{formatTime(item.created_at)}</span>
                <span>{item.recognizer === 'accurate' ? '精确模式' : item.recognizer === 'fast' ? '快速模式' : item.recognizer || '—'}</span>
              </div>
              {(item.notes != null || item.lyric_syllables != null) && (
                <div className="mt-1 text-[10px]" style={{ color: muted }}>
                  {item.notes != null ? `${item.notes} 音符` : ''}
                  {item.notes != null && item.lyric_syllables != null ? ' · ' : ''}
                  {item.lyric_syllables != null ? `${item.lyric_syllables} 个歌词音节` : ''}
                </div>
              )}
              {item.inference_ms != null && (
                <div className="mt-1 text-[10px]" style={{ color: muted }}>
                  耗时 {item.inference_ms >= 10000 ? `${(item.inference_ms / 1000).toFixed(1)} 秒` : `${item.inference_ms.toFixed(0)} ms`}
                </div>
              )}
              {item.error && <div className="mt-1 truncate text-[10px] text-red-500" title={item.error}>{item.error}</div>}
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}
