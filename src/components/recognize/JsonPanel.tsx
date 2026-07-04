import { useRef, useEffect, useState, useCallback } from 'react';
import type { Score } from '../../types';
import { calculateLayout, render, DEFAULT_THEME, setupCanvasDPI, DEFAULT_CONFIG } from '../../engine';
import { Copy, Check, Download, FileJson } from 'lucide-react';

interface JsonPanelProps {
  score: Score | null;
  isLoading: boolean;
  errorMessage?: string | null;
  isDarkTheme: boolean;
}

export default function JsonPanel({ score, isLoading, errorMessage, isDarkTheme }: JsonPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [copied, setCopied] = useState(false);
  const [showJson, setShowJson] = useState(false);

  // 渲染
  useEffect(() => {
    if (!score || !canvasRef.current) return;
    try {
      const config = { ...DEFAULT_CONFIG, canvasWidth: 900 };
      const layout = calculateLayout(score as any, config as any);
      canvasRef.current.width = layout.width;
      canvasRef.current.height = layout.height;
      const ctx = setupCanvasDPI(canvasRef.current, layout.width, layout.height);
      const theme = isDarkTheme ? DEFAULT_THEME : { ...DEFAULT_THEME, backgroundColor: '#ffffff', noteColor: '#000000', symbolColor: '#000000' };
      render(ctx, layout, score as any, config as any, theme as any);
    } catch (e) {
      console.error('渲染失败', e);
    }
  }, [score, isDarkTheme]);

  const handleCopy = useCallback(async () => {
    if (!score) return;
    const text = JSON.stringify(score, null, 2);
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.error('copy failed', e);
    }
  }, [score]);

  const handleDownloadJson = useCallback(() => {
    if (!score) return;
    const text = JSON.stringify(score, null, 2);
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(score as any).title || 'recognition-result'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [score]);

  const handleDownloadPng = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob(blob => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${(score as any)?.title || 'recognition-result'}.png`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }, [score]);

  if (isLoading) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3">
        <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin" />
        <p className="text-sm text-gray-500">识别中…</p>
      </div>
    );
  }

  if (errorMessage) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-8 text-center">
        <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center mb-3">
          <span className="text-red-500 text-xl">!</span>
        </div>
        <p className="text-sm text-red-500 font-medium">识别失败</p>
        <p className="text-xs text-gray-400 mt-1 max-w-md break-all">{errorMessage}</p>
      </div>
    );
  }

  if (!score) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-gray-400">
        <FileJson size={36} className="mb-2" />
        <p className="text-sm">等待识别结果…</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* 工具栏 */}
      <div className="h-9 flex items-center px-3 gap-2 border-b border-gray-200 bg-white">
        <button
          onClick={() => setShowJson(s => !s)}
          className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200"
        >
          {showJson ? '查看渲染' : '查看 JSON'}
        </button>
        <button
          onClick={handleCopy}
          className="text-xs px-2 py-1 rounded bg-blue-500 text-white hover:bg-blue-600 flex items-center gap-1"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? '已复制' : '拷贝 JSON'}
        </button>
        <button
          onClick={handleDownloadJson}
          className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200 flex items-center gap-1"
        >
          <Download size={12} /> JSON
        </button>
        <button
          onClick={handleDownloadPng}
          className="text-xs px-2 py-1 rounded bg-gray-100 hover:bg-gray-200 flex items-center gap-1"
        >
          <Download size={12} /> PNG
        </button>
        <span className="ml-auto text-xs text-gray-400">
          {score.measures?.length || 0} 小节
        </span>
      </div>

      <div className="flex-1 overflow-auto p-3" style={{ background: '#fafafa' }}>
        {showJson ? (
          <pre className="text-xs leading-relaxed font-mono whitespace-pre-wrap break-all bg-white p-3 rounded border border-gray-200">
            {JSON.stringify(score, null, 2)}
          </pre>
        ) : (
          <div className="flex items-center justify-center min-h-full">
            <canvas
              ref={canvasRef}
              style={{ maxWidth: '100%', background: '#fff', borderRadius: 4, boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
