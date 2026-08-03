import { useState, useCallback, useRef, useEffect } from 'react';
import { ArrowLeft, LayoutPanelLeft, LayoutPanelTop, Sun, Moon, RefreshCw, AlertCircle } from 'lucide-react';
import type { Score } from '../types';
import { recognizeImage, checkHealth, RecognizeResponse } from '../api/recognize';
import ImageUploader from './recognize/ImageUploader';
import JsonPanel from './recognize/JsonPanel';

type Layout = 'horizontal' | 'vertical';
type RecognizerMode = 'accurate' | 'fast';

interface RecognizeViewProps {
  isDarkTheme: boolean;
  onToggleTheme: () => void;
  onBack: () => void;
}

export default function RecognizeView({ isDarkTheme, onToggleTheme, onBack }: RecognizeViewProps) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [layout, setLayout] = useState<Layout>('horizontal');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<RecognizeResponse | null>(null);
  const [health, setHealth] = useState<{
    yolo_loaded: boolean;
    transformer_loaded: boolean;
    accurate_vlm_available?: boolean;
  } | null>(null);
  const [recognizerMode, setRecognizerMode] = useState<RecognizerMode>('accurate');
  const [progress, setProgress] = useState<string>('');
  const recognitionActiveRef = useRef(false);

  // 健康检查
  useEffect(() => {
    checkHealth().then(setHealth).catch(e => {
      console.error('health check failed', e);
      setHealth({ yolo_loaded: false, transformer_loaded: false, accurate_vlm_available: false });
    });
  }, []);

  // 释放 preview url
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleClear = useCallback(() => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResult(null);
    setErrorMsg(null);
    setProgress('');
  }, [previewUrl]);

  const runRecognize = useCallback(async (f: File) => {
    // File/drop events can fire twice before React commits isLoading. Keep a
    // synchronous guard so one user action launches at most one VLM process.
    if (recognitionActiveRef.current) return;
    recognitionActiveRef.current = true;
    setIsLoading(true);
    setErrorMsg(null);
    setResult(null);
    setProgress('上传图片…');
    const timers: ReturnType<typeof setTimeout>[] = [];
    try {
      timers.push(setTimeout(() => setProgress(
        recognizerMode === 'accurate' ? '正在切分谱行…' : 'YOLO 检测中…'), 200));
      timers.push(setTimeout(() => setProgress(
        recognizerMode === 'accurate' ? '本地视觉大模型正在识别音符、节奏和连线（约 3–6 分钟）…' : '几何拼装中…'), 800));
      const resp = await recognizeImage(f, {
        conf: recognizerMode === 'accurate' ? 0.12 : 0.20,
        useTransformer: false,
        recognizer: recognizerMode,
      });
      setProgress('');
      setResult(resp);
    } catch (e: unknown) {
      console.error(e);
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      timers.forEach(clearTimeout);
      recognitionActiveRef.current = false;
      setProgress('');
      setIsLoading(false);
    }
  }, [recognizerMode]);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setErrorMsg(null);
    // 自动开始识别
    runRecognize(f);
  }, [previewUrl, runRecognize]);

  const handleRetry = useCallback(() => {
    if (file) runRecognize(file);
  }, [file, runRecognize]);

  const score: Score | null = result?.score || null;

  return (
    <div
      className="h-screen w-screen flex flex-col overflow-hidden transition-colors duration-200"
      style={{ backgroundColor: isDarkTheme ? '#1e1e1e' : '#f5f5f5' }}
    >
      {/* 顶部工具栏 */}
      <div
        className="h-10 flex items-center px-4 gap-3 border-b shadow-sm"
        style={{
          backgroundColor: isDarkTheme ? '#252526' : '#ffffff',
          borderColor: isDarkTheme ? '#3e3e42' : '#e5e7eb',
        }}
      >
        <button
          onClick={onBack}
          className="text-xs px-3 py-1 rounded flex items-center gap-1 hover:opacity-80"
          style={{ backgroundColor: isDarkTheme ? '#3e3e42' : '#e5e7eb' }}
        >
          <ArrowLeft size={12} /> 返回编辑器
        </button>
        <h1 className="text-sm font-semibold" style={{ color: isDarkTheme ? '#fff' : '#111' }}>
          图片简谱识别
        </h1>
        <div
          className="flex items-center rounded p-0.5 text-[11px]"
          style={{ backgroundColor: isDarkTheme ? '#1e1e1e' : '#f3f4f6' }}
        >
          <button
            onClick={() => setRecognizerMode('accurate')}
            className={`px-2 py-1 rounded ${recognizerMode === 'accurate' ? 'bg-blue-600 text-white' : 'text-gray-400'}`}
            title="本地视觉大模型，优先保证音高骨架准确"
          >
            精确模式
          </button>
          <button
            onClick={() => setRecognizerMode('fast')}
            className={`px-2 py-1 rounded ${recognizerMode === 'fast' ? 'bg-blue-600 text-white' : 'text-gray-400'}`}
            title="YOLO 几何拼装，速度快但准确率较低"
          >
            快速模式
          </button>
        </div>
        {/* 布局切换 */}
        <div
          className="flex items-center rounded p-0.5"
          style={{ backgroundColor: isDarkTheme ? '#1e1e1e' : '#f3f4f6' }}
        >
          <button
            onClick={() => setLayout('horizontal')}
            className={`p-1 rounded transition ${layout === 'horizontal'
              ? (isDarkTheme ? 'bg-blue-600 text-white' : 'bg-white text-blue-600 shadow')
              : 'text-gray-400'}`}
            title="左右对比"
          >
            <LayoutPanelLeft size={14} />
          </button>
          <button
            onClick={() => setLayout('vertical')}
            className={`p-1 rounded transition ${layout === 'vertical'
              ? (isDarkTheme ? 'bg-blue-600 text-white' : 'bg-white text-blue-600 shadow')
              : 'text-gray-400'}`}
            title="上下对比"
          >
            <LayoutPanelTop size={14} />
          </button>
        </div>
        {/* 健康指示 */}
        {health && (
          <div className="flex items-center gap-2 text-xs">
            <span
              className="flex items-center gap-1"
              style={{ color: health.yolo_loaded ? '#10b981' : '#ef4444' }}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${health.yolo_loaded ? 'bg-green-500' : 'bg-red-500'}`} />
              YOLO
            </span>
            <span
              className="flex items-center gap-1"
              style={{ color: health.accurate_vlm_available ? '#10b981' : '#f59e0b' }}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${health.accurate_vlm_available ? 'bg-green-500' : 'bg-yellow-500'}`} />
              {health.accurate_vlm_available ? '本地 VLM' : 'VLM 不可用'}
            </span>
          </div>
        )}
        <div className="flex-1" />
        {/* 主题切换 */}
        <button
          onClick={onToggleTheme}
          className="p-1.5 rounded hover:opacity-80"
          style={{ backgroundColor: isDarkTheme ? '#3e3e42' : '#e5e7eb' }}
        >
          {isDarkTheme ? <Sun size={14} /> : <Moon size={14} />}
        </button>
      </div>

      {/* 后端未启动提示 */}
      {health && !health.yolo_loaded && (
        <div
          className="px-4 py-2 flex items-center gap-2 text-xs"
          style={{
            backgroundColor: isDarkTheme ? '#3a2a1a' : '#fef3c7',
            color: isDarkTheme ? '#fbbf24' : '#92400e',
          }}
        >
          <AlertCircle size={14} />
          <span>
            后端未启动或 YOLO 模型未训练。
            请先 <code className="px-1 rounded bg-black/20">cd backend && uvicorn main:app --port 8000</code>，
            并运行训练脚本生成 <code className="px-1 rounded bg-black/20">weights/best.pt</code>。
          </span>
        </div>
      )}

      {/* 主体 */}
      <div className="flex-1 overflow-hidden flex" style={{ flexDirection: layout === 'horizontal' ? 'row' : 'column' }}>
        {/* 左/上：原图 */}
        <div
          className="flex-1 overflow-hidden border-r"
          style={{
            borderColor: isDarkTheme ? '#3e3e42' : '#e5e7eb',
            backgroundColor: isDarkTheme ? '#1e1e1e' : '#ffffff',
            borderRightWidth: layout === 'horizontal' ? 1 : 0,
            borderBottomWidth: layout === 'vertical' ? 1 : 0,
          }}
        >
          <div
            className="h-7 flex items-center px-3 border-b text-[10px] font-medium uppercase tracking-wider"
            style={{
              borderColor: isDarkTheme ? '#3e3e42' : '#e5e7eb',
              color: isDarkTheme ? '#9ca3af' : '#6b7280',
              backgroundColor: isDarkTheme ? '#252526' : '#f9fafb',
            }}
          >
            <span>原图</span>
            {progress && <span className="ml-3 text-blue-500">{progress}</span>}
          </div>
          <div className="h-[calc(100%-1.75rem)]">
            <ImageUploader
              onFile={handleFile}
              previewUrl={previewUrl}
              onClear={handleClear}
              disabled={isLoading}
            />
          </div>
        </div>

        {/* 右/下：识别结果 */}
        <div className="flex-1 overflow-hidden" style={{ backgroundColor: isDarkTheme ? '#1e1e1e' : '#ffffff' }}>
          <div
            className="h-7 flex items-center px-3 border-b text-[10px] font-medium uppercase tracking-wider gap-2"
            style={{
              borderColor: isDarkTheme ? '#3e3e42' : '#e5e7eb',
              color: isDarkTheme ? '#9ca3af' : '#6b7280',
              backgroundColor: isDarkTheme ? '#252526' : '#f9fafb',
            }}
          >
            <span>识别结果</span>
            {result && (
              <span style={{ color: isDarkTheme ? '#6b7280' : '#9ca3af' }}>
                · {result.recognizer === 'accurate' ? '精确模式' : '快速模式'}
                · {result.inference_ms < 10000
                  ? `${result.inference_ms.toFixed(0)}ms`
                  : `${(result.inference_ms / 1000).toFixed(1)}s`}
              </span>
            )}
            {file && !isLoading && (
              <button
                onClick={handleRetry}
                className="ml-auto text-[10px] px-2 py-0.5 rounded bg-blue-500 text-white hover:bg-blue-600 flex items-center gap-1"
              >
                <RefreshCw size={10} /> 重新识别
              </button>
            )}
          </div>
          {result?.warnings?.map((warning, index) => (
            <div
              key={index}
              className="px-3 py-1 text-[11px] text-amber-700 bg-amber-50 border-b border-amber-100"
            >
              {warning}
            </div>
          ))}
          {result?.symbol_summary && (
            <div className="px-3 py-1 text-[11px] text-blue-700 bg-blue-50 border-b border-blue-100">
              {result.symbol_summary.notes} 音符 · {result.symbol_summary.eighth_notes} 八分 · {result.symbol_summary.sixteenth_notes} 十六分
              {result.symbol_summary.thirty_second_notes ? ` · ${result.symbol_summary.thirty_second_notes} 三十二分` : ''}
              {' · '}{result.symbol_summary.slurs} 圆滑线 · {result.symbol_summary.triplets} 三连音 · {result.symbol_summary.octave_marks} 八度标记
            </div>
          )}
          <div className="h-[calc(100%-1.75rem)]">
            <JsonPanel
              score={score}
              isLoading={isLoading}
              errorMessage={errorMsg}
              isDarkTheme={isDarkTheme}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
