import { useRef, useEffect, useCallback, useState } from 'react';
import type { Score, ScoreLayout } from '../types';
import { calculateLayout, DEFAULT_CONFIG, render, DEFAULT_THEME, setupCanvasDPI } from '../engine';
import type { LayoutConfig } from '../engine';


interface PreviewProps {
  score: Score | null;
  zoom: number;
  onLayoutChange?: (layout: ScoreLayout) => void;
}

export default function Preview({ score, zoom, onLayoutChange }: PreviewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasKey, setCanvasKey] = useState(0);

  const renderScore = useCallback(() => {
    if (!canvasRef.current || !score) return;

    const config: LayoutConfig = { ...DEFAULT_CONFIG };
    const layout = calculateLayout(score, config);

    const ctx = setupCanvasDPI(canvasRef.current, layout.width, layout.height);
    render(ctx, layout, score, config, DEFAULT_THEME);

    onLayoutChange?.(layout);
  }, [score, onLayoutChange]);

  useEffect(() => {
    if (score) {
      setCanvasKey(k => k + 1);
    }
  }, [score]);

  useEffect(() => {
    renderScore();
  }, [renderScore, canvasKey]);

  return (
    <div className="h-full w-full overflow-auto flex items-start justify-center p-6" style={{backgroundColor:'#1e1e1e'}}>
      {!score ? (
        <div className="flex items-center justify-center h-full w-full">
          <div className="text-center text-gray-500">
            <div className="text-5xl mb-4">♫</div>
            <p className="text-lg">在左侧编辑 JSON 即可预览简谱</p>
            <p className="text-sm mt-2 text-gray-600">或选择一个示例乐谱开始</p>
          </div>
        </div>
      ) : (
        <div style={{ transform: `scale(${zoom})`, transformOrigin: 'top center' }} className="transition-transform duration-200">
          <canvas key={canvasKey} ref={canvasRef} className="shadow-lg rounded-sm" />
        </div>
      )}
    </div>
  );
}
