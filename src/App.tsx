import { useState, useCallback, useRef } from 'react';
import type { Score, ScoreLayout } from './types';
import { examples } from './data/examples';
import { defaultScore } from './data/defaults';
import { downloadPNG } from './engine';
import Editor from './components/Editor';
import Preview from './components/Preview';
import Toolbar from './components/Toolbar';

export default function App() {
  const [jsonValue, setJsonValue] = useState(() => JSON.stringify(defaultScore, null, 2));
  const [score, setScore] = useState<Score | null>(defaultScore);
  const [zoom, setZoom] = useState(1);
  const [isValid, setIsValid] = useState(true);
  const canvasLayoutRef = useRef<ScoreLayout | null>(null);

  const handleJsonChange = useCallback((value: string, parsedScore: Score | null) => {
    setJsonValue(value);
    setScore(parsedScore);
    setIsValid(parsedScore !== null);
  }, []);

  const handleExampleSelect = useCallback((key: string) => {
    const example = examples[key];
    if (example) {
      const json = JSON.stringify(example.data, null, 2);
      setJsonValue(json);
      setScore(example.data);
      setIsValid(true);
    }
  }, []);

  const handleZoomIn = useCallback(() => {
    setZoom(z => Math.min(z + 0.25, 3));
  }, []);

  const handleZoomOut = useCallback(() => {
    setZoom(z => Math.max(z - 0.25, 0.25));
  }, []);

  const handleExport = useCallback(() => {
    const canvas = document.querySelector('canvas');
    if (canvas && canvasLayoutRef.current) {
      const title = score?.title || 'jianpu-score';
      downloadPNG(canvas, `${title}.png`, 2);
    }
  }, [score]);

  const handleLayoutChange = useCallback((layout: ScoreLayout) => {
    canvasLayoutRef.current = layout;
  }, []);

  return (
    <div className="h-screen w-screen flex flex-col bg-dark-900 overflow-hidden">
      <Toolbar
        onExampleSelect={handleExampleSelect}
        onExport={handleExport}
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        zoom={zoom}
        isValid={isValid}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* 左侧编辑器 */}
        <div className="w-[40%] min-w-[300px] border-r border-gray-800 bg-dark-800 relative">
          <div className="absolute top-0 left-0 right-0 h-7 bg-dark-900 flex items-center px-3 z-10 border-b border-gray-800">
            <span className="text-[10px] text-gray-500 font-medium uppercase tracking-wider">JSON Editor</span>
            <div className="ml-auto flex items-center gap-1.5">
              <div className={`w-1.5 h-1.5 rounded-full ${isValid ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-[10px] text-gray-500">{isValid ? '有效' : '语法错误'}</span>
            </div>
          </div>
          <div className="pt-7 h-full">
            <Editor value={jsonValue} onChange={handleJsonChange} />
          </div>
        </div>

        {/* 右侧预览 */}
        <div className="flex-1 relative">
          <div className="absolute top-0 left-0 right-0 h-7 bg-dark-900 flex items-center px-3 z-10 border-b border-gray-800">
            <span className="text-[10px] text-gray-500 font-medium uppercase tracking-wider">Preview</span>
            <span className="ml-auto text-[10px] text-gray-600">{Math.round(zoom * 100)}%</span>
          </div>
          <div className="pt-7 h-full">
            <Preview score={score} zoom={zoom} onLayoutChange={handleLayoutChange} />
          </div>
        </div>
      </div>
    </div>
  );
}
