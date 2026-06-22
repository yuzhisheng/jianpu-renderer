import { Download, ZoomIn, ZoomOut, FileMusic } from 'lucide-react';
import { examples } from '../data/examples';

interface ToolbarProps {
  onExampleSelect: (key: string) => void;
  onExport: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  zoom: number;
  isValid: boolean;
}

export default function Toolbar({ onExampleSelect, onExport, onZoomIn, onZoomOut, zoom, isValid }: ToolbarProps) {
  return (
    <div className="h-12 bg-dark-900 flex items-center justify-between px-4 border-b border-gray-800 shadow-lg">
      {/* 左侧标题 */}
      <div className="flex items-center gap-3">
        <FileMusic className="w-5 h-5 text-primary-400" />
        <h1 className="text-sm font-semibold text-gray-200 tracking-wide">简谱渲染器</h1>
        <span className="text-[10px] text-gray-500 bg-gray-800 px-2 py-0.5 rounded-full">v1.0</span>
      </div>

      {/* 右侧操作 */}
      <div className="flex items-center gap-2">
        {/* 示例选择 */}
        <select
          className="bg-gray-800 text-gray-300 text-xs rounded-md px-3 py-1.5 border border-gray-700 focus:border-primary-500 focus:outline-none cursor-pointer hover:border-gray-600 transition-colors"
          onChange={(e) => onExampleSelect(e.target.value)}
          defaultValue=""
        >
          <option value="" disabled>选择示例乐谱</option>
          {Object.entries(examples).map(([key, { name }]) => (
            <option key={key} value={key}>{name}</option>
          ))}
        </select>

        {/* 缩放控制 */}
        <div className="flex items-center bg-gray-800 rounded-md border border-gray-700 overflow-hidden">
          <button
            className="px-2 py-1.5 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
            onClick={onZoomOut}
            title="缩小"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="text-xs text-gray-400 min-w-[42px] text-center">{Math.round(zoom * 100)}%</span>
          <button
            className="px-2 py-1.5 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
            onClick={onZoomIn}
            title="放大"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* 导出按钮 */}
        <button
          className={`flex items-center gap-1.5 text-xs rounded-md px-3 py-1.5 font-medium transition-all ${
            isValid
              ? 'bg-primary-500 text-white hover:bg-primary-600 shadow-md shadow-primary-500/25'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          }`}
          onClick={onExport}
          disabled={!isValid}
          title="导出 PNG 图片"
        >
          <Download className="w-3.5 h-3.5" />
          导出 PNG
        </button>
      </div>
    </div>
  );
}
