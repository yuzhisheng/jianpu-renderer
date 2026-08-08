import { useState, useRef, useCallback, DragEvent, ChangeEvent } from 'react';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

interface ImageUploaderProps {
  onFile: (file: File) => void;
  previewUrl: string | null;
  previewType?: 'image' | 'pdf';
  onClear: () => void;
  disabled?: boolean;
}

function isSupportedFile(file: File): boolean {
  return file.type.startsWith('image/')
    || file.type === 'application/pdf'
    || file.name.toLowerCase().endsWith('.pdf');
}

export default function ImageUploader({ onFile, previewUrl, previewType = 'image', onClear, disabled }: ImageUploaderProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    if (!disabled) setIsDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file && isSupportedFile(file)) {
      onFile(file);
    }
  }, [disabled, onFile]);

  const handleSelect = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFile(file);
    }
    e.target.value = '';
  }, [onFile]);

  if (previewUrl) {
    return (
      <div className="relative h-full flex items-center justify-center p-4 overflow-auto">
        {previewType === 'pdf' ? (
          <iframe
            src={previewUrl}
            title="原始 PDF"
            className="h-full w-full rounded border border-gray-200 bg-white"
          />
        ) : (
          <img
            src={previewUrl}
            alt="原图"
            className="max-w-full max-h-full object-contain rounded border border-gray-200"
            style={{ background: '#fafafa' }}
          />
        )}
        <button
          onClick={onClear}
          disabled={disabled}
          className="absolute top-2 right-2 p-1.5 rounded-full bg-black/60 text-white hover:bg-black/80 disabled:opacity-50"
          title="清除"
        >
          <X size={14} />
        </button>
      </div>
    );
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={`
        h-full m-4 border-2 border-dashed rounded-lg flex flex-col items-center justify-center
        cursor-pointer transition-all duration-200
        ${isDragging
          ? 'border-blue-500 bg-blue-50 scale-[1.01]'
          : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        }
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg,image/webp,application/pdf,.pdf"
        onChange={handleSelect}
        className="hidden"
        disabled={disabled}
      />
      <div className={`p-4 rounded-full mb-3 ${isDragging ? 'bg-blue-100' : 'bg-gray-100'}`}>
        {isDragging ? <ImageIcon size={32} className="text-blue-500" /> : <Upload size={32} className="text-gray-400" />}
      </div>
      <p className="text-sm font-medium text-gray-700">
        {isDragging ? '松开鼠标上传' : '拖拽简谱图片或 PDF，或点击上传'}
      </p>
      <p className="text-xs text-gray-400 mt-1">支持 PNG / JPG / WEBP / PDF，图片最大 10MB，PDF 最大 50MB</p>
    </div>
  );
}
