import type { LayoutConfig } from './layout';
import { DEFAULT_CONFIG } from './layout';
import { jsPDF } from 'jspdf';

/** 导出 Canvas 内容为 PNG */
export function exportToPNG(
  canvas: HTMLCanvasElement,
  scale: number = 2,
): Promise<Blob> {
  const offscreen = document.createElement('canvas');
  offscreen.width = canvas.width * scale / (window.devicePixelRatio || 1);
  offscreen.height = canvas.height * scale / (window.devicePixelRatio || 1);
  const offCtx = offscreen.getContext('2d')!;
  offCtx.scale(scale, scale);
  offCtx.drawImage(canvas, 0, 0, canvas.width / (window.devicePixelRatio || 1), canvas.height / (window.devicePixelRatio || 1));

  return new Promise((resolve, reject) => {
    offscreen.toBlob(blob => {
      if (blob) resolve(blob);
      else reject(new Error('Failed to export PNG'));
    }, 'image/png');
  });
}

/** 下载 PNG 文件 */
export async function downloadPNG(
  canvas: HTMLCanvasElement,
  filename: string = 'jianpu-score.png',
  scale: number = 2,
) {
  const blob = await exportToPNG(canvas, scale);
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** 下载为 PDF；长谱面会按 A4 高度自动分页，避免被缩成一张不可读的长图。 */
export async function downloadPDF(
  canvas: HTMLCanvasElement,
  filename: string = 'jianpu-score.pdf',
) {
  const dpr = window.devicePixelRatio || 1;
  const logicalWidth = canvas.width / dpr;
  const logicalHeight = canvas.height / dpr;
  const pdfWidth = 595.28;
  const pdfHeight = 841.89;
  const margin = 18;
  const contentWidth = pdfWidth - margin * 2;
  const contentHeight = pdfHeight - margin * 2;
  const scale = contentWidth / logicalWidth;
  const logicalSliceHeight = contentHeight / scale;
  const sourceSliceHeight = Math.max(1, Math.round(logicalSliceHeight * dpr));
  const totalSlices = Math.max(1, Math.ceil(canvas.height / sourceSliceHeight));
  const document = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });

  for (let index = 0; index < totalSlices; index += 1) {
    if (index > 0) document.addPage();
    const sourceTop = index * sourceSliceHeight;
    const sliceHeight = Math.min(sourceSliceHeight, canvas.height - sourceTop);
    // Use a browser canvas for a cropped page; jsPDF itself does not expose a
    // portable image-cropping API across its supported browser builds.
    const pageCanvas = window.document.createElement('canvas');
    pageCanvas.width = canvas.width;
    pageCanvas.height = sliceHeight;
    const context = pageCanvas.getContext('2d');
    if (!context) throw new Error('无法创建 PDF 页面画布');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, pageCanvas.width, pageCanvas.height);
    context.drawImage(canvas, 0, -sourceTop);
    const imageData = pageCanvas.toDataURL('image/png');
    document.addImage(
      imageData, 'PNG', margin, margin, contentWidth,
      (sliceHeight / dpr) * scale,
    );
  }
  document.save(filename.endsWith('.pdf') ? filename : `${filename}.pdf`);
}

/** 计算渲染所需尺寸 */
export function getCanvasSize(
  layoutHeight: number,
  config: LayoutConfig = DEFAULT_CONFIG,
) {
  const dpr = window.devicePixelRatio || 1;
  return {
    width: config.canvasWidth * dpr,
    height: layoutHeight * dpr,
    cssWidth: config.canvasWidth,
    cssHeight: layoutHeight,
    dpr,
  };
}

/** 设置 Canvas DPI 适配 */
export function setupCanvasDPI(
  canvas: HTMLCanvasElement,
  cssWidth: number,
  cssHeight: number,
): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);
  return ctx;
}
