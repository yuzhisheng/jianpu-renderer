// 后端识别 API 封装
export interface Detection {
  class_id: number;
  class_name: string;
  cx: number;
  cy: number;
  w: number;
  h: number;
  conf: number;
}

export interface RecognizeResponse {
  score: any;          // Score JSON
  detections: Detection[];
  num_detections: number;
  inference_ms: number;
  src_tokens: string[];
  tgt_tokens: string[];
}

const DEFAULT_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:8000';

export async function recognizeImage(
  file: File,
  options: { conf?: number; useTransformer?: boolean; apiBase?: string } = {},
): Promise<RecognizeResponse> {
  const base = options.apiBase || DEFAULT_BASE;
  const form = new FormData();
  form.append('file', file);

  const params = new URLSearchParams();
  if (options.conf !== undefined) params.append('conf', String(options.conf));
  if (options.useTransformer !== undefined) {
    params.append('use_transformer', String(options.useTransformer));
  }

  const url = `${base}/recognize${params.toString() ? '?' + params.toString() : ''}`;
  const resp = await fetch(url, {
    method: 'POST',
    body: form,
  });

  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`后端错误 ${resp.status}: ${text}`);
  }
  return resp.json();
}

export async function checkHealth(apiBase?: string): Promise<{
  status: string;
  yolo_loaded: boolean;
  transformer_loaded: boolean;
}> {
  const base = apiBase || DEFAULT_BASE;
  const resp = await fetch(`${base}/health`);
  if (!resp.ok) throw new Error(`health ${resp.status}`);
  return resp.json();
}
