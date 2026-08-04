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
  recognizer: 'accurate' | 'fast' | 'visual';
  confidence?: number | null;
  warnings?: string[];
  row_results?: unknown[];
  symbol_summary?: {
    notes: number;
    eighth_notes: number;
    sixteenth_notes: number;
    thirty_second_notes?: number;
    octave_marks: number;
    augmentation_dots: number;
    ties: number;
    slurs: number;
    triplets: number;
  };
}

const DEFAULT_BASE = (import.meta as any).env?.VITE_API_BASE || 'http://localhost:8000';

export async function recognizeImage(
  file: File,
  options: {
    conf?: number;
    useTransformer?: boolean;
    recognizer?: 'accurate' | 'fast' | 'visual';
    apiBase?: string;
  } = {},
): Promise<RecognizeResponse> {
  const base = options.apiBase || DEFAULT_BASE;
  const form = new FormData();
  form.append('file', file);

  const params = new URLSearchParams();
  if (options.conf !== undefined) params.append('conf', String(options.conf));
  if (options.useTransformer !== undefined) {
    params.append('use_transformer', String(options.useTransformer));
  }
  if (options.recognizer) params.append('recognizer', options.recognizer);

  const url = `${base}/recognize${params.toString() ? '?' + params.toString() : ''}`;
  const resp = await fetch(url, {
    method: 'POST',
    body: form,
  });

  if (!resp.ok) {
    const text = await resp.text();
    let detail = text;
    try {
      const payload = JSON.parse(text);
      if (typeof payload?.detail === 'string') detail = payload.detail;
    } catch {
      // Preserve a non-JSON response, but never expose a JSON traceback wrapper.
    }
    throw new Error(detail || `识别请求失败（HTTP ${resp.status}）`);
  }
  return resp.json();
}

export async function checkHealth(apiBase?: string): Promise<{
  status: string;
  yolo_loaded: boolean;
  transformer_loaded: boolean;
  visual_transformer_loaded?: boolean;
  accurate_vlm_available?: boolean;
}> {
  const base = apiBase || DEFAULT_BASE;
  const resp = await fetch(`${base}/health`);
  if (!resp.ok) throw new Error(`health ${resp.status}`);
  return resp.json();
}
