// 后端识别 API 封装
import type { Score } from '../types';

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
  recognition_id?: string | null;
  score: Score;        // Score JSON
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
    lyric_syllables?: number;
  };
}

export interface RecognitionHistoryItem {
  id: string;
  created_at: string;
  updated_at: string;
  status: 'running' | 'succeeded' | 'failed' | string;
  original_filename: string;
  image_width?: number | null;
  image_height?: number | null;
  recognizer?: 'accurate' | 'fast' | 'visual' | string | null;
  confidence?: number | null;
  inference_ms?: number | null;
  error?: string | null;
  title?: string | null;
  notes?: number | null;
  lyric_syllables?: number | null;
}

export interface RecognitionHistoryDetail extends RecognitionHistoryItem {
  image_url?: string | null;
  response?: RecognizeResponse | null;
}

const DEFAULT_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

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

export async function listRecognitionHistory(
  options: { limit?: number; offset?: number; apiBase?: string } = {},
): Promise<RecognitionHistoryItem[]> {
  const base = options.apiBase || DEFAULT_BASE;
  const params = new URLSearchParams();
  if (options.limit !== undefined) params.set('limit', String(options.limit));
  if (options.offset !== undefined) params.set('offset', String(options.offset));
  const url = `${base}/recognition-history${params.toString() ? '?' + params.toString() : ''}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`历史记录读取失败（HTTP ${resp.status}）`);
  const payload = await resp.json() as { items?: RecognitionHistoryItem[] };
  return Array.isArray(payload.items) ? payload.items : [];
}

export async function getRecognitionHistory(
  id: string,
  apiBase?: string,
): Promise<RecognitionHistoryDetail> {
  const base = apiBase || DEFAULT_BASE;
  const resp = await fetch(`${base}/recognition-history/${encodeURIComponent(id)}`);
  if (!resp.ok) throw new Error(`识别记录读取失败（HTTP ${resp.status}）`);
  return resp.json();
}

export function recognitionHistoryImageUrl(id: string, apiBase?: string): string {
  const base = apiBase || DEFAULT_BASE;
  return `${base}/recognition-history/${encodeURIComponent(id)}/image`;
}
