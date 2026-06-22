import { useRef, useCallback } from 'react';
import Editor, { OnMount } from '@monaco-editor/react';
import type { Score } from '../types';
import schema from '../schema/jianpu.schema.json';

interface EditorProps {
  value: string;
  onChange: (value: string, score: Score | null) => void;
}

export default function JsonEditor({ value, onChange }: EditorProps) {
  const editorRef = useRef<Parameters<OnMount>[0] | null>(null);

  const handleMount: OnMount = useCallback((editor, monaco) => {
    editorRef.current = editor;
    monaco.languages.json.jsonDefaults.setDiagnosticsOptions({
      validate: true,
      schemas: [{
        uri: 'jianpu://schema',
        fileMatch: ['*'],
        schema,
      }],
    });
  }, []);

  const handleChange = useCallback((val: string | undefined) => {
    const str = val || '';
    let score: Score | null = null;
    try {
      const parsed = JSON.parse(str);
      if (parsed && typeof parsed === 'object' && parsed.measures) {
        score = parsed as Score;
      }
    } catch {
      // JSON 解析失败，score 保持 null
    }
    onChange(str, score);
  }, [onChange]);

  return (
    <div className="h-full w-full overflow-hidden">
      <Editor
        height="100%"
        defaultLanguage="json"
        value={value}
        onChange={handleChange}
        onMount={handleMount}
        theme="vs-dark"
        options={{
          minimap: { enabled: false },
          fontSize: 13,
          lineNumbers: 'on',
          scrollBeyondLastLine: false,
          wordWrap: 'on',
          tabSize: 2,
          automaticLayout: true,
          padding: { top: 12 },
          renderWhitespace: 'none',
          bracketPairColorization: { enabled: true },
        }}
      />
    </div>
  );
}
