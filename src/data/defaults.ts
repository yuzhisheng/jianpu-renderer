import type { Score } from '../types';

export const defaultScore: Score = {
  title: '未命名乐谱',
  key: 'C',
  timeSignature: { numerator: 4, denominator: 4 },
  tempo: 80,
  measures: [
    {
      notes: [
        { pitch: 1, duration: 1 },
        { pitch: 2, duration: 1 },
        { pitch: 3, duration: 1 },
        { pitch: 5, duration: 1 },
      ],
    },
    {
      notes: [
        { pitch: 6, duration: 1 },
        { pitch: 5, duration: 1 },
        { pitch: 3, duration: 1 },
        { pitch: 2, duration: 1 },
      ],
      barline: 'end',
    },
  ],
};
