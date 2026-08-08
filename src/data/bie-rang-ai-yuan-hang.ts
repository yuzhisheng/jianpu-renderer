import type { Dash, Note, Score } from '../types';

type NoteExtras = Omit<Partial<Note>, 'pitch' | 'duration'>;

const n = (pitch: Note['pitch'], duration: number, extras: NoteExtras = {}): Note => ({
  pitch,
  duration,
  ...extras,
});

const d = (duration: number = 1): Dash => ({ type: 'dash', duration });

/**
 * 《别让爱远航》扫描谱复刻。
 *
 * beamLevel 是扫描谱的显式减时线层级；duration 仍然用于拍点和小节布局。
 * lineBreakBefore 保留原图谱行，方便把渲染结果和原始图片逐行比对。
 */
export const bieRangAiYuanHang: Score = {
  title: '别让爱远航',
  subtitle: '每分钟70拍　思念、倾情地',
  lyricist: '田素安',
  composer: '周明仁',
  key: 'G',
  timeSignature: { numerator: 4, denominator: 4 },
  tempo: 70,
  measures: [
    // 第一谱行：前奏
    {
      lineBreakBefore: true,
      notes: [
        n(1, .5, { beamLevel: 1, parenthesisLeft: true }), n(5, .5, { beamLevel: 1, octave: -1 }),
        n(3, .5, { beamLevel: 1 }), n(5, .5, { beamLevel: 1, octave: -1 }),
        n(1, .5, { beamLevel: 1 }), n(5, .5, { beamLevel: 1, octave: -1 }),
        n(3, .5, { beamLevel: 1, octave: -1 }), n(5, .5, { beamLevel: 1, octave: -1 }),
      ],
    },
    {
      notes: [
        n(2, .5, { beamLevel: 1 }), n(5, .5, { beamLevel: 1, octave: -1 }),
        n(4, .5, { beamLevel: 1 }), n(5, .5, { beamLevel: 1, octave: -1 }),
        n(7, .5, { beamLevel: 1, octave: -1 }), n(5, .5, { beamLevel: 1, octave: -1 }),
        n(2, .5, { beamLevel: 1, octave: -1 }), n(5, .5, { beamLevel: 1, octave: -1 }),
      ],
    },
    {
      notes: [n(5, 1), d(), d(), n(2, .5, { dot: 1, octave: 1, beamLevel: 1 }), n(1, .25, { octave: 1, beamLevel: 2 })],
    },

    // 第二谱行
    {
      lineBreakBefore: true,
      notes: [n(1, 1, { octave: 1 }), d(), d(), n(3, .5, { beamLevel: 1 }), n(4, .5, { beamLevel: 1 })],
    },
    {
      notes: [
        n(5, .5, { beamLevel: 1 }), n(6, .5, { beamLevel: 1 }),
        n(1, .5, { beamLevel: 1, slurId: 'b2-1' }), n(1, .5, { beamLevel: 1, slurId: 'b2-1' }),
        n(6, .5, { beamLevel: 1, octave: -1 }), n(1, .5, { beamLevel: 1 }),
        n(4, .5, { dot: 1, beamLevel: 1 }), n(3, .25, { beamLevel: 2 }),
      ],
    },
    {
      notes: [n(2, 1), d(), d(), n(3, .5, { beamLevel: 1 }), n(4, .5, { beamLevel: 1 })],
    },

    // 第三谱行
    {
      lineBreakBefore: true,
      notes: [n(5, 1), d(), d(), n(2, .5, { dot: 1, octave: 1, beamLevel: 1 }), n(1, .25, { octave: 1, beamLevel: 2 })],
    },
    {
      notes: [n(1, 1), d(), d(), n(7, .5, { beamLevel: 1 }), n(6, .5, { beamLevel: 1 })],
    },
    {
      notes: [
        n(5, .5, { beamLevel: 1 }), n(6, .5, { beamLevel: 1 }),
        n(1, .5, { beamLevel: 1, slurId: 'c3-1' }), n(1, .5, { beamLevel: 1, slurId: 'c3-1' }),
        n(2, .5, { beamLevel: 1 }), n(3, .5, { beamLevel: 1 }),
        n(4, .25, { beamLevel: 1, slurId: 'c3-4' }),
        n(4, .5, { beamLevel: 1, slurId: 'c3-4' }),
        n(3, .125, { beamLevel: 2 }), n(2, .125, { beamLevel: 2 }),
      ],
    },

    // 第一段进入歌词
    {
      lineBreakBefore: true,
      notes: [n(1, 1), d(), d(), n(1, 1, { parenthesisRight: true, lyric: '月', dynamic: 'mp' })],
    },
    {
      notes: [n(5, 1, { lyric: '亮' }), d(), d(), d()],
    },
    {
      notes: [
        n(0, .5, { beamLevel: 1, lyric: '升' }), n(6, .5, { beamLevel: 1, lyric: '起' }),
        n(5, .5, { beamLevel: 1, lyric: '来' }), n(4, .5, { dot: 1, beamLevel: 1, lyric: '的' }),
        n(1, .25, { beamLevel: 2, lyric: '时' }), n(3, .5, { beamLevel: 1, lyric: '候，', slurId: 'd3-345' }),
        n(4, .5, { beamLevel: 1 }), n(5, .5, { beamLevel: 1, slurId: 'd3-345' }),
      ],
    },
    {
      notes: [n(5, 1), d(), d(), d()],
    },

    // 第二段
    {
      lineBreakBefore: true,
      notes: [
        n(6, .25, { beamLevel: 2, lyric: '轻' }), n(6, .25, { beamLevel: 2, lyric: '轻' }),
        n(1, .25, { beamLevel: 2, lyric: '地', slurId: 'e1-11' }), n(1, .25, { beamLevel: 2, slurId: 'e1-11' }),
        n(4, .25, { beamLevel: 2, lyric: '把' }), n(4, .25, { beamLevel: 2, lyric: '六' }),
        n(3, .25, { beamLevel: 2, lyric: '弦', slurId: 'e1-33' }), n(3, .25, { beamLevel: 2, lyric: '琴', slurId: 'e1-33' }),
        n(2, 1, { beamLevel: 1, lyric: '揉', slurId: 'e1-21' }), n(1, 1, { beamLevel: 1, slurId: 'e1-21' }),
      ],
    },
    {
      notes: [
        n(2, 1, { lyric: '啊，' }), d(), d(),
        n(0, .25, { beamLevel: 2, lyric: '让' }), n(3, .25, { beamLevel: 2 }), n(4, .5, { beamLevel: 1 }),
      ],
    },
    {
      notes: [
        n(5, .5, { dot: 1, beamLevel: 1, lyric: '我' }), n(5, .25, { beamLevel: 2, lyric: '的' }),
        n(0, .5, { beamLevel: 1, lyric: '思' }), n(6, .5, { beamLevel: 1, lyric: '念' }),
        n(4, .5, { beamLevel: 1 }), n(0, .5, { beamLevel: 1 }),
        n(3, .5, { beamLevel: 1, lyric: '推', slurId: 'e3-034' }), n(4, .5, { beamLevel: 1, slurId: 'e3-034' }),
      ],
    },

    // 第三段
    {
      lineBreakBefore: true,
      notes: [
        n(5, 1, { lyric: '开' }), n(5, .5, { beamLevel: 1, lyric: '你' }),
        n(5, .5, { beamLevel: 1, lyric: '的' }), n(2, .5, { dot: 1, beamLevel: 1, lyric: '小' }),
        n(3, .25, { beamLevel: 2, lyric: '窗，' }), d(),
      ],
    },
    {
      notes: [
        n(2, .25, { beamLevel: 2, lyric: '让', slurId: 'f2-22' }),
        n(2, .25, { beamLevel: 2 }), n(2, .25, { beamLevel: 2, slurId: 'f2-22' }),
        n(2, .25, { beamLevel: 2 }), n(2, .25, { beamLevel: 2 }),
        n(3, .5, { beamLevel: 1, lyric: '你' }), n(4, .5, { beamLevel: 1, lyric: '的' }),
        n(3, .5, { beamLevel: 1, lyric: '床' }), n(2, 1, { beamLevel: 1, lyric: '前' }), n(1, .25, { beamLevel: 2, lyric: '洒' }),
      ],
    },
    {
      notes: [
        n(2, .25, { beamLevel: 1, lyric: '月', octave: 1 }), n(3, .25, { beamLevel: 1 }), d(), d(),
        n(0, .5, { beamLevel: 1, lyric: '光。' }), n(3, .5, { beamLevel: 1, lyric: '让' }), n(4, .5, { beamLevel: 1 }),
      ],
    },

    // 重复一次副歌式句子
    {
      lineBreakBefore: true,
      notes: [
        n(5, .5, { dot: 1, beamLevel: 1, lyric: '我' }), n(5, .25, { beamLevel: 2, lyric: '的' }),
        n(0, .5, { beamLevel: 1, lyric: '思' }), n(1, .5, { beamLevel: 1, lyric: '念' }),
        n(4, .5), n(0, .5, { beamLevel: 1 }), n(3, .5, { beamLevel: 1, lyric: '推', slurId: 'g1-034' }), n(4, .5, { beamLevel: 1, slurId: 'g1-034' }),
      ],
    },
    {
      notes: [n(5, 1, { lyric: '开' }), n(5, .5, { beamLevel: 1, lyric: '你' }), n(5, .5, { beamLevel: 1, lyric: '的' }), n(4, .5, { dot: 1, beamLevel: 1, lyric: '小' }), n(3, .25, { beamLevel: 2, lyric: '窗，' }), d()],
    },
    {
      notes: [
        n(5, .25, { beamLevel: 2, lyric: '让', slurId: 'g3-55' }), n(5, .25, { beamLevel: 2 }),
        n(5, .25, { beamLevel: 2, lyric: '你', slurId: 'g3-55' }), n(5, .25, { beamLevel: 2, lyric: '的' }),
        n(6, .5, { beamLevel: 1, lyric: '床' }), n(1, .5, { beamLevel: 1, lyric: '前' }),
        n(4, .5, { beamLevel: 1, lyric: '洒' }), n(3, .5, { beamLevel: 1, lyric: '满' }),
        n(2, .5, { beamLevel: 1, lyric: '月', slurId: 'g3-23' }), n(3, .5, { beamLevel: 1, slurId: 'g3-23' }),
      ],
    },

    // 过门：原图单独括起，下一行重新起句
    {
      lineBreakBefore: true,
      notes: [n(0, .25, { beamLevel: 2, parenthesisLeft: true }), n(2, .25, { beamLevel: 2 }), n(3, .5, { beamLevel: 1 })],
      barline: 'single',
    },
    {
      notes: [
        n(4, .5, { beamLevel: 1 }), n(5, .5, { beamLevel: 1 }),
        n(6, .5, { beamLevel: 1, slurId: 'pickup-66' }), n(6, .5, { beamLevel: 1, slurId: 'pickup-66' }),
        n(7, .5, { beamLevel: 1 }), n(1, .5, { beamLevel: 1, parenthesisRight: true }),
      ],
      barline: 'none',
    },

    // 结尾第一行
    {
      lineBreakBefore: true,
      notes: [n(1, 1, { techniques: [{ type: 'yinyin', graceNotes: [3] }], lyric: '光。' }), d(), d(), d()],
    },
    {
      notes: [n(0, 1), n(0, 1), n(0, 1), n(0, 1)],
    },
    {
      notes: [
        n(6, .25, { beamLevel: 1, lyric: '请' }), n(6, .25, { beamLevel: 1, lyric: '你' }),
        n(6, .25, { beamLevel: 1, lyric: '别', slurId: 'h3-66' }), n(6, .25, { beamLevel: 1, slurId: 'h3-66' }),
        n(4, .5, { beamLevel: 1, lyric: '疑' }), n(1, .5, { beamLevel: 1, lyric: '是' }),
        n(4, .5, { beamLevel: 1, lyric: '地' }), n(1, .5, { beamLevel: 1, lyric: '上', slurId: 'h3-176' }),
        n(7, .5, { beamLevel: 1, slurId: 'h3-176' }), n(6, .5, { beamLevel: 1 }),
      ],
    },

    // 最后一谱行
    {
      lineBreakBefore: true,
      notes: [n(6, 1, { lyric: '霜，' }), d(), d(), d()],
    },
    {
      notes: [
        n(5, .5, { beamLevel: 1, lyric: '那', slurId: 'i2-55' }), n(5, .5, { beamLevel: 1, lyric: '是', slurId: 'i2-55' }),
        n(5, .5, { beamLevel: 1, lyric: '我' }), n(6, .5, { beamLevel: 1, lyric: '撒' }),
        n(5, .5, { beamLevel: 1, lyric: '下' }), n(1, .5, { beamLevel: 1, lyric: '的' }),
        n(4, .5, { beamLevel: 1, lyric: '情' }), n(4, .25, { beamLevel: 2 }), n(3, .25, { beamLevel: 2, lyric: '网，' }),
      ],
    },
    {
      notes: [n(3, 1), d(), d(), d()],
    },
    {
      barline: 'end',
      notes: [
        n(6, .25, { beamLevel: 1, lyric: '请' }), n(6, .25, { beamLevel: 1, lyric: '你' }),
        n(6, .25, { beamLevel: 1, lyric: '别', slurId: 'i4-66' }), n(6, .25, { beamLevel: 1, slurId: 'i4-66' }),
        n(4, .5, { beamLevel: 1, lyric: '举' }), n(1, .5, { beamLevel: 1, lyric: '头' }),
        n(4, .5, { beamLevel: 1, lyric: '望' }), n(1, .5, { beamLevel: 1, lyric: '明', slurId: 'i4-176' }),
        n(7, .5, { beamLevel: 1, slurId: 'i4-176' }), n(6, .5, { beamLevel: 1, lyric: '月' }),
      ],
    },
  ],
};
