// mend: one command-line tool that applies AI-written edits to a tree of text
// files. It reads a patch in one of four shapes (unified diff, SEARCH/REPLACE
// blocks, V4A, JSON edits), finds each edit's old lines at the first of three
// matching tiers, and writes every change or none at all, printing what it
// changed or why it failed. Node 22, standard library only; run as
// `node main.ts [options] [PATCH]`.
//
// In a trial the grader runs `node /bot/main.ts` with MAKEFUN_ENV_URL set, and
// then this file hands over to runner.ts (the last lines below): keep them.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

const FORMATS = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

/** A parse fault: the patch cannot be read in its format, at line `no`. */
class ParseFault {
  readonly no: number;
  constructor(no: number) {
    this.no = no;
  }
}

/** One edit failed, with the code and path the report line names. */
class EditFault {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const isBlank = (s: string): boolean => s.trim() === '';

/** A patch's lines: split at `\n`, one trailing `\r` off each, no empty piece
 * after a final `\n`. */
function patchLines(text: string): string[] {
  if (text === '') return [];
  const lines = text.split('\n');
  if (text.endsWith('\n')) lines.pop();
  return lines.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}

/** A path is safe when it is not empty, does not start with `/`, holds no
 * `\`, and none of its `/`-separated segments is empty, `.` or `..`. */
function safePath(p: string): boolean {
  return p !== '' && !p.startsWith('/') && !p.includes('\\') && p.split('/').every((s) => s !== '' && s !== '.' && s !== '..');
}

/** Byte order of two paths, as UTF-8. */
function byteOrder(a: string, b: string): number {
  return Buffer.compare(Buffer.from(a, 'utf8'), Buffer.from(b, 'utf8'));
}

/** The leading spaces and tabs of a line. */
function leadWs(s: string): string {
  const m = /^[ \t]*/.exec(s);
  return m === null ? '' : m[0];
}

/** A line compared at one matching tier: exact, trailing space/tab blind, or
 * leading and trailing space/tab blind. */
function tierLine(s: string, tier: number): string {
  if (tier === 0) return s;
  if (tier === 1) return s.replace(/[ \t]+$/, '');
  return s.replace(/^[ \t]+/, '').replace(/[ \t]+$/, '');
}

/** One file's lines, its line break, and whether it ends with one. */
interface FileState {
  lines: string[];
  br: string;
  end: boolean;
}

function readState(text: string): FileState {
  if (text === '') return { lines: [], br: '\n', end: false };
  const nl = text.indexOf('\n');
  const br = nl > 0 && text[nl - 1] === '\r' ? '\r\n' : '\n';
  const end = text.endsWith(br);
  const body = end ? text.slice(0, -br.length) : text;
  return { lines: body.split(br), br, end };
}

function writeState(s: FileState): string {
  if (s.lines.length === 0) return '';
  return s.lines.join(s.br) + (s.end ? s.br : '');
}

/** The tree as the edits leave it: paths changed in `now` (null: deleted),
 * the rest read from disk on demand. Moves are remembered for the report. */
class Tree {
  private readonly orig = new Map<string, string | null>();
  private now = new Map<string, string | null>();
  private moves: { old: string; next: string }[] = [];

  original(p: string): string | null {
    if (!this.orig.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.orig.set(p, v);
    }
    return this.orig.get(p)!;
  }

  current(p: string): string | null {
    return this.now.has(p) ? this.now.get(p)! : this.original(p);
  }

  set(p: string, v: string | null): void {
    this.now.set(p, v);
  }

  noteMove(old: string, next: string): void {
    this.moves.push({ old, next });
  }

  snapshot(): { now: Map<string, string | null>; moves: { old: string; next: string }[] } {
    return { now: new Map(this.now), moves: [...this.moves] };
  }

  restore(s: { now: Map<string, string | null>; moves: { old: string; next: string }[] }): void {
    this.now = s.now;
    this.moves = s.moves;
  }

  changedPaths(): string[] {
    return [...this.now.keys()];
  }

  moveList(): { old: string; next: string }[] {
    return this.moves;
  }
}

type Edit = (tree: Tree) => void;

/** Positions where `old` occurs in `lines` at one tier (overlaps allowed). */
function occurrences(lines: string[], old: string[], tier: number): number[] {
  const found: number[] = [];
  for (let i = 0; i + old.length <= lines.length; i += 1) {
    let ok = true;
    for (let j = 0; j < old.length; j += 1) {
      if (tierLine(lines[i + j]!, tier) !== tierLine(old[j]!, tier)) {
        ok = false;
        break;
      }
    }
    if (ok) found.push(i);
  }
  return found;
}

/** The first tier at which `old` occurs in `lines`, with its positions. */
function matchBlock(lines: string[], old: string[], strict: boolean): { tier: number; at: number[] } | null {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const t of tiers) {
    const at = occurrences(lines, old, t);
    if (at.length > 0) return { tier: t, at };
  }
  return null;
}

/** The occurrence nearest to `target`; of two equally near, the earlier. */
function nearest(at: number[], target: number): number {
  let best = at[0]!;
  let gap = Math.abs(best - target);
  for (const p of at) {
    const d = Math.abs(p - target);
    if (d < gap) {
      best = p;
      gap = d;
    }
  }
  return best;
}

interface Piece {
  text: string;
  kind: 'ctx' | 'old' | 'new';
  backslash?: boolean;
}

/** The new lines that replace the matched lines: a context line is written as
 * the file's own matched line, never the patch's; at tier 2 every other line
 * is re-indented from the old lines' indent to the file's. */
function replacement(lines: string[], at: number, oldPieces: Piece[], newPieces: Piece[], tier: number): string[] {
  const oldCtx = oldPieces.map((p, i) => (p.kind === 'ctx' ? i : -1)).filter((i) => i >= 0);
  const fileLead = leadWs(lines[at]!);
  const oldLead = leadWs(oldPieces[0]!.text);
  const out: string[] = [];
  let seenCtx = 0;
  for (const piece of newPieces) {
    if (piece.kind === 'ctx') {
      out.push(lines[at + oldCtx[seenCtx]!]!);
      seenCtx += 1;
      continue;
    }
    let text = piece.text;
    if (tier === 2 && text !== '' && text.trim() !== '' && text.startsWith(oldLead)) {
      text = fileLead + text.slice(oldLead.length);
    }
    out.push(text);
  }
  return out;
}

/** Apply one matched replacement to a file state, keeping its line break and
 * final break (only a unified hunk may change the final break, elsewhere). */
function spliceIn(state: FileState, at: number, oldLen: number, made: string[]): void {
  state.lines = [...state.lines.slice(0, at), ...made, ...state.lines.slice(at + oldLen)];
}

/** The format of a patch, by the first of the detection rules that holds. */
function detect(lines: string[]): string | null {
  const first = lines.findIndex((l) => !isBlank(l));
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const c = lines[first]!.trimStart()[0];
  if (c === '[' || c === '{') return 'json';
  if (lines.some((l) => l.trimEnd() === '<<<<<<< SEARCH')) return 'search-replace';
  if (lines.some((l, i) => l.startsWith('diff --git ') || (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')))) return 'unified';
  return null;
}

// ---- JSON edits (section 9) ----

/** One JSON edit object, or null when it is not one. */
function jsonEdit(v: unknown): Edit | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const p = o.path;
  if (typeof p !== 'string') return null;
  const present = (tree: Tree): string => {
    if (!safePath(p)) throw new EditFault('unsafe', p);
    const t = tree.current(p);
    if (t === null) throw new EditFault('missing', p);
    return t;
  };
  if (o.command === 'create' && typeof o.file_text === 'string') {
    const text = o.file_text;
    return (tree) => {
      if (!safePath(p)) throw new EditFault('unsafe', p);
      if (tree.current(p) !== null) throw new EditFault('exists', p);
      tree.set(p, text);
    };
  }
  if (o.command === 'str_replace' && typeof o.old_str === 'string' && o.old_str !== '' && (o.new_str === undefined || typeof o.new_str === 'string')) {
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      const before = present(tree);
      const nl = before.indexOf('\n');
      const crlf = nl > 0 && before[nl - 1] === '\r';
      const flat = (s: string): string => (crlf ? s.replaceAll('\r\n', '\n') : s);
      const t = flat(before);
      const old = flat(oldStr);
      let count = 0;
      for (let k = t.indexOf(old); k >= 0; k = t.indexOf(old, k + 1)) count += 1;
      if (count === 0) throw new EditFault('no-match', p);
      if (count > 1) throw new EditFault('ambiguous', p);
      const at = t.indexOf(old);
      const out = t.slice(0, at) + flat(newStr) + t.slice(at + old.length);
      tree.set(p, crlf ? out.replaceAll('\n', '\r\n') : out);
    };
  }
  if (o.command === 'insert' && typeof o.insert_line === 'number' && Number.isInteger(o.insert_line) && typeof o.new_str === 'string') {
    const line = o.insert_line;
    const add = patchLines(o.new_str);
    return (tree) => {
      const before = present(tree);
      const state = readState(before);
      if (line < 0 || line > state.lines.length) throw new EditFault('no-match', p);
      state.lines.splice(line, 0, ...add);
      tree.set(p, writeState(state));
    };
  }
  return null;
}

function jsonEdits(text: string, lines: string[]): Edit[] {
  const first = lines.findIndex((l) => !isBlank(l));
  if (first < 0) throw new ParseFault(1);
  const edits: Edit[] = [];
  if (lines[first]!.trimStart().startsWith('[')) {
    let v: unknown;
    try {
      v = JSON.parse(text);
    } catch {
      throw new ParseFault(first + 1);
    }
    if (!Array.isArray(v)) throw new ParseFault(first + 1);
    for (const x of v) {
      const e = jsonEdit(x);
      if (e === null) throw new ParseFault(first + 1);
      edits.push(e);
    }
    return edits;
  }
  lines.forEach((l, i) => {
    if (isBlank(l)) return;
    let e: Edit | null = null;
    try {
      e = jsonEdit(JSON.parse(l));
    } catch {
      e = null;
    }
    if (e === null) throw new ParseFault(i + 1);
    edits.push(e);
  });
  return edits;
}

// ---- Unified diffs (section 6) ----

interface Hunk {
  start: number;
  body: Piece[];
}

interface UniSection {
  oldPath: string | null;
  newPath: string | null;
  renameFrom: string | null;
  renameTo: string | null;
  diffOld: string | null;
  diffNew: string | null;
  isCreate: boolean;
  isDelete: boolean;
  pairSeen: boolean;
  sawHunk: boolean;
  hunks: Hunk[];
}

/** A `--- ` / `+++ ` path: text after the four characters, up to the first
 * tab, surrounding spaces removed; `/dev/null` means no file. */
function headerPath(line: string): string | null {
  const raw = line.slice(4).split('\t')[0]!.trim();
  const stripped = raw.replace(/^[ab]\//, '');
  return raw === '/dev/null' ? null : stripped;
}

function parseUnified(lines: string[], strict: boolean): Edit[] {
  const sections: UniSection[] = [];
  let cur: UniSection | null = null;
  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    if (line.startsWith('diff --git ')) {
      const m = /^diff --git a\/(\S+) b\/(\S+)$/.exec(line);
      if (m === null) throw new ParseFault(i + 1);
      cur = {
        oldPath: null, newPath: null, renameFrom: null, renameTo: null,
        diffOld: m[1]!, diffNew: m[2]!, isCreate: false, isDelete: false,
        pairSeen: false, sawHunk: false, hunks: [],
      };
      sections.push(cur);
      i += 1;
      continue;
    }
    if (line.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) {
      if (cur !== null && !cur.sawHunk && !cur.pairSeen) {
        cur.oldPath = headerPath(line);
        cur.newPath = headerPath(lines[i + 1]!);
        cur.pairSeen = true;
      } else {
        cur = {
          oldPath: headerPath(line), newPath: headerPath(lines[i + 1]!),
          renameFrom: null, renameTo: null, diffOld: null, diffNew: null,
          isCreate: false, isDelete: false, pairSeen: true, sawHunk: false, hunks: [],
        };
        sections.push(cur);
      }
      i += 2;
      continue;
    }
    if (line.startsWith('@@')) {
      if (cur === null) throw new ParseFault(i + 1);
      const m = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: |$)/.exec(line);
      if (m === null) throw new ParseFault(i + 1);
      const start = Number(m[1]);
      const body: Piece[] = [];
      let j = i + 1;
      while (j < lines.length) {
        const l = lines[j]!;
        if (l.startsWith('--- ') && (lines[j + 1] ?? '').startsWith('+++ ')) break;
        if (l === '' || l.startsWith(' ') || l.startsWith('-') || l.startsWith('+') || l.startsWith('\\')) {
          j += 1;
          continue;
        }
        break;
      }
      const raw = lines.slice(i + 1, j);
      while (raw.length > 0 && raw[raw.length - 1] === '') raw.pop();
      for (const l of raw) {
        if (l.startsWith('\\')) {
          if (body.length > 0) body[body.length - 1]!.backslash = true;
          continue;
        }
        if (l === '') body.push({ text: '', kind: 'ctx' });
        else if (l.startsWith(' ')) body.push({ text: l.slice(1), kind: 'ctx' });
        else if (l.startsWith('-')) body.push({ text: l.slice(1), kind: 'old' });
        else body.push({ text: l.slice(1), kind: 'new' });
      }
      cur.hunks.push({ start, body });
      cur.sawHunk = true;
      i = j;
      continue;
    }
    if (cur !== null && !cur.sawHunk) {
      if (line.startsWith('--- ')) throw new ParseFault(i + 2);
      if (line.startsWith('rename from ')) cur.renameFrom = line.slice('rename from '.length).trim();
      else if (line.startsWith('rename to ')) cur.renameTo = line.slice('rename to '.length).trim();
      else if (line.startsWith('new file mode')) cur.isCreate = true;
      else if (line.startsWith('deleted file mode')) cur.isDelete = true;
      i += 1;
      continue;
    }
    i += 1;
  }
  return sections.map((s) => unifiedEdit(s, strict));
}

/** Apply a section's hunks in order to a file state, each to the result of
 * the ones before it. */
function applyHunks(state: FileState, hunks: Hunk[], strict: boolean, path: string): void {
  let d = 0;
  for (const h of hunks) {
    const oldPieces = h.body.filter((e) => e.kind !== 'new');
    const newPieces = h.body.filter((e) => e.kind !== 'old');
    const oldTexts = oldPieces.map((e) => e.text);
    if (oldTexts.length > 0) {
      const m = matchBlock(state.lines, oldTexts, strict);
      if (m === null) throw new EditFault('no-match', path);
      const at = nearest(m.at, h.start - 1 + d);
      const made = replacement(state.lines, at, oldPieces, newPieces, m.tier);
      const reachesEnd = at + oldTexts.length === state.lines.length;
      if (reachesEnd) {
        const lastNew = [...newPieces].reverse().find((e) => e.kind === 'ctx' || e.kind === 'new');
        if (lastNew === undefined) state.end = at > 0;
        else state.end = lastNew.backslash !== true;
      }
      spliceIn(state, at, oldTexts.length, made);
      d += newPieces.length - oldPieces.length;
    } else {
      const at = Math.min(h.start + d, state.lines.length);
      const made = newPieces.map((e) => e.text);
      if (at === state.lines.length && made.length > 0) {
        const lastNew = newPieces[newPieces.length - 1]!;
        state.end = lastNew.backslash !== true;
      }
      spliceIn(state, at, 0, made);
      d += made.length;
    }
  }
}

/** One unified section as an edit. */
function unifiedEdit(s: UniSection, strict: boolean): Edit {
  const oldP = s.oldPath ?? s.renameFrom ?? s.diffOld;
  const newP = s.newPath ?? s.renameTo ?? s.diffNew;
  return (tree) => {
    if (s.isCreate || oldP === null) {
      // A creation: the new path must not exist; the file is the `+` lines
      // of all its hunks, in order, ended by a line break unless a `\` line
      // follows its last `+` line.
      if (newP === null) return;
      if (!safePath(newP)) throw new EditFault('unsafe', newP);
      if (tree.current(newP) !== null) throw new EditFault('exists', newP);
      const added: string[] = [];
      let lastPlus: Piece | null = null;
      for (const h of s.hunks) {
        for (const e of h.body) {
          if (e.kind === 'new') {
            added.push(e.text);
            lastPlus = e;
          }
        }
      }
      const noBreak = lastPlus !== null && lastPlus.backslash === true;
      tree.set(newP, added.length === 0 ? '' : added.join('\n') + (noBreak ? '' : '\n'));
      return;
    }
    if (s.isDelete || newP === null) {
      if (!safePath(oldP)) throw new EditFault('unsafe', oldP);
      if (tree.current(oldP) === null) throw new EditFault('missing', oldP);
      tree.set(oldP, null);
      return;
    }
    if (oldP !== newP) {
      if (!safePath(oldP)) throw new EditFault('unsafe', oldP);
      if (!safePath(newP)) throw new EditFault('unsafe', newP);
      const before = tree.current(oldP);
      if (before === null) throw new EditFault('missing', oldP);
      if (tree.current(newP) !== null) throw new EditFault('exists', newP);
      const state = readState(before);
      applyHunks(state, s.hunks, strict, oldP);
      tree.set(oldP, null);
      tree.set(newP, writeState(state));
      tree.noteMove(oldP, newP);
      return;
    }
    if (!safePath(oldP)) throw new EditFault('unsafe', oldP);
    const before = tree.current(oldP);
    if (before === null) throw new EditFault('missing', oldP);
    const state = readState(before);
    applyHunks(state, s.hunks, strict, oldP);
    tree.set(oldP, writeState(state));
  };
}

// ---- SEARCH/REPLACE blocks (section 7) ----

function parseSearchReplace(lines: string[], strict: boolean): Edit[] {
  const edits: Edit[] = [];
  let prevReplace = -1;
  let prevPath: string | null = null;
  let i = 0;
  while (i < lines.length) {
    if (lines[i]!.trimEnd() !== '<<<<<<< SEARCH') {
      i += 1;
      continue;
    }
    const searchAt = i;
    let path: string | null = null;
    for (let j = i - 1; j > prevReplace; j -= 1) {
      const t = lines[j]!.trim();
      if (t === '' || t.startsWith('```')) continue;
      path = t;
      break;
    }
    if (path === null) {
      if (prevPath === null) throw new ParseFault(searchAt + 1);
      path = prevPath;
    }
    let k = i + 1;
    let sep = -1;
    while (k < lines.length) {
      const t = lines[k]!.trimEnd();
      if (t === '=======') {
        sep = k;
        break;
      }
      if (t === '>>>>>>> REPLACE' || t === '<<<<<<< SEARCH') break;
      k += 1;
    }
    if (sep < 0) throw new ParseFault(searchAt + 1);
    let m = sep + 1;
    let end = -1;
    while (m < lines.length) {
      const t = lines[m]!.trimEnd();
      if (t === '>>>>>>> REPLACE') {
        end = m;
        break;
      }
      if (t === '<<<<<<< SEARCH') break;
      m += 1;
    }
    if (end < 0) throw new ParseFault(searchAt + 1);
    const oldLines = lines.slice(searchAt + 1, sep);
    const newLines = lines.slice(sep + 1, end);
    const p = path;
    edits.push((tree) => {
      if (!safePath(p)) throw new EditFault('unsafe', p);
      if (oldLines.length === 0) {
        if (tree.current(p) !== null) throw new EditFault('exists', p);
        tree.set(p, newLines.length === 0 ? '' : newLines.join('\n') + '\n');
        return;
      }
      const before = tree.current(p);
      if (before === null) throw new EditFault('missing', p);
      const state = readState(before);
      const found = matchBlock(state.lines, oldLines, strict);
      if (found === null) throw new EditFault('no-match', p);
      if (found.at.length !== 1) throw new EditFault('ambiguous', p);
      const oldPieces: Piece[] = oldLines.map((t) => ({ text: t, kind: 'old' as const }));
      const newPieces: Piece[] = newLines.map((t) => ({ text: t, kind: 'new' as const }));
      const made = replacement(state.lines, found.at[0]!, oldPieces, newPieces, found.tier);
      spliceIn(state, found.at[0]!, oldLines.length, made);
      tree.set(p, writeState(state));
    });
    prevPath = path;
    prevReplace = end;
    i = end + 1;
  }
  return edits;
}

// ---- V4A patches (section 8) ----

interface V4aChunk {
  anchor: string | null;
  eof: boolean;
  body: Piece[];
}

function parseV4a(lines: string[], strict: boolean): Edit[] {
  const first = lines.findIndex((l) => !isBlank(l));
  if (first < 0) throw new ParseFault(1);
  if (lines[first]!.trim() !== '*** Begin Patch') throw new ParseFault(first + 1);
  const last = lines.map((l) => !isBlank(l)).lastIndexOf(true);
  if (lines[last]!.trim() !== '*** End Patch') throw new ParseFault(lines.length + 1);
  const edits: Edit[] = [];
  let i = first + 1;
  while (i < last) {
    const line = lines[i]!;
    if (isBlank(line)) {
      i += 1;
      continue;
    }
    let kind: 'add' | 'delete' | 'update' | null = null;
    let path = '';
    for (const [prefix, k] of [['*** Add File:', 'add'], ['*** Delete File:', 'delete'], ['*** Update File:', 'update']] as const) {
      if (line.startsWith(prefix)) {
        kind = k;
        path = line.slice(prefix.length).trim();
        break;
      }
    }
    if (kind === null) throw new ParseFault(i + 1);
    const raw: { text: string; at: number }[] = [];
    let j = i + 1;
    while (j < last) {
      const l = lines[j]!;
      if (l.startsWith('*** Add File:') || l.startsWith('*** Delete File:') || l.startsWith('*** Update File:')) break;
      raw.push({ text: l, at: j });
      j += 1;
    }
    while (raw.length > 0 && isBlank(raw[raw.length - 1]!.text)) raw.pop();
    edits.push(v4aEdit(kind, path, i, raw, strict));
    i = j;
  }
  return edits;
}

/** Parse one V4A section's lines into its edit. */
function v4aEdit(kind: 'add' | 'delete' | 'update', path: string, headerAt: number, raw: { text: string; at: number }[], strict: boolean): Edit {
  if (kind === 'add') {
    const made: string[] = [];
    for (const { text, at } of raw) {
      if (text.startsWith('+')) made.push(text.slice(1));
      else if (text === '') made.push('');
      else throw new ParseFault(at + 1);
    }
    return (tree) => {
      if (!safePath(path)) throw new EditFault('unsafe', path);
      if (tree.current(path) !== null) throw new EditFault('exists', path);
      tree.set(path, made.length === 0 ? '' : made.join('\n') + '\n');
    };
  }
  if (kind === 'delete') {
    if (raw.length > 0) throw new ParseFault(raw[0]!.at + 1);
    return (tree) => {
      if (!safePath(path)) throw new EditFault('unsafe', path);
      if (tree.current(path) === null) throw new EditFault('missing', path);
      tree.set(path, null);
    };
  }
  // Update: an optional `*** Move to:` first line, then chunks.
  let moveTo: string | null = null;
  let idx = 0;
  if (raw.length > 0 && raw[0]!.text.startsWith('*** Move to:')) {
    moveTo = raw[0]!.text.slice('*** Move to:'.length).trim();
    idx = 1;
  }
  const chunks: V4aChunk[] = [];
  let cur: V4aChunk | null = null;
  const isBody = (t: string): boolean => t === '' || t.startsWith(' ') || t.startsWith('-') || t.startsWith('+');
  while (idx < raw.length) {
    const { text, at } = raw[idx]!;
    if (text.startsWith('@@')) {
      cur = { anchor: text.slice(2).trim() === '' ? null : text.slice(2).trim(), eof: false, body: [] };
      chunks.push(cur);
      idx += 1;
      continue;
    }
    if (text === '*** End of File') {
      if (cur === null) throw new ParseFault(at + 1);
      cur.eof = true;
      cur = null;
      idx += 1;
      continue;
    }
    if (!isBody(text)) throw new ParseFault(at + 1);
    if (cur === null) {
      if (chunks.length > 0) throw new ParseFault(at + 1);
      cur = { anchor: null, eof: false, body: [] };
      chunks.push(cur);
    }
    if (text === '') cur.body.push({ text: '', kind: 'ctx' });
    else if (text.startsWith(' ')) cur.body.push({ text: text.slice(1), kind: 'ctx' });
    else if (text.startsWith('-')) cur.body.push({ text: text.slice(1), kind: 'old' });
    else cur.body.push({ text: text.slice(1), kind: 'new' });
    idx += 1;
  }
  if (chunks.length === 0 && moveTo === null) throw new ParseFault(headerAt + 1);
  const target = moveTo;
  return (tree) => {
    if (!safePath(path)) throw new EditFault('unsafe', path);
    const before = tree.current(path);
    if (before === null) throw new EditFault('missing', path);
    if (target !== null) {
      if (!safePath(target)) throw new EditFault('unsafe', target);
      if (tree.current(target) !== null) throw new EditFault('exists', target);
    }
    const state = readState(before);
    applyChunks(state, chunks, strict, path);
    const out = writeState(state);
    if (target !== null) {
      tree.set(path, null);
      tree.set(target, out);
      tree.noteMove(path, target);
    } else {
      tree.set(path, out);
    }
  };
}

/** Apply a V4A section's chunks in order, with the cursor that starts at
 * line 0 and moves to just after the lines each chunk wrote. */
function applyChunks(state: FileState, chunks: V4aChunk[], strict: boolean, path: string): void {
  let cursor = 0;
  for (const chunk of chunks) {
    if (chunk.anchor !== null) {
      let found: number | null = null;
      const tiers = strict ? [0] : [0, 1, 2];
      for (const t of tiers) {
        for (let j = cursor; j < state.lines.length; j += 1) {
          if (tierLine(state.lines[j]!, t) === tierLine(chunk.anchor, t)) {
            found = j;
            break;
          }
        }
        if (found !== null) break;
      }
      if (found === null) throw new EditFault('no-match', path);
      cursor = found + 1;
    }
    const oldPieces = chunk.body.filter((e) => e.kind !== 'new');
    const newPieces = chunk.body.filter((e) => e.kind !== 'old');
    const oldTexts = oldPieces.map((e) => e.text);
    if (oldTexts.length === 0) {
      const at = chunk.eof ? state.lines.length : cursor;
      const made = newPieces.map((e) => e.text);
      spliceIn(state, at, 0, made);
      cursor = at + made.length;
      continue;
    }
    let hit: { tier: number; at: number } | null = null;
    const tiers = strict ? [0] : [0, 1, 2];
    for (const t of tiers) {
      const at = occurrences(state.lines, oldTexts, t).filter((p) => p >= cursor && (!chunk.eof || p + oldTexts.length === state.lines.length));
      if (at.length > 0) {
        hit = { tier: t, at: at[0]! };
        break;
      }
    }
    if (hit === null) throw new EditFault('no-match', path);
    const made = replacement(state.lines, hit.at, oldPieces, newPieces, hit.tier);
    spliceIn(state, hit.at, oldTexts.length, made);
    cursor = hit.at + made.length;
  }
}

// ---- Running the CLI (sections 1, 2, 10) ----

function run(args: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let check = false;
  let strict = false;
  const pos: string[] = [];
  let ended = false;
  for (const a of args) {
    if (!ended && a === '--') ended = true;
    else if (!ended && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null && FORMATS.includes(a.slice(9))) format = a.slice(9);
      else if (a === '--check' && !check) check = true;
      else if (a === '--strict' && !strict) strict = true;
      else return { code: 2, out: ['error: usage'] };
    } else pos.push(a);
  }
  if (pos.length > 1) return { code: 2, out: ['error: usage'] };
  const file = pos.length === 0 || pos[0] === '-' ? null : pos[0]!;
  let text = '';
  try {
    text = readFileSync(file ?? 0, 'utf8');
  } catch {
    if (file !== null) return { code: 2, out: ['error: no-patch'] };
  }
  const lines = patchLines(text);
  const kind = format === null || format === 'auto' ? detect(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };
  let edits: Edit[];
  try {
    if (kind === 'json') edits = jsonEdits(text, lines);
    else if (kind === 'unified') edits = parseUnified(lines, strict);
    else if (kind === 'search-replace') edits = parseSearchReplace(lines, strict);
    else edits = parseV4a(lines, strict);
    if (edits.length === 0) throw new ParseFault(1);
  } catch (e) {
    if (e instanceof ParseFault) return { code: 2, out: [`error: parse ${e.no}`] };
    throw e;
  }
  const tree = new Tree();
  const fails: EditFault[] = [];
  for (const edit of edits) {
    const before = tree.snapshot();
    try {
      edit(tree);
    } catch (e) {
      if (!(e instanceof EditFault)) throw e;
      tree.restore(before);
      fails.push(e);
    }
  }
  if (fails.length > 0) {
    return { code: 1, out: [...fails.map((f) => `error: ${f.code} ${f.path}`), `failed: ${fails.length} of ${edits.length} edits`] };
  }
  // The report rows, in byte order of the paths (a move sorts by its new
  // path and replaces that pair's delete and add lines).
  const rows: { key: string; line: string }[] = [];
  const hidden = new Set<string>();
  const seenMoves = new Set<string>();
  for (const mv of tree.moveList()) {
    const tag = `${mv.old}\n${mv.next}`;
    if (seenMoves.has(tag)) continue;
    seenMoves.add(tag);
    if (tree.original(mv.old) !== null && tree.current(mv.old) === null && tree.original(mv.next) === null && tree.current(mv.next) !== null) {
      rows.push({ key: mv.next, line: `R ${mv.old} -> ${mv.next}` });
      hidden.add(`D:${mv.old}`);
      hidden.add(`A:${mv.next}`);
    }
  }
  for (const p of tree.changedPaths()) {
    const was = tree.original(p);
    const now = tree.current(p);
    if (was === now) continue;
    const kind = was === null ? 'A' : now === null ? 'D' : 'M';
    if (hidden.has(`${kind}:${p}`)) continue;
    rows.push({ key: p, line: `${kind} ${p}` });
  }
  rows.sort((a, b) => byteOrder(a.key, b.key));
  if (!check) {
    for (const p of tree.changedPaths()) {
      const now = tree.current(p);
      if (now === null) rmSync(p, { force: true });
      else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  const linesOut = rows.map((r) => r.line);
  return { code: 0, out: [...linesOut, `ok: ${linesOut.length} ${linesOut.length === 1 ? 'file' : 'files'}`] };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = run(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}
