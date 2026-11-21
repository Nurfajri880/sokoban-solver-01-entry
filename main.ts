// Sokoban solver for sokoban-solver-01-run-20261121.
//
// Idea: every level is a 1-wide hallway with 1-4 rooms off it, and a box pushed
// out of a room into the hallway can never come back, so the rooms are
// independent problems joined only by the player's walk. This program splits a
// level into its rooms, solves each room push-optimally on its own, and joins
// the room solutions with the walks between them.
//
// Each room is solved by a breadth-first search over pushes: a state is the
// boxes' squares plus the player's reachable region (named by its lowest
// square, so every player square in the same region is one state), and the
// search expands states in order of pushes made, so the first solved state it
// meets is push-optimal. Two prunings keep it small, and neither prunes a
// solvable state:
//   - dead squares: a square from which no push, ignoring the other boxes,
//     can ever bring a box onto a goal (found by pulling back from the goals);
//     a box on one is lost.
//   - frozen 2x2 blocks: four squares of walls and boxes with a box among
//     them off its goal; no push inside the block can ever move that box.
//
// The search is bounded by work (a fixed number of states per room), never by
// the clock, and every tie is broken by a fixed direction order, so two runs
// of this program answer byte for byte alike. Node's standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// The most states one room's search keeps before it gives up. The
// competition's generator keeps a room only when its own push-optimal search
// solves it within this many states, so a room drawn by it always fits.
const ROOM_STATE_BUDGET = 250_000;

// The four steps, in the fixed order every tie in this program breaks by.
const DIRS = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
] as const;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array; // 1 where a wall stands (the outermost frame counts as wall)
  goals: number[]; // sorted
  boxes: number[]; // sorted
  player: number;
}

/** A level in XSB text (# wall, space floor, . goal, $ box, * box on a goal, @ player, + player on a goal). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  goals.sort((a, b) => a - b);
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * The squares from which a box alone can be pushed onto some goal. Start at
 * the goals and pull backwards: a box on square y can be pushed in direction
 * d onto a live square x = y + d exactly when y and the square the player
 * must stand on, y - d, are both open floor.
 */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const size = wall.length;
  const live = new Uint8Array(size);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const dir of DIRS) {
      const d = dir.dy * w + dir.dx;
      const y = x - d;
      if (y < 0 || y >= size || live[y]) continue;
      const stand = y - d;
      if (stand < 0 || stand >= size) continue;
      if (wall[y] || wall[stand]) continue;
      live[y] = 1;
      queue.push(y);
    }
  }
  return live;
}

/** A room: its floor squares and the boxes and goals on them. */
interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * The level's rooms. Cut the dead squares of the player's row out of the
 * floor — that row is the hallway, and a box there reaches no goal — and each
 * connected part of the floor left that holds a box is a room. Null when the
 * level is no set of rooms (a box starts in the hallway, a goal is cut off,
 * or some part holds more boxes than goals).
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  const stack: number[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares: number[] = [s];
    part[s] = rooms.length;
    stack.length = 0;
    stack.push(s);
    while (stack.length > 0) {
      const p = stack.pop()!;
      for (const dir of DIRS) {
        const q = p + dir.dy * w + dir.dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
          stack.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * Flood the squares the player reaches from `from` around walls and the
 * boxes in `occupied`, marking them in `mark` with `stamp`; returns the
 * lowest square reached. `from` and every square reached are open floor, so
 * every neighbour read stays inside the board.
 */
function floodLow(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  mark: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  mark[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    const up = p - w;
    if (!wall[up] && !occupied[up] && mark[up] !== stamp) {
      mark[up] = stamp;
      stack[n++] = up;
    }
    const down = p + w;
    if (!wall[down] && !occupied[down] && mark[down] !== stamp) {
      mark[down] = stamp;
      stack[n++] = down;
    }
    const left = p - 1;
    if (!wall[left] && !occupied[left] && mark[left] !== stamp) {
      mark[left] = stamp;
      stack[n++] = left;
    }
    const right = p + 1;
    if (!wall[right] && !occupied[right] && mark[right] !== stamp) {
      mark[right] = stamp;
      stack[n++] = right;
    }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  goal: Uint8Array,
  t: number,
): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    const filled = (i: number): boolean => wall[i] === 1 || occupied[i] === 1;
    if (!filled(a) || !filled(b) || !filled(c) || !filled(d)) continue;
    const stuck = (i: number): boolean => occupied[i] === 1 && goal[i] === 0;
    if (stuck(a) || stuck(b) || stuck(c) || stuck(d)) return true;
  }
  return false;
}

/** A state's key: the region's lowest square and the boxes' squares, as digits in base `size`. */
function stateKey(boxes: number[], low: number, size: number): number {
  let k = low;
  for (const b of boxes) k = k * size + b;
  return k;
}

/** The shortest walk from `from` to `to` around walls and the boxes in `occupied`, in lower LURD letters. */
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how: (string | undefined)[] = new Array<string | undefined>(size);
  const queue: number[] = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const dir of DIRS) {
      const q = p + dir.dy * w + dir.dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = dir.step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** The square the player stands on after `moves` (LURD, either case) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves) {
    const c = ch.toLowerCase();
    p += c === 'u' ? -w : c === 'd' ? w : c === 'l' ? -1 : 1;
  }
  return p;
}

interface PushSolution {
  pushes: number;
  moves: string;
  states: number;
}

/**
 * The push-optimal solution of `level`, searching at most `maxStates` states,
 * or null when the search gives up (or the level has no solution). A state is
 * the boxes' squares and the player's reachable region (named by its lowest
 * square); states are expanded in order of pushes made, so the first solved
 * state met is push-optimal.
 */
function solvePushes(level: Level, maxStates: number): PushSolution | null {
  const { w, wall, goals } = level;
  const size = wall.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;

  const occupied = new Uint8Array(size);
  // Two stamp arrays: one holds the region of the state being expanded, the
  // other the region of each new state as its key is made, so the second
  // never overwrites the first mid-expansion.
  const regionMark = new Int32Array(size);
  const keyMark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b] === 1);

  // The states in the order found: their boxes, the player's square after the
  // push that made them, and their parent. The array itself is the queue.
  const statesBoxes: number[][] = [level.boxes.slice()];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const index = new Map<number, number>();
  for (const b of level.boxes) occupied[b] = 1;
  const firstLow = floodLow(w, wall, occupied, level.player, keyMark, ++stamp, stack);
  index.set(stateKey(level.boxes, firstLow, size), 0);
  for (const b of level.boxes) occupied[b] = 0;

  let found = solved(level.boxes) ? 0 : -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    floodLow(w, wall, occupied, statesPlayer[head]!, regionMark, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const dir of DIRS) {
        const d = dir.dy * w + dir.dx;
        const dest = b + d;
        const stand = b - d;
        if (regionMark[stand] !== here) continue; // the player cannot reach the push square
        if (wall[dest] || occupied[dest] || !live[dest]) continue;
        occupied[b] = 0;
        occupied[dest] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, dest);
        let key = -1;
        let nextBoxes: number[] | null = null;
        if (!frozen) {
          nextBoxes = boxes.slice();
          nextBoxes[bi] = dest;
          nextBoxes.sort((a, b2) => a - b2);
          const nextLow = floodLow(w, wall, occupied, b, keyMark, ++stamp, stack);
          key = stateKey(nextBoxes, nextLow, size);
        }
        occupied[dest] = 0;
        occupied[b] = 1;
        if (frozen || index.has(key)) continue;
        if (statesBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, statesBoxes.length);
        statesBoxes.push(nextBoxes!);
        statesPlayer.push(b);
        parent.push(head);
        if (solved(nextBoxes!)) {
          found = statesBoxes.length - 1;
          break;
        }
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // The path of states, then the moves: walk to each push's square, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes.slice();
  for (const s of path) {
    const next = statesBoxes[s]!;
    const from = boxes.find((b) => !next.includes(b))!;
    const to = next.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find((dd) => dd.dy * w + dd.dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir.push;
    player = from;
    boxes = next;
  }
  return { pushes: path.length, moves, states: statesBoxes.length };
}

/**
 * The level's moves: each room solved push-optimally in turn, the other
 * rooms' boxes standing as walls (on their goals once their room is done),
 * the player walking on from where the last room left it. An empty string
 * when a room cannot be solved (a valid answer, worth no points, so the other
 * instances still score).
 */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = roomsOf(level);
  if (rooms === null) return '';
  let player = level.player;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub: Level = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushes(sub, ROOM_STATE_BUDGET);
    if (solved === null) return '';
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(INPUT).sort()) {
  const moves = solveLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'));
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), `${moves}\n`);
}
