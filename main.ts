import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const MAX_STATES = 500_000;
const INF = 9999;

type Push = [box: number, dir: number];

interface Level {
  w: number;
  h: number;
  n: number;
  wall: Uint8Array;
  goal: Uint8Array;
  box: Uint8Array;
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const box = new Uint8Array(n);
  let player = 0;
  for (let y = 0; y < h; y++) {
    const row = rows[y]!;
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goal[i] = 1;
      if (ch === '$' || ch === '*') box[i] = 1;
      if (ch === '@' || ch === '+') player = i;
    }
  }
  return { w, h, n, wall, goal, box, player };
}

function walkPath(w: number, wall: Uint8Array, boxes: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const n = wall.length;
  const prev = new Int32Array(n).fill(-1);
  const move = new Uint8Array(n);
  const dirs = [-w, w, -1, 1];
  const letters = ['u', 'd', 'l', 'r'];
  const q = [from];
  prev[from] = from;
  for (let qi = 0; qi < q.length; qi++) {
    const u = q[qi]!;
    for (let di = 0; di < 4; di++) {
      const v = u + dirs[di]!;
      if (v < 0 || v >= n || wall[v] || boxes[v] || prev[v] !== -1) continue;
      prev[v] = u;
      move[v] = di;
      if (v === to) {
        let out = '';
        for (let cur = to; cur !== from; cur = prev[cur]!) out = letters[move[cur]!]! + out;
        return out;
      }
      q.push(v);
    }
  }
  return '';
}

function solveRoom(
  w: number,
  wall: Uint8Array,
  goals: number[],
  initBoxes: number[],
  initPlayer: number,
): Push[] | null {
  const n = wall.length;
  const k = goals.length;
  if (k === 0) return [];
  const dirs = [-w, w, -1, 1];
  const isGoal = new Uint8Array(n);
  for (const g of goals) isGoal[g] = 1;

  // Reverse-BFS from each goal to compute exact single-box push distances and static deadlocks.
  const dist: Int16Array[] = [];
  const live = new Uint8Array(n);
  for (let gi = 0; gi < k; gi++) {
    const dArr = new Int16Array(n).fill(INF);
    const g = goals[gi]!;
    dArr[g] = 0;
    live[g] = 1;
    const q = [g];
    for (let qi = 0; qi < q.length; qi++) {
      const u = q[qi]!;
      const du = dArr[u]! + 1;
      for (let di = 0; di < 4; di++) {
        const d = dirs[di]!;
        const prev = u - d;
        const ppos = prev - d;
        if (prev < 0 || prev >= n || ppos < 0 || ppos >= n) continue;
        if (wall[prev] || wall[ppos] || dArr[prev]! <= du) continue;
        dArr[prev] = du;
        live[prev] = 1;
        q.push(prev);
      }
    }
    dist.push(dArr);
  }

  // Exact minimum-weight bipartite matching heuristic via bitmask DP (admissible & consistent).
  const maskSize = 1 << k;
  const dp = new Int16Array(maskSize);
  const popcnt = new Uint8Array(maskSize);
  for (let m = 1; m < maskSize; m++) popcnt[m] = popcnt[m & (m - 1)]! + 1;

  function heuristic(bs: number[]): number {
    dp.fill(INF);
    dp[0] = 0;
    for (let m = 0; m < maskSize - 1; m++) {
      const base = dp[m]!;
      if (base >= INF) continue;
      const b = bs[popcnt[m]!]!;
      for (let g = 0; g < k; g++) {
        if (m & (1 << g)) continue;
        const v = base + dist[g]![b]!;
        const nm = m | (1 << g);
        if (v < dp[nm]!) dp[nm] = v;
      }
    }
    return dp[maskSize - 1]!;
  }

  const boxMark = new Uint8Array(n);
  const vis = new Int32Array(n);
  const outerVis = new Int32Array(n);
  let visStamp = 0;
  let outerStamp = 0;
  const bfsQ = new Int32Array(n);

  function reachAndMin(start: number, mark: Int32Array, stamp: number): number {
    let head = 0;
    let tail = 0;
    bfsQ[tail++] = start;
    mark[start] = stamp;
    let minSq = start;
    while (head < tail) {
      const u = bfsQ[head++]!;
      for (let di = 0; di < 4; di++) {
        const v = u + dirs[di]!;
        if (v < 0 || v >= n || wall[v] || boxMark[v] || mark[v] === stamp) continue;
        mark[v] = stamp;
        if (v < minSq) minSq = v;
        bfsQ[tail++] = v;
      }
    }
    return minSq;
  }

  const quadOffsets = [-w - 1, -w, -1, 0];
  function is2x2Deadlock(sq: number): boolean {
    for (let oi = 0; oi < 4; oi++) {
      const tl = sq + quadOffsets[oi]!;
      const c0 = tl;
      const c1 = tl + 1;
      const c2 = tl + w;
      const c3 = tl + w + 1;
      if (
        (wall[c0] || boxMark[c0]) &&
        (wall[c1] || boxMark[c1]) &&
        (wall[c2] || boxMark[c2]) &&
        (wall[c3] || boxMark[c3])
      ) {
        if (
          (boxMark[c0] && !isGoal[c0]) ||
          (boxMark[c1] && !isGoal[c1]) ||
          (boxMark[c2] && !isGoal[c2]) ||
          (boxMark[c3] && !isGoal[c3])
        ) {
          return true;
        }
      }
    }
    return false;
  }

  const useNumKey = k <= 5 && n <= 256;
  const makeKey = (p: number, bs: number[]): number | string => {
    if (useNumKey) {
      let key = p;
      for (let i = 0; i < k; i++) key = key * n + bs[i]!;
      return key;
    }
    return String.fromCharCode(p, ...bs);
  };

  const startBoxes = initBoxes.slice().sort((a, b) => a - b);
  for (const b of startBoxes) boxMark[b] = 1;
  const startMin = reachAndMin(initPlayer, vis, ++visStamp);
  for (const b of startBoxes) boxMark[b] = 0;

  const h0 = heuristic(startBoxes);
  if (h0 >= INF) return null;

  const sp: number[] = [startMin];
  const sb: number[][] = [startBoxes];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const pushBox: number[] = [0];
  const pushDir: number[] = [0];

  const best = new Map<number | string, number>();
  best.set(makeKey(startMin, startBoxes), 0);

  const buckets: number[][] = [];
  buckets[h0] = [0];
  let curF = h0;
  let expanded = 0;

  while (expanded < MAX_STATES) {
    while (curF < buckets.length && (!buckets[curF] || buckets[curF]!.length === 0)) curF++;
    if (curF >= buckets.length) return null;
    const s = buckets[curF]!.pop()!;
    const bs = sb[s]!;
    const g = sg[s]!;
    const p = sp[s]!;
    if (best.get(makeKey(p, bs)) !== s) continue;

    // Goal check
    let solved = true;
    for (let i = 0; i < k; i++) {
      if (!isGoal[bs[i]!]) {
        solved = false;
        break;
      }
    }
    if (solved) {
      const pushes: Push[] = [];
      for (let t = s; t > 0; t = parent[t]!) pushes.push([pushBox[t]!, pushDir[t]!]);
      pushes.reverse();
      return pushes;
    }

    expanded++;
    for (let i = 0; i < k; i++) boxMark[bs[i]!] = 1;
    const curStamp = ++outerStamp;
    reachAndMin(p, outerVis, curStamp);

    for (let i = 0; i < k; i++) {
      const b = bs[i]!;
      for (let di = 0; di < 4; di++) {
        const d = dirs[di]!;
        const pFrom = b - d;
        const bTo = b + d;
        if (outerVis[pFrom] !== curStamp) continue;
        if (wall[bTo] || boxMark[bTo] || !live[bTo]) continue;

        boxMark[b] = 0;
        boxMark[bTo] = 1;
        if (!is2x2Deadlock(bTo)) {
          const nb = bs.slice();
          nb[i] = bTo;
          // Insertion sort single moved element
          let j = i;
          while (j > 0 && nb[j - 1]! > nb[j]!) {
            const tmp = nb[j - 1]!;
            nb[j - 1] = nb[j]!;
            nb[j] = tmp;
            j--;
          }
          while (j + 1 < k && nb[j]! > nb[j + 1]!) {
            const tmp = nb[j + 1]!;
            nb[j + 1] = nb[j]!;
            nb[j] = tmp;
            j++;
          }
          const h = heuristic(nb);
          if (h < INF) {
            const nMin = reachAndMin(b, vis, ++visStamp);
            const nk = makeKey(nMin, nb);
            const ng = g + 1;
            const prevIdx = best.get(nk);
            if (prevIdx === undefined || sg[prevIdx]! > ng) {
              const id = sp.length;
              sp.push(nMin);
              sb.push(nb);
              sg.push(ng);
              parent.push(s);
              pushBox.push(b);
              pushDir.push(d);
              best.set(nk, id);
              const nf = ng + h;
              let bkt = buckets[nf];
              if (!bkt) {
                bkt = [];
                buckets[nf] = bkt;
              }
              bkt.push(id);
            }
          }
        }
        boxMark[bTo] = 0;
        boxMark[b] = 1;
      }
    }

    for (let i = 0; i < k; i++) boxMark[bs[i]!] = 0;
  }
  return null;
}

interface RoomTask {
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  hall: number;
}

function decomposeRooms(lv: Level): RoomTask[] | null {
  const { w, h, n, wall, goal, box, player } = lv;
  const hallY = Math.floor(player / w);
  // Ensure no box or goal sits on the hallway row.
  for (let x = 0; x < w; x++) {
    const i = hallY * w + x;
    if (box[i] || goal[i]) return null;
  }
  const seen = new Uint8Array(n);
  const dirs = [-w, w, -1, 1];
  const rooms: RoomTask[] = [];
  let totalBoxes = 0;
  let totalGoals = 0;
  for (let i = 0; i < n; i++) {
    if (box[i]) totalBoxes++;
    if (goal[i]) totalGoals++;
  }
  let assignedBoxes = 0;
  let assignedGoals = 0;

  for (let i = 0; i < n; i++) {
    if (wall[i] || seen[i] || Math.floor(i / w) === hallY) continue;
    const comp: number[] = [i];
    seen[i] = 1;
    const rBoxes: number[] = [];
    const rGoals: number[] = [];
    const halls: number[] = [];
    for (let qi = 0; qi < comp.length; qi++) {
      const u = comp[qi]!;
      if (box[u]) rBoxes.push(u);
      if (goal[u]) rGoals.push(u);
      for (let di = 0; di < 4; di++) {
        const v = u + dirs[di]!;
        if (v < 0 || v >= n || wall[v]) continue;
        if (Math.floor(v / w) === hallY) {
          if (!halls.includes(v)) halls.push(v);
        } else if (!seen[v]) {
          seen[v] = 1;
          comp.push(v);
        }
      }
    }
    if (rBoxes.length !== rGoals.length) return null;
    if (rBoxes.length === 0) continue;
    if (halls.length !== 1) return null;
    const rWall = new Uint8Array(n).fill(1);
    for (const c of comp) rWall[c] = 0;
    const hallSq = halls[0]!;
    rWall[hallSq] = 0;
    rooms.push({ wall: rWall, goals: rGoals.sort((a, b) => a - b), boxes: rBoxes.sort((a, b) => a - b), hall: hallSq });
    assignedBoxes += rBoxes.length;
    assignedGoals += rGoals.length;
  }
  if (assignedBoxes !== totalBoxes || assignedGoals !== totalGoals || rooms.length === 0) return null;
  return rooms;
}

function solve(text: string): string {
  const lv = parseLevel(text);
  const { w, wall } = lv;
  const dirLetter = new Map<number, string>([
    [-w, 'U'],
    [w, 'D'],
    [-1, 'L'],
    [1, 'R'],
  ]);

  const rooms = decomposeRooms(lv);
  const curBoxes = lv.box.slice();
  let curPlayer = lv.player;
  let result = '';

  const replayPushes = (pushes: Push[]): void => {
    for (const [b, d] of pushes) {
      result += walkPath(w, wall, curBoxes, curPlayer, b - d);
      result += dirLetter.get(d)!;
      curBoxes[b] = 0;
      curBoxes[b + d] = 1;
      curPlayer = b;
    }
  };

  if (rooms !== null) {
    for (const room of rooms) {
      const pushes = solveRoom(w, room.wall, room.goals, room.boxes, room.hall);
      if (pushes === null) return '';
      result += walkPath(w, wall, curBoxes, curPlayer, room.hall);
      curPlayer = room.hall;
      replayPushes(pushes);
    }
    return result;
  }

  // Fallback: solve the whole board as one room.
  const allGoals: number[] = [];
  const allBoxes: number[] = [];
  for (let i = 0; i < lv.n; i++) {
    if (lv.goal[i]) allGoals.push(i);
    if (lv.box[i]) allBoxes.push(i);
  }
  const pushes = solveRoom(w, wall, allGoals, allBoxes, lv.player);
  if (pushes === null) return '';
  replayPushes(pushes);
  return result;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
