import { ActionSchema } from "@browserbasehq/stagehand-protocol/schemas";
import type { Action, ActResult, ObserveResult } from "@browserbasehq/stagehand-protocol/types";
import { z } from "zod";
import type { Page } from "./page.js";
import type { Stagehand } from "./stagehand.js";

/**
 * Minimum Jaccard similarity between two snapshot fingerprints for a page to
 * count as a known state. A page matching more than one node is ambiguous and
 * never drives replay.
 */
export const STATE_MATCH_THRESHOLD = 0.95;

/**
 * Path cost of one `page.goto()`, counted in clicks. A URL shortcut replaces a
 * path of this many clicks or more; on a tie it wins, because a stored URL does
 * not depend on where a link is placed on the page.
 */
export const GOTO_COST = 2;

/** Times a URL must have led to the same state before replay may `goto` it. */
export const SHORTCUT_MIN_HITS = 2;

const NavigationNodeSchema = z.strictObject({
  id: z.string(),
  url: z.string(),
  fingerprint: z.array(z.string()),
  frontier: z.array(ActionSchema),
});

const NavigationEdgeSchema = z.strictObject({
  from: z.string(),
  to: z.string(),
  actions: z.array(ActionSchema).min(1),
});

/** A URL seen at a recorded state; `nodeId` is null once it was seen at different states. */
const NavigationUrlSchema = z.strictObject({
  url: z.string(),
  nodeId: z.string().nullable(),
  hits: z.number().int().positive(),
});

const NavigationGraphSchema = z.strictObject({
  nodes: z.array(NavigationNodeSchema),
  edges: z.array(NavigationEdgeSchema),
  urls: z.array(NavigationUrlSchema).default([]),
});

export type NavigationNode = z.infer<typeof NavigationNodeSchema>;
export type NavigationEdge = z.infer<typeof NavigationEdgeSchema>;
export type NavigationGraph = z.infer<typeof NavigationGraphSchema>;
export type NavigationGoto = { to: string; url: string };
export type NavigationStep = NavigationEdge | NavigationGoto;

export type StateRecognition =
  | { status: "matched"; nodeId: string }
  | { status: "unknown" }
  | { status: "ambiguous"; nodeIds: string[] };

export type ReplayResult =
  | { status: "completed" }
  | { status: "diverged"; step: number; expectedNodeId: string; actual: StateRecognition }
  | { status: "failed"; step: number; message: string };

type StateObservation = Pick<NavigationNode, "url" | "fingerprint">;

type ReplayParams = { stagehand: Pick<Stagehand, "act">; page: Page; path: NavigationStep[] };

/** Runs one edge's actions, returning a failure message or `undefined`. */
type EdgeRunner = (
  graph: NavigationGraph,
  run: Omit<ReplayParams, "path"> & { edge: NavigationEdge },
) => Promise<string | undefined>;

export function create(): NavigationGraph {
  return { nodes: [], edges: [], urls: [] };
}

export function toJSON(graph: NavigationGraph): string {
  return JSON.stringify(graph);
}

export function fromJSON(json: string): NavigationGraph {
  return NavigationGraphSchema.parse(JSON.parse(json));
}

export function frontier(graph: NavigationGraph, nodeId: string): readonly Action[] {
  return getNode(graph, nodeId).frontier;
}

export async function recognize(graph: NavigationGraph, page: Page): Promise<StateRecognition> {
  return matchState(graph, await observeState(page));
}

/**
 * Recognizes the current page, adding it as a new node when it is unknown, and
 * counts the page URL as one observation of a shortcut to that node.
 */
export async function recordState(graph: NavigationGraph, page: Page): Promise<StateRecognition> {
  const state = await observeState(page);
  const recognition = matchState(graph, state);
  if (recognition.status === "ambiguous") return recognition;
  const nodeId = recognition.status === "matched" ? recognition.nodeId : addNode(graph, state);
  recordUrl(graph, { url: state.url, nodeId });
  return { status: "matched", nodeId };
}

/** Records a successful `act()` as an edge from `from` to the state the page reached. */
export async function recordAct(
  graph: NavigationGraph,
  { page, from, result }: { page: Page; from: string; result: ActResult },
): Promise<StateRecognition> {
  const source = getNode(graph, from);
  if (!result.data.success || result.data.actions.length === 0) return recognize(graph, page);
  const target = await recordState(graph, page);
  if (target.status !== "matched") return target;
  const edge = { from, to: target.nodeId, actions: result.data.actions };
  if (!graph.edges.some((known) => sameEdge(known, edge))) graph.edges.push(edge);
  const tried = new Set(edge.actions.map(actionKey));
  source.frontier = source.frontier.filter((action) => !tried.has(actionKey(action)));
  return target;
}

/** Adds observed actions that are neither in the frontier nor already tried from `nodeId`. */
export function recordObserve(
  graph: NavigationGraph,
  { nodeId, result }: { nodeId: string; result: ObserveResult },
): void {
  const node = getNode(graph, nodeId);
  const known = new Set(
    graph.edges
      .filter((edge) => edge.from === nodeId)
      .flatMap((edge) => edge.actions)
      .concat(node.frontier)
      .map(actionKey),
  );
  for (const action of result.data) {
    if (known.has(actionKey(action))) continue;
    known.add(actionKey(action));
    node.frontier.push(action);
  }
}

/**
 * Cheapest path, counting each edge as one click and a trusted URL shortcut as
 * `GOTO_COST`, or `undefined` when `to` is unreachable. A shortcut works from
 * any state, so the cheapest path uses at most one, as its first step.
 */
export function findPath(
  graph: NavigationGraph,
  from: string,
  to: string,
): NavigationStep[] | undefined {
  const shortcuts = graph.urls.flatMap(({ url, nodeId, hits }) => {
    if (nodeId === null || hits < SHORTCUT_MIN_HITS) return [];
    const rest = edgePath(graph, nodeId, to);
    return rest ? [[{ to: nodeId, url }, ...rest]] : [];
  });
  const direct = edgePath(graph, from, to);
  const candidates: NavigationStep[][] = direct ? [...shortcuts, direct] : shortcuts;
  return candidates.reduce<NavigationStep[] | undefined>(
    (best, path) => (best && pathCost(best) <= pathCost(path) ? best : path),
    undefined,
  );
}

/** Shortest path over recorded edges by edge count (BFS). */
function edgePath(graph: NavigationGraph, from: string, to: string): NavigationEdge[] | undefined {
  const reachedBy = new Map<string, NavigationEdge | undefined>([[from, undefined]]);
  const queue = [from];
  for (const nodeId of queue) {
    if (nodeId === to) return unwindPath(reachedBy, to);
    for (const edge of graph.edges) {
      if (edge.from !== nodeId || reachedBy.has(edge.to)) continue;
      reachedBy.set(edge.to, edge);
      queue.push(edge.to);
    }
  }
  return undefined;
}

/**
 * Replays `path` through `act(action)` and `page.goto(url)` only, stopping as
 * soon as the page leaves it.
 */
export async function replay(graph: NavigationGraph, params: ReplayParams): Promise<ReplayResult> {
  return replaySteps(graph, params, runEdge);
}

/**
 * Like `replay`, but retries a failed action without arguments once through
 * `act(action.description)`, which uses inference. When the page then reaches
 * the edge's target state, the edge in `graph` keeps the healed actions, so
 * `path` must come from `findPath(graph, ...)`. Actions with arguments are not
 * retried, so their values never reach a prompt.
 */
export async function replayWithHealing(
  graph: NavigationGraph,
  params: ReplayParams,
): Promise<ReplayResult> {
  return replaySteps(graph, params, healEdge);
}

async function replaySteps(
  graph: NavigationGraph,
  { stagehand, page, path }: ReplayParams,
  runner: EdgeRunner,
): Promise<ReplayResult> {
  for (const [step, edge] of path.entries()) {
    if ("url" in edge) {
      await page.goto(edge.url);
      continue;
    }
    const divergence = await checkState(graph, page, { step, expectedNodeId: edge.from });
    if (divergence) return divergence;
    const message = await runner(graph, { stagehand, page, edge });
    if (message !== undefined) return { status: "failed", step, message };
  }
  const last = path.at(-1);
  if (!last) return { status: "completed" };
  const divergence = await checkState(graph, page, { step: path.length, expectedNodeId: last.to });
  return divergence ?? { status: "completed" };
}

const runEdge: EdgeRunner = async (_graph, { stagehand, page, edge }) => {
  for (const action of edge.actions) {
    const result = await stagehand.act(action, { page });
    if (!result.data.success) return result.data.message;
  }
  return undefined;
};

const healEdge: EdgeRunner = async (graph, { stagehand, page, edge }) => {
  const healed: Action[] = [];
  for (const action of edge.actions) {
    const result = await stagehand.act(action, { page });
    const final =
      result.data.success || action.arguments?.length
        ? result
        : await stagehand.act(action.description, { page });
    if (!final.data.success) return final.data.message;
    healed.push(...final.data.actions);
  }
  if (actionsKey(healed) === actionsKey(edge.actions)) return undefined;
  const reached = await recognize(graph, page);
  if (reached.status === "matched" && reached.nodeId === edge.to) edge.actions = healed;
  return undefined;
};

async function checkState(
  graph: NavigationGraph,
  page: Page,
  { step, expectedNodeId }: { step: number; expectedNodeId: string },
): Promise<ReplayResult | undefined> {
  const actual = await recognize(graph, page);
  if (actual.status === "matched" && actual.nodeId === expectedNodeId) return undefined;
  return { status: "diverged", step, expectedNodeId, actual };
}

async function observeState(page: Page): Promise<StateObservation> {
  const url = normalizeUrl(await page.url());
  const { formattedTree } = await page.snapshot();
  return { url, fingerprint: fingerprint(formattedTree) };
}

function matchState(graph: NavigationGraph, state: StateObservation): StateRecognition {
  const nodeIds = graph.nodes
    .filter((node) => node.url === state.url)
    .filter((node) => similarity(node.fingerprint, state.fingerprint) >= STATE_MATCH_THRESHOLD)
    .map((node) => node.id);
  const [nodeId, ...others] = nodeIds;
  if (nodeId === undefined) return { status: "unknown" };
  if (others.length > 0) return { status: "ambiguous", nodeIds };
  return { status: "matched", nodeId };
}

function normalizeUrl(raw: string): string {
  const url = new URL(raw);
  url.hash = "";
  url.searchParams.sort();
  return url.href;
}

/** Snapshot lines without their per-load element ids, deduplicated and sorted. */
function fingerprint(formattedTree: string): string[] {
  const lines = formattedTree.split("\n").map((line) => line.replace(/^\s*\[[^\]]*\]/, "").trim());
  return [...new Set(lines.filter(Boolean))].sort();
}

function similarity(known: string[], current: string[]): number {
  const knownLines = new Set(known);
  const shared = current.filter((line) => knownLines.has(line)).length;
  const union = knownLines.size + current.length - shared;
  return union === 0 ? 1 : shared / union;
}

function addNode(graph: NavigationGraph, state: StateObservation): string {
  const id = `node-${graph.nodes.length}`;
  graph.nodes.push({ id, ...state, frontier: [] });
  return id;
}

function recordUrl(graph: NavigationGraph, { url, nodeId }: { url: string; nodeId: string }): void {
  const known = graph.urls.find((entry) => entry.url === url);
  if (!known) {
    graph.urls.push({ url, nodeId, hits: 1 });
    return;
  }
  known.hits += 1;
  if (known.nodeId !== nodeId) known.nodeId = null;
}

function pathCost(path: NavigationStep[]): number {
  return path.reduce((cost, step) => cost + ("url" in step ? GOTO_COST : 1), 0);
}

function unwindPath(
  reachedBy: Map<string, NavigationEdge | undefined>,
  to: string,
): NavigationEdge[] {
  const path: NavigationEdge[] = [];
  for (let edge = reachedBy.get(to); edge; edge = reachedBy.get(edge.from)) path.unshift(edge);
  return path;
}

function sameEdge(a: NavigationEdge, b: NavigationEdge): boolean {
  return a.from === b.from && a.to === b.to && actionsKey(a.actions) === actionsKey(b.actions);
}

function actionsKey(actions: Action[]): string {
  return JSON.stringify(actions.map(actionKey));
}

function actionKey(action: Action): string {
  return JSON.stringify([action.selector, action.method, action.arguments]);
}

function getNode(graph: NavigationGraph, nodeId: string): NavigationNode {
  const node = graph.nodes.find((candidate) => candidate.id === nodeId);
  if (!node) throw new Error(`Unknown navigation graph node: ${nodeId}`);
  return node;
}
