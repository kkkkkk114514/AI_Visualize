import type { GraphEdge, GraphIR, GraphNode } from "./ir";

const COLUMN_GAP = 300;
const ROW_GAP = 130;

/** 分层自动布局：按从 Input 出发的最长路径分层，列内垂直居中。 */
export function computeLayout(ir: GraphIR): Map<string, { x: number; y: number }> {
  const nodes: GraphNode[] = ir.nodes;
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  for (const node of nodes) {
    incoming.set(node.id, []);
    outgoing.set(node.id, []);
  }
  for (const edge of ir.edges as GraphEdge[]) {
    if (incoming.has(edge.target) && outgoing.has(edge.source)) {
      incoming.get(edge.target)!.push(edge.source);
      outgoing.get(edge.source)!.push(edge.target);
    }
  }

  const depth = new Map<string, number>();
  const roots = nodes.filter((n) => (incoming.get(n.id) ?? []).length === 0);
  const queue: string[] = roots.map((n) => n.id);
  for (const id of queue) depth.set(id, 0);
  while (queue.length) {
    const current = queue.shift()!;
    const currentDepth = depth.get(current) ?? 0;
    for (const next of outgoing.get(current) ?? []) {
      const candidate = currentDepth + 1;
      if ((depth.get(next) ?? -1) < candidate) {
        depth.set(next, candidate);
        queue.push(next);
      }
    }
  }
  for (const node of nodes) {
    if (!depth.has(node.id)) depth.set(node.id, 0);
  }

  const columns = new Map<number, string[]>();
  for (const node of nodes) {
    const d = depth.get(node.id)!;
    if (!columns.has(d)) columns.set(d, []);
    columns.get(d)!.push(node.id);
  }

  const positions = new Map<string, { x: number; y: number }>();
  for (const [d, ids] of columns) {
    const totalHeight = (ids.length - 1) * ROW_GAP;
    ids.forEach((id, index) => {
      positions.set(id, { x: d * COLUMN_GAP, y: index * ROW_GAP - totalHeight / 2 });
    });
  }
  return positions;
}
