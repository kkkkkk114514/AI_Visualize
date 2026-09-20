import type { GraphEdge, GraphIR } from "./ir";
import { specOf } from "./specs";

export interface LocalIssue {
  code: string;
  messageKey: string;
  severity: "error" | "warning";
  nodeId?: string;
  edgeId?: string;
  args?: Record<string, unknown>;
}

const MAX_FAN_IN = 4;

/** 结构性预校验（环路、悬空端口、Input/Output 唯一性、入度上限）；形状级校验交给后端 dry-run。 */
export function validateLocal(ir: GraphIR): LocalIssue[] {
  const issues: LocalIssue[] = [];
  const byId = new Map(ir.nodes.map((n) => [n.id, n]));

  const seen = new Set<string>();
  for (const node of ir.nodes) {
    if (seen.has(node.id)) {
      issues.push({ code: "duplicate_node_id", messageKey: "errors.graph.duplicateNode", severity: "error", nodeId: node.id });
      continue;
    }
    seen.add(node.id);
    if (!specOf(node.type)) {
      issues.push({
        code: "unknown_node_type",
        messageKey: "errors.graph.unknownType",
        severity: "error",
        nodeId: node.id,
        args: { type: node.type },
      });
    }
  }

  const inputs = ir.nodes.filter((n) => n.type === "Input");
  const outputs = ir.nodes.filter((n) => n.type === "Output");
  if (inputs.length === 0) {
    issues.push({ code: "missing_input", messageKey: "errors.graph.missingInput", severity: "error" });
  } else if (inputs.length > 1) {
    issues.push({
      code: "multiple_inputs",
      messageKey: "errors.graph.multipleInput",
      severity: "error",
      nodeId: inputs[1].id,
      args: { count: inputs.length },
    });
  }
  if (outputs.length === 0) {
    issues.push({ code: "missing_output", messageKey: "errors.graph.missingOutput", severity: "error" });
  } else if (outputs.length > 1) {
    issues.push({
      code: "multiple_outputs",
      messageKey: "errors.graph.multipleOutput",
      severity: "error",
      nodeId: outputs[1].id,
      args: { count: outputs.length },
    });
  }

  const fanIn = new Map<string, GraphEdge[]>();
  for (const node of ir.nodes) fanIn.set(node.id, []);
  const adjacency = new Map<string, string[]>();
  for (const node of ir.nodes) adjacency.set(node.id, []);

  for (const edge of ir.edges) {
    if (!byId.has(edge.source) || !byId.has(edge.target)) {
      issues.push({
        code: "dangling_edge",
        messageKey: "errors.graph.danglingEdge",
        severity: "error",
        edgeId: edge.id,
        args: { source: edge.source, target: edge.target },
      });
      continue;
    }
    if (edge.source_port !== "out") {
      issues.push({
        code: "bad_port",
        messageKey: "errors.graph.badPort",
        severity: "error",
        edgeId: edge.id,
        args: { port: edge.source_port },
      });
      continue;
    }
    fanIn.get(edge.target)!.push(edge);
    adjacency.get(edge.source)!.push(edge.target);
  }

  for (const node of ir.nodes) {
    const spec = specOf(node.type);
    if (!spec) continue;
    const incoming = fanIn.get(node.id) ?? [];
    const limit = spec.multiInput ? MAX_FAN_IN : 1;
    if (incoming.length > limit) {
      issues.push({
        code: "fan_in_overflow",
        messageKey: "errors.graph.fanInOverflow",
        severity: "error",
        nodeId: node.id,
        args: { count: incoming.length, limit },
      });
    }
    if (spec.multiInput && incoming.length < 2) {
      issues.push({
        code: "fan_in_missing",
        messageKey: "errors.graph.fanInMissing",
        severity: "error",
        nodeId: node.id,
        args: { need: 2, count: incoming.length },
      });
    }
    if (node.type === "Output" && incoming.length === 0) {
      issues.push({ code: "output_no_input", messageKey: "errors.graph.outputNoInput", severity: "error", nodeId: node.id });
    }
  }

  // 与后端一致：已有结构性错误时不再做环路/可达性推导，避免同一根因刷出多条派生问题。
  const hasErrors = issues.some((issue) => issue.severity === "error");
  if (inputs.length === 1 && outputs.length === 1 && !hasErrors) {
    const state = new Map<string, number>();
    let cyclic = false;
    const visit = (id: string): void => {
      state.set(id, 1);
      for (const next of adjacency.get(id) ?? []) {
        const s = state.get(next) ?? 0;
        if (s === 1) {
          cyclic = true;
          return;
        }
        if (s === 0) visit(next);
        if (cyclic) return;
      }
      state.set(id, 2);
    };
    for (const node of ir.nodes) {
      if ((state.get(node.id) ?? 0) === 0) visit(node.id);
      if (cyclic) break;
    }
    if (cyclic) {
      issues.push({ code: "cycle", messageKey: "errors.graph.cycle", severity: "error" });
    } else {
      const reachable = new Set<string>([inputs[0].id]);
      const stack = [inputs[0].id];
      while (stack.length) {
        const current = stack.pop()!;
        for (const next of adjacency.get(current) ?? []) {
          if (!reachable.has(next)) {
            reachable.add(next);
            stack.push(next);
          }
        }
      }
      if (!reachable.has(outputs[0].id)) {
        issues.push({
          code: "output_unreachable",
          messageKey: "errors.graph.outputUnreachable",
          severity: "error",
          nodeId: outputs[0].id,
        });
      }
      for (const node of ir.nodes) {
        if (!reachable.has(node.id)) {
          issues.push({ code: "unreachable_node", messageKey: "errors.graph.unreachable", severity: "warning", nodeId: node.id });
        }
      }
    }
  }

  return issues;
}
