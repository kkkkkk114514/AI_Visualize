import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import type { FlowGraphNode } from "../../graph/flow";
import { specOf } from "../../graph/specs";
import { useGraphStore } from "../../stores/graphStore";
import { ParamField } from "./ParamField";
import { formatIssueArgs, formatParams, formatShape, formatShapes } from "./format";

const MAX_INPUT_PORTS = 4;

export function GraphNodeCard({ id, data, selected }: NodeProps<FlowGraphNode>) {
  const { t } = useTranslation();
  const spec = specOf(data.nodeType);
  const readOnly = useGraphStore((state) => state.readOnly);
  const info = useGraphStore((state) => state.infer.byNode[id]);
  const edges = useGraphStore((state) => state.edges);
  const inferErrors = useGraphStore((state) => state.infer.errors);
  const localIssues = useGraphStore((state) => state.localIssues);
  const updateParams = useGraphStore((state) => state.updateParams);

  const issue = useMemo(() => {
    const fromInfer = inferErrors.find((item) => item.node_id === id);
    if (fromInfer) return fromInfer;
    return localIssues.find((item) => item.nodeId === id && item.severity === "error") ?? null;
  }, [inferErrors, localIssues, id]);

  const inputPorts = useMemo(() => {
    const used = edges
      .filter((edge) => edge.target === id)
      .map((edge) => String(edge.targetHandle ?? "in"))
      .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    if (!spec?.multiInput) return ["in"];
    if (readOnly) return used.length ? used : ["in1", "in2"];
    const ports = [...used];
    for (let index = 1; index <= MAX_INPUT_PORTS; index += 1) {
      const port = `in${index}`;
      if (!ports.includes(port)) {
        ports.push(port);
        break;
      }
    }
    return ports.length >= 2 ? ports : ["in1", "in2"];
  }, [edges, id, readOnly, spec?.multiInput]);

  const issueText = useMemo(() => {
    if (!issue) return null;
    const messageKey = "message_key" in issue ? issue.message_key : issue.messageKey;
    const args = "args" in issue ? issue.args : undefined;
    return t(messageKey, { ...formatIssueArgs(args, t), defaultValue: messageKey });
  }, [issue, t]);

  const inText = info?.in_shapes ? formatShapes(info.in_shapes) : formatShape(info?.in_shape);
  const outText = formatShape(info?.out_shape);
  const paramsText = formatParams(info?.params);

  return (
    <div
      className={[
        "graph-node",
        `graph-node--${spec?.category ?? "basic"}`,
        selected ? "graph-node--selected" : "",
        issue ? "graph-node--error" : "",
        readOnly ? "graph-node--readonly" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      {inputPorts.map((port, index) => (
        <Handle
          key={port}
          id={port}
          type="target"
          position={Position.Left}
          className="graph-node__handle"
          style={{ top: `${((index + 0.5) / inputPorts.length) * 100}%` }}
          isConnectable={!readOnly}
        />
      ))}

      <div className="graph-node__header">
        <span className="graph-node__type">{data.nodeType}</span>
        {spec?.multiInput ? <span className="graph-node__tag">{t("editor.multiInput")}</span> : null}
      </div>

      {spec && spec.params.length > 0 ? (
        <div className="graph-node__params">
          {spec.params.map((paramSpec) => (
            <ParamField
              key={paramSpec.key}
              spec={paramSpec}
              value={data.params[paramSpec.key]}
              readOnly={readOnly}
              onChange={(value) => updateParams(id, { [paramSpec.key]: value })}
            />
          ))}
        </div>
      ) : null}

      <div className="graph-node__shapes">
        <span title={t("editor.shapeIn")}>
          <span className="graph-node__shapes-label">in</span> {inText}
        </span>
        <span title={t("editor.shapeOut")}>
          <span className="graph-node__shapes-label">out</span> {outText}
        </span>
        <span title={t("editor.paramsCount")}>
          <span className="graph-node__shapes-label">#</span> {paramsText}
        </span>
      </div>

      {issueText ? (
        <div className="graph-node__error" title={issue && "detail" in issue ? issue.detail || undefined : undefined}>
          {issueText}
        </div>
      ) : null}

      <Handle
        id="out"
        type="source"
        position={Position.Right}
        className="graph-node__handle"
        isConnectable={!readOnly}
      />
    </div>
  );
}
