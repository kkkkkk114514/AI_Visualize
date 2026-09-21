import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import type { GridEnv } from "../../api/types";
import { useAlgoStore } from "../../stores/algoStore";
import { ParamFields } from "../algo/ParamFields";

type Brush = "obstacle" | "start" | "goal" | "reward";

const BRUSHES: Brush[] = ["obstacle", "start", "goal", "reward"];

/** 格子比较：`rewards` 的条目带第三位奖励值，只看前两位。 */
const sameCell = (cell: readonly number[], x: number, y: number) => cell[0] === x && cell[1] === y;

/** 应用画笔：非法位置静默跳过，障碍与奖励再点取消（docs/02 §13.6）。 */
function nextEnv(env: GridEnv, brush: Brush, x: number, y: number): GridEnv {
  const onStart = sameCell(env.start, x, y);
  const onGoal = sameCell(env.goal, x, y);
  const isObstacle = env.obstacles.some((cell) => sameCell(cell, x, y));
  const hasReward = env.rewards.some((cell) => sameCell(cell, x, y));

  if (brush === "obstacle") {
    if (onStart || onGoal) return env;
    if (isObstacle) {
      return { ...env, obstacles: env.obstacles.filter((cell) => !sameCell(cell, x, y)) };
    }
    return {
      ...env,
      obstacles: [...env.obstacles, [x, y]],
      rewards: env.rewards.filter((cell) => !sameCell(cell, x, y)),
    };
  }
  if (brush === "start") {
    if (onGoal) return env;
    return {
      ...env,
      start: [x, y],
      obstacles: env.obstacles.filter((cell) => !sameCell(cell, x, y)),
    };
  }
  if (brush === "goal") {
    if (onStart) return env;
    return {
      ...env,
      goal: [x, y],
      obstacles: env.obstacles.filter((cell) => !sameCell(cell, x, y)),
    };
  }
  if (isObstacle) return env;
  return {
    ...env,
    rewards: hasReward
      ? env.rewards.filter((cell) => !sameCell(cell, x, y))
      : [...env.rewards, [x, y, 1.0]],
  };
}

/** RL 训练配置（docs/02 §13.6）：顶部环境网格编辑器 + schema 驱动的算法参数表单。 */
export function RLConfigForm() {
  const { t } = useTranslation();
  const spec = useAlgoStore((state) => state.spec);
  const schemas = useAlgoStore((state) => state.schemas);
  const schemasError = useAlgoStore((state) => state.schemasError);
  const loadSchemas = useAlgoStore((state) => state.loadSchemas);
  const setParam = useAlgoStore((state) => state.setParam);
  const setEnv = useAlgoStore((state) => state.setEnv);
  const fromRunId = useAlgoStore((state) => state.fromRunId);
  const [brush, setBrush] = useState<Brush>("obstacle");

  useEffect(() => {
    void loadSchemas();
  }, [loadSchemas]);

  if (!spec) return null;
  const env = spec.env;
  const locked = fromRunId !== null;
  const schema = schemas?.[spec.algo] ?? null;
  const everyN = spec.probe_defaults?.every_n_steps ?? null;

  return (
    <div className="config">
      <div className="config__section">
        <div className="config__title">{t("algo.grid.editorTitle")}</div>
        <div className="grid-editor__brushes" role="group">
          {BRUSHES.map((item) => (
            <button
              key={item}
              type="button"
              className={`grid-editor__brush${brush === item ? " grid-editor__brush--active" : ""}`}
              disabled={locked}
              data-testid={`brush-${item}`}
              onClick={() => setBrush(item)}
              title={t(`algo.grid.brush.${item}`)}
            >
              <i className={`grid-editor__swatch grid-editor__swatch--${item}`} />
              {t(`algo.grid.brush.${item}`)}
            </button>
          ))}
        </div>
        {env ? (
          <div
            className="grid-editor"
            style={{ gridTemplateColumns: `repeat(${env.width}, 1fr)` }}
            data-testid="grid-editor"
            data-size={`${env.width}x${env.height}`}
          >
            {Array.from({ length: env.height }, (_, row) => {
              // 网格 y 向上、行 0 在底部，DOM 自上而下渲染需倒序
              const y = env.height - 1 - row;
              return Array.from({ length: env.width }, (_, x) => {
                const onStart = sameCell(env.start, x, y);
                const onGoal = sameCell(env.goal, x, y);
                const isObstacle = env.obstacles.some((cell) => sameCell(cell, x, y));
                const reward = env.rewards.find((cell) => sameCell(cell, x, y));
                const kind = isObstacle
                  ? "obstacle"
                  : onStart
                    ? "start"
                    : onGoal
                      ? "goal"
                      : reward
                        ? "reward"
                        : "plain";
                return (
                  <button
                    key={`${x}-${y}`}
                    type="button"
                    className={`grid-cell grid-cell--${kind}`}
                    disabled={locked}
                    data-testid={`cell-${x}-${y}`}
                    data-cell={kind}
                    onClick={() => env && setEnv(nextEnv(env, brush, x, y))}
                    title={`(${x}, ${y})`}
                  />
                );
              });
            })}
          </div>
        ) : (
          <div className="config__meta">{t("algo.grid.noEnv")}</div>
        )}
        <div className="config__meta">
          {env
            ? `${env.width}×${env.height} · ${t("algo.grid.envMeta", {
                obstacles: env.obstacles.length,
                rewards: env.rewards.length,
              })}`
            : null}
        </div>
        <div className="config__hint">{t("algo.grid.hint")}</div>
      </div>

      <div className="config__section">
        <div className="config__title">{t("algo.params.title")}</div>
        {schema ? (
          <ParamFields
            fields={schema.fields}
            params={spec.params}
            disabled={locked}
            onChange={setParam}
          />
        ) : (
          <div className="config__meta">{schemasError ?? t("common.loading")}</div>
        )}
        {everyN ? <div className="config__hint">{t("algo.probeNote", { n: everyN })}</div> : null}
        {locked ? <div className="config__hint">{t("run.replayLockedHint")}</div> : null}
      </div>
    </div>
  );
}
