import { useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";

import type { SnapshotPayload } from "../../../api/types";
import type { ColormapName } from "../paint";
import { colormapLut, decodePayload, formatValue, matrixViews } from "../paint";
import { MatrixGridView } from "./MatrixGridView";

interface HeatmapCanvasProps {
  payload: SnapshotPayload;
  colormap: ColormapName;
  kind: string;
}

/** 注意力矩阵 / 隐藏态：HTT 每头一张（带 token 刻度），TH 单张。 */
export function HeatmapCanvas({ payload, colormap, kind }: HeatmapCanvasProps) {
  const { t } = useTranslation();
  const decoded = useMemo(() => decodePayload(payload), [payload]);
  const views = useMemo(() => matrixViews(decoded), [decoded]);
  const lut = useMemo(() => colormapLut(colormap), [colormap]);
  const attention = kind === "attention" && views.length > 1;
  const causal = Boolean(decoded.meta?.causal);
  const aspect = views.length > 0 ? views[0].rows / views[0].cols : 1;

  const caption = useCallback(
    (index: number) => (attention ? t("run.probe.caption.head", { n: index }) : String(index)),
    [attention, t],
  );
  const readout = useCallback(
    (index: number, row: number, col: number, real: number) => {
      // 隐藏态是 (T,H) 单张矩阵，没有「头」的概念，读数按时间步 × 单元
      if (kind !== "attention") {
        return t("run.probe.readout.hidden", { row, col, value: formatValue(real) });
      }
      const parts = [t("run.probe.caption.head", { n: index }), `(${row}, ${col})`, formatValue(real)];
      if (causal && col > row) parts.push(t("run.probe.readout.masked"));
      return parts.join(" · ");
    },
    [causal, kind, t],
  );

  return (
    <MatrixGridView
      matrices={views}
      decoded={decoded}
      lut={lut}
      caption={attention ? caption : undefined}
      axes={attention}
      aspect={aspect}
      readout={readout}
    />
  );
}
