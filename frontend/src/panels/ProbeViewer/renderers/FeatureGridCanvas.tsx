import { useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";

import type { SnapshotPayload } from "../../../api/types";
import type { ColormapName } from "../paint";
import { colormapLut, decodePayload, formatValue, matrixViews } from "../paint";
import { MatrixGridView } from "./MatrixGridView";

interface FeatureGridCanvasProps {
  payload: SnapshotPayload;
  colormap: ColormapName;
}

/** 特征图 / 权重：CHW → 每通道一张热力图平铺，hover 由 min/max 反量化出真实数值。 */
export function FeatureGridCanvas({ payload, colormap }: FeatureGridCanvasProps) {
  const { t } = useTranslation();
  const decoded = useMemo(() => decodePayload(payload), [payload]);
  const views = useMemo(() => matrixViews(decoded), [decoded]);
  const lut = useMemo(() => colormapLut(colormap), [colormap]);
  const aspect = views.length > 0 ? views[0].rows / views[0].cols : 1;

  const caption = useCallback(
    (index: number) => t("run.probe.caption.channel", { n: index }),
    [t],
  );
  const readout = useCallback(
    (index: number, row: number, col: number, real: number) =>
      t("run.probe.readout.pixel", {
        channel: index,
        row,
        col,
        value: formatValue(real),
      }),
    [t],
  );

  return (
    <MatrixGridView
      matrices={views}
      decoded={decoded}
      lut={lut}
      caption={caption}
      aspect={aspect}
      readout={readout}
    />
  );
}
