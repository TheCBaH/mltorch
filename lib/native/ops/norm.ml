(* Normalisations over a set of axes (NHWC). [BatchNorm]/[BatchNormNoStats]
   (inference and no-running-stats), [RmsNorm] (ATen's `aten.rms_norm`) and
   [LayerNorm] (ATen's `aten.layer_norm` / `aten.native_layer_norm`); a
   category file, the way [Reduce] holds the reductions.

   This file is now a thin facade: the shared [Target] vocabulary and the
   NORMALIZED-axes layout/count helpers live in norm_shared.ml, [BatchNorm]
   and [BatchNormNoStats] in norm_batch.ml, [RmsNorm] in norm_rms.ml,
   [LayerNorm] in norm_layer.ml and [GroupNorm] in norm_group.ml. Every alias
   below is exactly what was a top-level binding in this file before the
   split, so external references such as [Norm.BatchNorm] or
   [Norm.normalized_shape] are unaffected.. *)

module Target = Norm_shared.Target

let normalized_shape = Norm_shared.normalized_shape
let normalized_count = Norm_shared.normalized_count
let normalized_count_unchecked = Norm_shared.normalized_count_unchecked

module BatchNorm = Norm_batch.BatchNorm
module BatchNormNoStats = Norm_batch.BatchNormNoStats
module RmsNorm = Norm_rms.RmsNorm
module LayerNorm = Norm_layer.LayerNorm
module GroupNorm = Norm_group.GroupNorm
