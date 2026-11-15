(* Spatial-window arithmetic shared by windowed ops (conv/pool): the cascade
   repairs that keep a windowed config valid. A machinery helper, not a per-op
   walk description, so both backends' cascades may call it. The operands are
   the guarded geometry types ([Core.Geometry]) and extents, so a pad cannot be
   passed for a kernel. The int arithmetic stays inside this module: it is the
   window domain's own, bounded by the walk's [Limits]. *)

open Core

type extent = Dim.extent Dim.t

(* Smallest input >= [in_size] that keeps the output extent >= 1. Output >= 1
   iff in >= (kernel-1)*dilation + 1 - 2*pad, so grow up to that floor. *)
let grow_input ~(in_size : extent) ~(pad : Geometry.Nonneg.t) ~(kernel : extent)
    ~(dilation : Geometry.Pos.t) : extent =
  let floor =
    max 1 ((((kernel :> int) - 1) * (dilation :> int)) + 1 - (2 * (pad :> int)))
  in
  if floor > (in_size :> int) then Dim.extent floor else in_size

(* Pooling requires 2*pad <= kernel; clamp pad down to satisfy it. *)
let clamp_pad_pool ~(pad : Geometry.Nonneg.t) ~(kernel : extent) :
    Geometry.Nonneg.t =
  Geometry.Nonneg.of_int (min (pad :> int) ((kernel :> int) / 2))

(* Round [n] up to the nearest multiple of [m]. Used to keep conv channel
   counts divisible by groups. *)
let round_up_multiple ~(n : extent) ~(m : extent) : extent =
  let m = (m :> int) in
  Dim.extent (((n :> int) + m - 1) / m * m)
