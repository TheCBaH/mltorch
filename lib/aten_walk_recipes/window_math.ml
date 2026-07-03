(* Spatial-window arithmetic shared by the conv and pool recipes: the output
   extent formula and the cascade repairs that keep a windowed config valid. *)

(* Output extent of one spatial axis, matching Window_axis.output_extent:
     (in + 2*pad - ((kernel-1)*dilation + 1)) / stride + 1. *)
let output_dim ~in_size ~pad ~kernel ~dilation ~stride =
  ((in_size + (2 * pad) - ((kernel - 1) * dilation) - 1) / stride) + 1

(* Smallest input >= [in_size] that keeps the output extent >= 1. Output >= 1
   iff in >= (kernel-1)*dilation + 1 - 2*pad, so grow up to that floor. *)
let grow_input ~in_size ~pad ~kernel ~dilation =
  let floor = max 1 (((kernel - 1) * dilation) + 1 - (2 * pad)) in
  max in_size floor

(* Pooling requires 2*pad <= kernel; clamp pad down to satisfy it. *)
let clamp_pad_pool ~pad ~kernel = min pad (kernel / 2)

(* Round [n] up to the nearest positive multiple of [m]. Used to keep conv
   channel counts divisible by groups. *)
let round_up_multiple ~n ~m =
  let m = max 1 m in
  let r = (n + m - 1) / m * m in
  if r < m then m else r
