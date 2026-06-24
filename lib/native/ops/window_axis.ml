(* Shared windowed-axis arithmetic for conv, pool: 0 <= out*stride + k - pad < in_extent.
   [output_extent] asks "how many output positions"; [Compute.window] asks "for this
   output position, which k values and where do they read." Two closed forms over the
   same four numbers, co-located here instead of duplicated per op.
   See .ai/native_compute_design.md §2b. *)

let output_extent ~(kernel : Dim.extent Dim.t) ~(stride : Op_config.Pos.t)
    ~(pad : Op_config.Nonneg.t) ~(in_extent : Dim.extent Dim.t) :
    Dim.extent Dim.t =
  Dim.extent
    (((in_extent :> int) + (2 * (pad :> int)) - (kernel :> int))
     / (stride :> int)
    + 1)

module Compute (S : sig
  type 'role ix

  val iext : Dim.extent Dim.t -> Semantics.delta ix
  val iconst : int -> Semantics.delta ix
  val of_index : Semantics.index ix -> Semantics.delta ix
  val iadd : Semantics.delta ix -> Semantics.delta ix -> Semantics.delta ix
  val iscale : int -> Semantics.delta ix -> Semantics.delta ix
  val imin : Semantics.delta ix -> Semantics.delta ix -> Semantics.delta ix
  val clamp_low : Semantics.delta ix -> Semantics.index ix
  val assume_index : Semantics.delta ix -> Semantics.index ix
end) =
struct
  (* Valid kernel-offset range [lo, hi) for one output position, plus [src k]:
     the source position for offset k. Callers iterate k in [lo,hi) and read
     at [src k] — always a valid index by construction (the clip above), so
     max-pooling's [maxr] never silently reads 0 from the pad region.
     [lo]/[src k] are [index]; [hi] is [delta] (an upper bound, never read). *)
  type window = {
    lo : Semantics.index S.ix;
    hi : Semantics.delta S.ix;
    src : Semantics.index S.ix -> Semantics.index S.ix;
  }

  let window ~(kernel : Dim.extent Dim.t) ~(stride : Op_config.Pos.t)
      ~(pad : Op_config.Nonneg.t) ~(in_extent : Dim.extent Dim.t)
      (out_axis : Semantics.index S.ix) : window =
    let stride = (stride :> int) and pad = (pad :> int) in
    let out = S.of_index out_axis in
    (* base = pad - stride*out; may be negative, hence [delta] and [clamp_low] *)
    let base = S.iadd (S.iconst pad) (S.iscale (-stride) out) in
    let lo = S.clamp_low base in
    let hi = S.imin (S.iext kernel) (S.iadd (S.iext in_extent) base) in
    (* [assume_index]: for k in [lo,hi), stride*out + k - pad is in [0,in_extent) *)
    let src k =
      S.assume_index
        (S.iadd (S.iscale stride out) (S.iadd (S.of_index k) (S.iconst (-pad))))
    in
    { lo; hi; src }
end
