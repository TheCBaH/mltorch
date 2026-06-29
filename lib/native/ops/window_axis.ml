(* Shared windowed-axis arithmetic for conv, pool: 0 <= out*stride + k - pad < in_extent.
   [output_extent] asks "how many output positions"; [Compute.window] asks "for this
   output position, which k values and where do they read." Two closed forms over the
   same four numbers, co-located here instead of duplicated per op.
   See .ai/native_compute_design.md §2b. *)

let output_extent ~(kernel : Dim.extent Dim.t) ~(stride : Op_config.Pos.t)
    ~(pad_before : Op_config.Nonneg.t) ~(pad_after : Op_config.Nonneg.t)
    ~(dilation : Op_config.Pos.t) ~(in_extent : Dim.extent Dim.t) :
    Dim.extent Dim.t =
  let effective_kernel = (((kernel :> int) - 1) * (dilation :> int)) + 1 in
  Dim.extent
    (((in_extent :> int)
     + (pad_before :> int)
     + (pad_after :> int)
     - effective_kernel)
     / (stride :> int)
    + 1)

module Compute (S : sig
  type 'role index

  val index_extent : Dim.extent Dim.t -> Semantics.delta index
  val index_const : int -> Semantics.delta index
  val of_index : Semantics.position index -> Semantics.delta index

  val index_add :
    Semantics.delta index -> Semantics.delta index -> Semantics.delta index

  val index_scale : int -> Semantics.delta index -> Semantics.delta index

  val index_floor_div_pos :
    Semantics.delta index -> Op_config.Pos.t -> Semantics.delta index

  val index_ceil_div_pos :
    Semantics.delta index -> Op_config.Pos.t -> Semantics.delta index

  val index_min :
    Semantics.delta index -> Semantics.delta index -> Semantics.delta index

  val clamp_low : Semantics.delta index -> Semantics.position index
  val assume_index : Semantics.delta index -> Semantics.position index
end) =
struct
  (* Valid kernel-offset range [lo, hi) for one output position, plus [src k]:
     the source position for offset k. Callers iterate k in [lo,hi) and read
     at [src k] — always a valid index by construction (the clip above), so
     max-pooling's [max_reduce] never silently reads 0 from the pad region.
     [lo]/[src k] are [position]; [hi] is [delta] (an upper bound, never read). *)
  type window = {
    lo : Semantics.position S.index;
    hi : Semantics.delta S.index;
    src : Semantics.position S.index -> Semantics.position S.index;
  }

  let window ~(kernel : Dim.extent Dim.t) ~(stride : Op_config.Pos.t)
      ~(pad_before : Op_config.Nonneg.t) ~(dilation : Op_config.Pos.t)
      ~(in_extent : Dim.extent Dim.t) (out_axis : Semantics.position S.index) :
      window =
    let stride = (stride :> int) and pad_before = (pad_before :> int) in
    let out = S.of_index out_axis in
    let start =
      S.index_add (S.index_scale stride out) (S.index_const (-pad_before))
    in
    let neg_start = S.index_scale (-1) start in
    let lo = S.clamp_low (S.index_ceil_div_pos neg_start dilation) in
    let hi =
      S.index_min (S.index_extent kernel)
        (S.index_add
           (S.index_floor_div_pos
              (S.index_add
                 (S.index_add (S.index_extent in_extent) (S.index_const (-1)))
                 neg_start)
              dilation)
           (S.index_const 1))
    in
    (* [assume_index]: for k in [lo,hi),
       stride*out + k*dilation - pad_before is in [0,in_extent). *)
    let src k =
      S.assume_index
        (S.index_add start (S.index_scale (dilation :> int) (S.of_index k)))
    in
    { lo; hi; src }
end
