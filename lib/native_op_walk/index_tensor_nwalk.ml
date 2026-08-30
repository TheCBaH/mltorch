(* Assembles Index_tensor's walk: an ad hoc (n, h, w, c, axis, l) config
   space, mirroring [Bmm_nwalk]'s style since [Index_tensor] has no shared
   [Walk] functor in lib/native/ops/index_tensor.ml. [self]'s shape varies on
   the four mutable axes [Unbind_nwalk] also uses (N/H/W/C); [axis] picks
   which one is gathered, and [l] is [index]'s own length (the C-axis extent
   of its rank-1-scoped shape, per round 9 -- see .ai/index_tensor_design.md).

   [index]'s values are drawn as random INT64s in [0, extent) at [axis],
   never negative: this walk exists to prove Direct/Symbolic AGREE (they
   share [Compute.pixel]), not to re-prove ATen's negative-index
   normalization, which the hand-derived fixtures in
   test/native/load_index_test.ml and index_tensor_test.ml already pin. *)

module M = struct
  type cfg = { n : int; h : int; w : int; c : int; axis : Axis.t; l : int }
  type subject = Native_subject.t

  let target = "index_tensor"
  let initial = { n = 1; h = 2; w = 3; c = 4; axis = Axis.W; l = 2 }
  let cascade c = c

  let axes =
    Walk_core.Walk.
      [
        int_axis "n" ~lo:1 ~hi:3 (fun (c : cfg) v -> { c with n = v });
        int_axis "h" ~lo:1 ~hi:4 (fun (c : cfg) v -> { c with h = v });
        int_axis "w" ~lo:1 ~hi:4 (fun (c : cfg) v -> { c with w = v });
        int_axis "c" ~lo:1 ~hi:4 (fun (c : cfg) v -> { c with c = v });
        field_axis "axis"
          Axis.[ N; H; W; C ]
          (fun (c : cfg) v -> { c with axis = v });
        int_axis "l" ~lo:1 ~hi:4 (fun (c : cfg) v -> { c with l = v });
      ]

  let pp fmt c =
    Format.fprintf fmt "{n=%d h=%d w=%d c=%d axis=%a l=%d}" c.n c.h c.w c.c
      Axis.pp c.axis c.l

  let self_shape c = Vec6.shape ~n:c.n ~t:1 ~d:1 ~h:c.h ~w:c.w ~c:c.c
  let index_shape c = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:c.l

  let self_extent c =
    match c.axis with
    | Axis.N -> c.n
    | Axis.H -> c.h
    | Axis.W -> c.w
    | Axis.C -> c.c
    | Axis.T | Axis.D -> assert false (* [axes]' field_axis never draws these *)

  (* [Native_tensor.synth]'s F32 uniform draw doesn't fit here: [index] must
     be I64, and its values must be valid positions for [self]'s own gathered
     axis, not arbitrary floats. *)
  let synth_index pcg ~extent shape =
    let numel = (Vec6.numel shape :> int) in
    let rec draw k pcg acc =
      if k = 0 then (acc, pcg)
      else
        let v, pcg =
          Walk_core.Pcg.uniform ~low:0. ~high:(float_of_int extent) pcg
        in
        let i = min (extent - 1) (max 0 (int_of_float v)) in
        draw (k - 1) pcg (Int64.of_int i :: acc)
    in
    let vals, pcg = draw numel pcg [] in
    let arr = Array.of_list vals in
    let i = ref 0 in
    let t =
      Tensor.materialize_i64 shape (fun _ ->
          let v = arr.(!i) in
          incr i;
          v)
    in
    (t, pcg)

  let build pcg c =
    let self_v, pcg = Native_tensor.synth pcg (self_shape c) in
    let index_v, pcg =
      synth_index pcg ~extent:(self_extent c) (index_shape c)
    in
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"index_tensor" ~outputs:(fun r -> [ r ])
          @@
          let* self = input ~shape:(self_shape c) ~name:"self" () in
          let* index =
            input ~shape:(index_shape c) ~fmt:(Payload.Fmt Payload.I64)
              ~name:"index" ()
          in
          index_tensor ~name:"out"
            { Index_tensor.Index_tensor.axis = c.axis }
            ~self ~index)
    in
    let inputs = List.combine g.Graph_ir.Graph.inputs [ self_v; index_v ] in
    ({ Native_subject.target; graph = g; inputs }, pcg)
end
