(* `index.Tensor(Tensor self, Tensor?[] indices) -> Tensor`: a genuine runtime
   gather, restricted to the one evidenced shape family (`.ai/index_tensor_design.md`,
   `.ai/index_tensor_impl.md`) -- an optional-tensor index list with exactly
   one live entry, itself of ATen rank 1. That entry becomes [index] here;
   [axis] is the frame position its list index [p] maps to (computed by the
   importer from ATen's dim position and [self]'s real rank, via the shared
   `dims_arg`/`axes_for_rank` machinery -- see op_bridge_shape.ml/
   native_interp_lower_shape.ml). A rank-1 index insertion is
   identity-preserving on [self]'s own rank (it replaces exactly one axis's
   extent with [index]'s own length and touches nothing else), which is what
   lets this op need no rank fields at all: `output_rank = self_rank`. *)

module Index_tensor = struct
  type params = { axis : Axis.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"index_tensor_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) = Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis.pp p.axis

  type t = { params : params; self : Tensor_ref.t; index : Tensor_ref.t }

  let name = "IndexTensor"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          self = get "self" Tensor_ref.jsont;
          index = get "index" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("self", Json_util.enc Tensor_ref.jsont t.self);
            ("index", Json_util.enc Tensor_ref.jsont t.index);
          ])
      Jsont.json

  let operands (t : t) = [ t.self; t.index ]
  let map_operands f (t : t) = { t with self = f t.self; index = f t.index }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>index_tensor@ self=%a@ index=%a@ params=%a@]" pp_ref
      t.self pp_ref t.index pp_params t.params

  (* Only [self_shape]'s [axis] extent changes, replaced by [index_shape]'s
     own length (always read at frame axis [C] -- a rank-1 tensor's real
     content is right-aligned there, the same assumption [Bmm]'s own
     contraction-axis reads already make).

     [output_shape] cannot recover [index]'s true ATen rank (erased by the
     six-axis frame), but it CAN observe extents, and that is enough to
     enforce the rank-1 restriction at this boundary too (round 12): reachable
     from a direct [Graph_builder] call or a JSON-decoded graph, not only from
     importer validation, so it rejects unless every index frame axis other
     than [C] has extent exactly 1. A true rank-1 [\[L\]] and a rank-2
     [\[1, L\]] are behaviorally identical for this restricted op, so refusing
     only a genuinely non-unit leading axis is not a gap. *)
  let output_shape ~(self_shape : Vec6.shape) ~(index_shape : Vec6.shape)
      (p : params) =
    let non_unit_leading =
      List.find_opt
        (fun a ->
          (not (Axis.equal a Axis.C))
          && not (Dim.equal (Vec6.get index_shape a) Dim.one))
        Axis.all
    in
    match non_unit_leading with
    | Some axis ->
        Err.fail
          (`Index_tensor
             Shape_error.Index_tensor.
               { axis; extent = Vec6.get index_shape axis })
    | None ->
        Err.return (Vec6.copy index_shape ~src:Axis.C ~dst:p.axis self_shape)

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [index]'s own read coordinate is built EXPLICITLY, never inherited from
       [out]: after the rank-1 scoping, [index]'s only real dimension sits at
       frame axis [C], so it must be read at ([out]'s value on the gathered
       axis, 0, 0, 0, 0, 0) -- not at [out] itself, which reads the wrong axis
       whenever [axis <> C] and is out-of-bounds on [index]'s own (every-other-
       axis-extent-1) shape otherwise. [self]'s read coordinate is [out] with
       [axis] replaced by the resolved gather position; every other axis of
       [out] passes through unchanged. *)
    let pixel (p : params) ~(self_shape : Vec6.shape) ~self ~index
        (out : Semantics.position S.index Vec6.t) =
      let gathered = Vec6.get out p.axis in
      let zero =
        Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
          ~h:S.index_zero ~w:S.index_zero ~c:S.index_zero
      in
      let index_coord = Vec6.set zero Axis.C gathered in
      let extent = Vec6.get self_shape p.axis in
      let position = S.load_index index index_coord ~extent in
      S.load self (Vec6.set out p.axis position)
  end
end
