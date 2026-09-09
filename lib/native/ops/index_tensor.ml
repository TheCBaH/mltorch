(* `index.Tensor(Tensor self, Tensor?[] indices) -> Tensor`: a genuine runtime
   gather, restricted to the evidenced shape family (`.ai/index_tensor_design.md`,
   `.ai/index_tensor_impl.md`) -- an optional-tensor index list with exactly
   one live entry. That entry becomes [index] here; [axis] is the frame
   position its list index [p] maps to (computed by the importer from ATen's
   dim position and [self]'s real rank, via the shared `dims_arg`/
   `axes_for_rank` machinery -- see op_bridge_shape.ml/
   native_interp_lower_shape.ml).

   [index_rank] (the live entry's own ATen rank, read once at import time,
   before the six-axis frame erases it) generalizes the original rank-1-only
   scope: real ATen's own rule is
   [output = self.shape[:axis] ++ index.shape ++ self.shape[axis+1:]], so a
   rank-1 index (the original scope) simply REPLACES [self]'s own axis at
   [axis] with index's single value, identity-preserving on [self]'s rank;
   a rank-M index for M > 1 INSERTS M-1 extra axes there instead, borrowing
   frame room from the axes strictly left of [axis] -- valid only when
   [self] has no real content there to lose (checked below), the same
   "restricted to the one evidenced shape family" discipline the original
   landing applied to index's own rank. *)

module Index_tensor = struct
  type params = { axis : Axis.t; index_rank : int }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"index_tensor_params" (fun axis index_rank ->
        { axis; index_rank })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "index_rank" Jsont.int ~enc:(fun p -> p.index_rank)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a index_rank=%d}@]" Axis.pp p.axis p.index_rank

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

  (* The window of [index_rank] frame axes ending at [axis] (counting
     backward: [axis], [axis]-1, ... [axis]-[index_rank]+1) -- where index's
     own [index_rank] real axes land in OUTPUT/self frame terms, the same
     right-alignment convention [Aten_shape.used_axes] states for a tensor's
     own natural frame, here re-anchored to end at [axis] instead of [C].
     [None] if [index_rank] doesn't fit (more axes than [axis] has room for
     on its left, i.e. would need to reach past frame axis [N]). *)
  let window ~axis ~index_rank =
    let ai = Axis.to_int axis in
    if index_rank > ai + 1 then None
    else
      Some (List.filteri (fun i _ -> i > ai - index_rank && i <= ai) Axis.all)

  (* [index]'s own [index_rank] real frame axes, right-aligned ending at [C]
     -- [Aten_shape.used_axes] states this convention for every tensor's own
     natural frame; paired positionally with [window] above (both length
     [index_rank], both in canonical N/T/D/H/W/C order) to relate index's
     own axis to the OUTPUT/self axis carrying the same logical dimension. *)
  let index_axes ~index_rank = Aten_shape.used_axes ~rank:index_rank

  (* [index_shape]'s own extents outside [index_axes ~index_rank] must be
     exactly 1 -- [index_rank] is a compile-time claim about [index]'s real
     ATen rank (read once at import time), and this proves the claim
     against [index]'s own shape rather than trusting it, the same defense
     round 12's original rank-1 check gave the rank-1 case (a rank claim
     paired with a mismatched tensor would otherwise silently drop or
     misread whichever of [index]'s own axes the mismatch hides). *)
  let check_index_shape ~index_rank (index_shape : Vec6.shape) =
    let real = index_axes ~index_rank in
    match
      List.find_opt
        (fun a ->
          (not (List.mem a real))
          && not (Dim.equal (Vec6.get index_shape a) Dim.one))
        Axis.all
    with
    | Some axis ->
        Err.fail
          (`Index_tensor
             Shape_error.Index_tensor.(
               Index_shape_mismatch
                 { index_rank; axis; extent = Vec6.get index_shape axis }))
    | None -> Err.return ()

  (* [self_shape]'s own extents strictly left of [axis] must be exactly 1
     whenever [index_rank > 1]: those frame axes are borrowed to hold
     index's own extra axes, so any real (non-unit) content [self] has
     there would be silently discarded rather than represented anywhere in
     the output -- true ATen never loses data this way ([self.shape[:axis]]
     is always preserved), so this restricts to the shape family where
     there is nothing to lose. For [index_rank = 1] this window is empty by
     construction ([window]'s span is just [axis] itself), so no check is
     needed and [self]'s own leading content (real or not) simply passes
     through unchanged -- the original rank-1 landing's exact behavior. *)
  let check_no_collision ~axis ~index_rank (self_shape : Vec6.shape) =
    match window ~axis ~index_rank with
    | None ->
        Err.fail
          (`Index_tensor
             (Shape_error.Index_tensor.Rank_overflow { axis; index_rank }))
    | Some win -> (
        let colliding =
          List.find_opt
            (fun a ->
              (not (Axis.equal a axis))
              && not (Dim.equal (Vec6.get self_shape a) Dim.one))
            win
        in
        match colliding with
        | Some colliding_axis ->
            Err.fail
              (`Index_tensor
                 Shape_error.Index_tensor.(
                   Self_collision
                     {
                       axis;
                       colliding_axis;
                       extent = Vec6.get self_shape colliding_axis;
                     }))
        | None -> Err.return ())

  (* Builds the output shape axis by axis: index's own extent -- read from
     its natural, [C]-ending frame position -- at every frame axis in
     [window], [self_shape]'s own extent everywhere else ([axis]'s own
     trailing part unchanged, and [check_no_collision] has already proved
     [self_shape] is 1 at every OTHER [window] axis, so copying it there
     unconditionally would also be correct -- reading index's own extent
     instead is what actually replaces those axes). *)
  let output_shape ~(self_shape : Vec6.shape) ~(index_shape : Vec6.shape)
      (p : params) =
    let open Err.Syntax in
    let* () = check_index_shape ~index_rank:p.index_rank index_shape in
    let* () =
      check_no_collision ~axis:p.axis ~index_rank:p.index_rank self_shape
    in
    let win = Option.get (window ~axis:p.axis ~index_rank:p.index_rank) in
    let pairs = List.combine win (index_axes ~index_rank:p.index_rank) in
    Err.return
      (Vec6.of_fn (fun a ->
           match List.assoc_opt a pairs with
           | Some index_axis -> Vec6.get index_shape index_axis
           | None -> Vec6.get self_shape a))

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [index]'s own read coordinate: zero everywhere except [index_axes]'s
       [index_rank] real positions, each read from [out]'s corresponding
       [window] position (the same [(window, index_axes)] positional
       pairing [output_shape] uses). [self]'s read coordinate is [out] with
       [axis] itself set to the resolved gather position, every OTHER
       [window] axis zeroed (nothing real for self there -- borrowed room
       for index's own extra axes, which [check_no_collision] proved is
       always extent-1 on self), and every axis outside [window] passing
       through from [out] unchanged -- [self]'s own real content, whether
       left or right of [axis]. *)
    let pixel (p : params) ~(self_shape : Vec6.shape) ~self ~index
        (out : Semantics.position S.index Vec6.t) =
      let win = Option.get (window ~axis:p.axis ~index_rank:p.index_rank) in
      let pairs = List.combine win (index_axes ~index_rank:p.index_rank) in
      let index_coord =
        List.fold_left
          (fun acc (out_axis, index_axis) ->
            Vec6.set acc index_axis (Vec6.get out out_axis))
          (Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
             ~h:S.index_zero ~w:S.index_zero ~c:S.index_zero)
          pairs
      in
      let extent = Vec6.get self_shape p.axis in
      let position = S.load_index index index_coord ~extent in
      let self_coord =
        Vec6.of_fn (fun a ->
            if Axis.equal a p.axis then position
            else if List.mem a win then S.index_zero
            else Vec6.get out a)
      in
      S.load self self_coord
  end
end
