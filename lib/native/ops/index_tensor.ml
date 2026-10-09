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
  type params = { axis : Axis.t; index_rank : Rank.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"index_tensor_params" (fun axis index_rank ->
        { axis; index_rank })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.mem "index_rank" Rank.jsont ~enc:(fun p -> p.index_rank)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{axis=%a index_rank=%a}@]" Axis.pp p.axis Rank.pp
      p.index_rank

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
    let index_rank = (index_rank : Rank.t :> int) in
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

(* `index.Tensor(Tensor self, Tensor?[] indices)` with exactly two live,
   leading indices: [self[i0, i1]], the advanced-indexing form attention masks
   (`mask[batch_idx, kv_idx]`) and last-token pooling (`h[arange, argmax]`) use.
   ATen's rule: the two index tensors broadcast to a shape B, and the result is
   [B ++ self.shape[2:]].

   Frame layout. Ranks are ATen ranks, which the right-aligned frame erases, so
   [params] carries all three. With R = [self_rank] and t = R - 2 trailing self
   axes, B has rank rb = max of the index ranks and the result has rank
   rb + t. Right-alignment makes three things line up on one fact -- the
   trailing t axes of self and of the result are the same frame axes, and an
   index axis [a] sits [t] axes to the RIGHT of the result axis for the same
   logical dimension -- so the index is read at the result coordinate shifted
   left by t, and self at (i0, i1) on its two leading real axes with the
   trailing axes taken from the result unchanged. *)
module Index_pair = struct
  type params = {
    self_rank : Rank.t;
    index0_rank : Rank.t;
    index1_rank : Rank.t;
  }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"index_pair_params"
      (fun self_rank index0_rank index1_rank ->
        { self_rank; index0_rank; index1_rank })
    |> Jsont.Object.mem "self_rank" Rank.jsont ~enc:(fun p -> p.self_rank)
    |> Jsont.Object.mem "index0_rank" Rank.jsont ~enc:(fun p -> p.index0_rank)
    |> Jsont.Object.mem "index1_rank" Rank.jsont ~enc:(fun p -> p.index1_rank)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{self_rank=%a index0_rank=%a index1_rank=%a}@]" Rank.pp
      p.self_rank Rank.pp p.index0_rank Rank.pp p.index1_rank

  type t = {
    params : params;
    self : Tensor_ref.t;
    index0 : Tensor_ref.t;
    index1 : Tensor_ref.t;
  }

  let name = "IndexPair"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          self = get "self" Tensor_ref.jsont;
          index0 = get "index0" Tensor_ref.jsont;
          index1 = get "index1" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("self", Json_util.enc Tensor_ref.jsont t.self);
            ("index0", Json_util.enc Tensor_ref.jsont t.index0);
            ("index1", Json_util.enc Tensor_ref.jsont t.index1);
          ])
      Jsont.json

  let operands (t : t) = [ t.self; t.index0; t.index1 ]

  let map_operands f (t : t) =
    { t with self = f t.self; index0 = f t.index0; index1 = f t.index1 }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>index_pair@ self=%a@ index0=%a@ index1=%a@ params=%a@]"
      pp_ref t.self pp_ref t.index0 pp_ref t.index1 pp_params t.params

  let axis_at k = List.nth Axis.all k
  let rank_int (r : Rank.t) = (r :> int)

  (* The shift between an index axis and the result axis for the same logical
     dimension, and the broadcast rank. *)
  let trailing (p : params) = rank_int p.self_rank - 2

  let broadcast_rank (p : params) =
    max (rank_int p.index0_rank) (rank_int p.index1_rank)

  (* Extents outside a tensor's own [rank] real axes must be 1: the rank is a
     claim made before the frame erased it, proved here against the shape. *)
  let check_extents ~rank (shape : Vec6.shape) ~fail =
    let real = Aten_shape.used_axes ~rank in
    match
      List.find_opt
        (fun a ->
          (not (List.mem a real)) && not (Dim.equal (Vec6.get shape a) Dim.one))
        Axis.all
    with
    | Some axis -> fail axis (Vec6.get shape axis)
    | None -> Err.return ()

  let output_shape ~(self_shape : Vec6.shape) ~(index0_shape : Vec6.shape)
      ~(index1_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let t = trailing p and rb = broadcast_rank p in
    let* () =
      if rank_int p.self_rank < 2 || rb + t > 6 then
        Err.fail
          (`Index_tensor
             (Shape_error.Index_tensor.Pair_rank
                { self_rank = p.self_rank; broadcast_rank = Rank.of_int rb }))
      else Err.return ()
    in
    let* () =
      check_extents ~rank:p.self_rank self_shape ~fail:(fun axis extent ->
          Err.fail
            (`Index_tensor
               (Shape_error.Index_tensor.Pair_self_mismatch
                  { self_rank = p.self_rank; axis; extent })))
    in
    let index_mismatch rank axis extent =
      Err.fail
        (`Index_tensor
           (Shape_error.Index_tensor.Index_shape_mismatch
              { index_rank = rank; axis; extent }))
    in
    let* () =
      check_extents ~rank:p.index0_rank index0_shape
        ~fail:(index_mismatch p.index0_rank)
    in
    let* () =
      check_extents ~rank:p.index1_rank index1_shape
        ~fail:(index_mismatch p.index1_rank)
    in
    let ro = rb + t in
    (* The B dimension at result frame axis [k] (k in [6 - ro, 6 - t)): each
       index contributes the extent at the axis [t] to the right, if it is one
       of its real axes. *)
    let b_extent k =
      let from shape rank =
        let a = k + t in
        if a >= 6 - rank_int rank then Vec6.get shape (axis_at a) else Dim.one
      in
      let e0 = from index0_shape p.index0_rank
      and e1 = from index1_shape p.index1_rank in
      if Dim.equal e0 e1 then Err.return e0
      else if Dim.equal e0 Dim.one then Err.return e1
      else if Dim.equal e1 Dim.one then Err.return e0
      else
        Err.fail
          (`Broadcast
             Shape_error.Broadcast.{ axis = axis_at k; lhs = e0; rhs = e1 })
    in
    Err.List.fold_left
      (fun s k ->
        let a = axis_at k in
        if k >= 6 - t then Err.return (Vec6.set s a (Vec6.get self_shape a))
        else if k >= 6 - ro then
          let+ e = b_extent k in
          Vec6.set s a e
        else Err.return s)
      (Vec6.of_fn (fun _ -> Dim.one))
      [ 0; 1; 2; 3; 4; 5 ]

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (p : params) ~(self_shape : Vec6.shape)
        ~(index0_shape : Vec6.shape) ~(index1_shape : Vec6.shape) ~self ~index0
        ~index1 (out : Semantics.position S.index Vec6.t) =
      let t = trailing p in
      let ax0 = axis_at (6 - rank_int p.self_rank) in
      let ax1 = axis_at (6 - rank_int p.self_rank + 1) in
      let index_coord shape =
        let shifted =
          Vec6.of_fn (fun a ->
              let k = Axis.to_int a - t in
              if k >= 0 then Vec6.get out (axis_at k) else S.index_zero)
        in
        Pointwise.broadcast_coord ~index_zero:S.index_zero shape shifted
      in
      let p0 =
        S.load_index index0 (index_coord index0_shape)
          ~extent:(Vec6.get self_shape ax0)
      in
      let p1 =
        S.load_index index1 (index_coord index1_shape)
          ~extent:(Vec6.get self_shape ax1)
      in
      let self_coord =
        Vec6.of_fn (fun a ->
            if Axis.equal a ax0 then p0
            else if Axis.equal a ax1 then p1
            else if Axis.to_int a >= 6 - t then Vec6.get out a
            else S.index_zero)
      in
      S.load self self_coord
  end
end
