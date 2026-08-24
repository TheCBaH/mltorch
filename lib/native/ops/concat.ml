(* `aten.cat.default`'s Native form: join a variadic list of tensors along one
   named axis. The inverse data-movement shape of [Split.Unbind] (many inputs,
   one output, rather than one input many outputs) -- own file rather than
   [split.ml], whose header scopes that module to operations that DIVIDE one
   tensor. `aten.stack.default` (see [Stack] below) reuses this algorithm
   over each operand's unsqueezed shape rather than naming a second variadic
   one here. See .ai/pt2_model_support.md.

   [Concat] is the first op whose OPERAND count, not just its output count, is
   variable -- [Graph_ir.OP]'s [operands]/[map_operands] already generalise to
   a list (see [operands]/[map_operands] below), so nothing upstream needed to
   change to admit it. *)

module Concat = struct
  type params = { axis : Axis.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"concat_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) = Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis.pp p.axis

  type t = { params : params; xs : Tensor_ref.t list }

  let name = "Concat"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          xs = get "xs" (Jsont.list Tensor_ref.jsont);
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ( "xs",
              Json_util.jarr (List.map (Json_util.enc Tensor_ref.jsont) t.xs) );
          ])
      Jsont.json

  let operands (t : t) = t.xs
  let map_operands f (t : t) = { t with xs = List.map f t.xs }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>concat@ xs=%a@ params=%a@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_ref))
      t.xs pp_params t.params

  (* Every axis but the concatenated one must agree across every operand
     (checked pairwise against the FIRST, which is transitive since equality
     is); the concatenated axis is the SUM of each operand's extent there,
     bounded the same way [Pad]'s output extent is -- a sum of
     model-supplied factors, and [Kernel.Limits.Hard.extent] is the
     JS-reachable-path ceiling CLAUDE.md's aggregate rule requires. Reuses
     [Shape_error.Window_over_limit]'s [`Output_extent] row for that overflow
     rather than a second one: the concatenated axis IS the output extent
     there. *)
  let output_shape ~(xs_shapes : Vec6.shape list) (p : params) =
    let open Err.Syntax in
    match xs_shapes with
    | [] -> Err.fail (`Concat Shape_error.Concat.Empty)
    | first :: rest ->
        let* () =
          Err.List.iter
            (fun sh ->
              Err.List.iter
                (fun axis ->
                  if Axis.equal axis p.axis then Err.return ()
                  else
                    let lhs = Vec6.get first axis in
                    let rhs = Vec6.get sh axis in
                    if Dim.equal lhs rhs then Err.return ()
                    else
                      Err.fail
                        (`Concat
                           (Shape_error.Concat.Axis_mismatch
                              { axis; first = lhs; other = rhs })))
                Axis.all)
            rest
        in
        let* total =
          Err.List.fold_left
            (fun acc (sh : Vec6.shape) ->
              let e = Vec6.get sh p.axis in
              let sum = Int64.add acc (Int64.of_int (e :> int)) in
              if Int64.compare sum Kernel.Limits.Hard.extent >= 0 then
                Err.fail
                  (`Window_over_limit
                     Shape_error.Window_over_limit.
                       {
                         what = `Output_extent;
                         value = sum;
                         limit = Kernel.Limits.Hard.extent;
                       })
              else Err.return sum)
            0L xs_shapes
        in
        Err.return (Vec6.set first p.axis (Dim.extent (Int64.to_int total)))

  module Compute (S : Semantics.SEMANTICS) = struct
    (* Clamp a delta index into [0, extent-1] -- the same two-line arithmetic
       [Pad.Compute.clamp_into] uses. No shared primitive exists for a helper
       this small, used by exactly these two ops. *)
    let clamp_into ~(extent : Dim.extent Dim.t) j =
      S.clamp_low (S.index_min (S.index_const ((extent :> int) - 1)) j)

    (* Which operand [out]'s concat-axis coordinate falls in, via the same
       clamp-and-compare idiom [Pad]'s in/out-of-source test uses -- there is
       no index COMPARISON primitive, deliberately (semantics.mli) -- applied
       once per operand instead of once for a single boundary test. The LAST
       segment needs no test: [output_shape] already proved every valid
       coordinate lands in exactly one segment, so it is the right answer
       whenever every earlier test failed, and skipping it also keeps the
       staged Symbolic term one [select] shorter per operand. *)
    let rec select_segment axis out = function
      | [] -> invalid_arg "Concat.Compute.pixel: no operands"
      | [ (offset, extent, input) ] ->
          let j =
            S.index_add
              (S.of_index (Vec6.get out axis))
              (S.index_const (-offset))
          in
          let src = clamp_into ~extent j in
          S.load input (Vec6.set out axis src)
      | (offset, extent, input) :: rest ->
          let j =
            S.index_add
              (S.of_index (Vec6.get out axis))
              (S.index_const (-offset))
          in
          let src = clamp_into ~extent j in
          let inside = S.index_eq j (S.of_index src) in
          S.select inside
            (S.load input (Vec6.set out axis src))
            (select_segment axis out rest)

    let pixel (p : params) ~(xs : (Vec6.shape * S.input) list)
        (out : Semantics.position S.index Vec6.t) : S.t =
      let axis = p.axis in
      let _, segments =
        List.fold_left
          (fun (offset, acc) ((shape, input) : Vec6.shape * S.input) ->
            let extent = Vec6.get shape axis in
            (offset + (extent :> int), (offset, extent, input) :: acc))
          (0, []) xs
      in
      select_segment axis out (List.rev segments)
  end
end

(* `stack.default(Tensor[] tensors, int dim=0) -> Tensor`: inserts a NEW axis
   at [dim] and joins the operands along it, unlike [Concat] which joins along
   an axis every operand already has. Design-goal fix (see
   .ai/native_add_op.md): this used to lower to one [Reshape] per operand (an
   [Unsqueeze]) followed by a separate [Concat] node, so no node in the graph
   named [stack.default]. Fixed by reusing [Concat]'s implementation — its
   [output_shape] and [Compute.pixel] — rather than its node, after deriving
   each operand's own unsqueezed shape, exactly as the bridge used to spell it
   out across N+1 nodes but now owned by one.

   [axis] here is the OUTPUT axis (rank+1-valued, ATen's [dim] normalized
   against rank+1 positions the same way [Op_bridge.norm_unsqueeze_dim]
   normalizes [unsqueeze.default]'s), not an axis any operand's OWN shape has
   — that is the one respect in which this payload cannot just be
   [Concat.params]. *)
module Stack = struct
  type params = { axis : Axis.t }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"stack_params" (fun axis -> { axis })
    |> Jsont.Object.mem "axis" Axis.jsont ~enc:(fun p -> p.axis)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) = Fmt.pf fmt "@[<hv>{axis=%a}@]" Axis.pp p.axis

  type t = { params : params; xs : Tensor_ref.t list }

  let name = "Stack"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          xs = get "xs" (Jsont.list Tensor_ref.jsont);
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ( "xs",
              Json_util.jarr (List.map (Json_util.enc Tensor_ref.jsont) t.xs) );
          ])
      Jsont.json

  let operands (t : t) = t.xs
  let map_operands f (t : t) = { t with xs = List.map f t.xs }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>stack@ xs=%a@ params=%a@]"
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_ref))
      t.xs pp_params t.params

  (* Each operand's shape with a size-1 axis inserted at [p.axis] — the same
     no-op reshape [Unsqueeze.default]'s legalization computes, applied here to
     a [Vec6.shape] directly via [Aten_shape.repack_dropped] rather than round
     tripping through an ATen-level int array (there is no ATen array at this
     layer, only the shapes [Graph_shape]/a JSON-decoded graph already carry).

     THE DIRECTION IS THE OPPOSITE OF [Select]'s: [Select.output_shape] maps a
     FULL shape's axis [kin] onto the narrower result's [oax] (dropping an
     axis moves data inward); inserting one does the reverse, so the result's
     [kin] takes its value from the operand's [oax]. Getting this backwards
     does not fail loudly — both directions produce A shape, of the right
     rank — it silently mislabels which axis carries the operand's real
     extent whenever [dim] is not the outermost insertion position (e.g.
     `stack(dim=1)` on rank-2 operands moves the real extent from native W
     onto native H rather than leaving it on W), so verify against
     [Op_bridge]'s ATen-list splice (`front @ (1 :: back)`) on a case where
     [dim] is NOT 0, not just the boundary case where both directions agree. *)
  let unsqueezed_shape ~(x_shape : Vec6.shape) (p : params) =
    let ones = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
    List.fold_left
      (fun s (kin, oax) -> Vec6.copy x_shape ~src:oax ~dst:kin s)
      ones
      (Aten_shape.repack_dropped ~dropped:[ p.axis ])

  let output_shape ~(xs_shapes : Vec6.shape list) (p : params) =
    Concat.output_shape
      ~xs_shapes:(List.map (fun sh -> unsqueezed_shape ~x_shape:sh p) xs_shapes)
      { Concat.axis = p.axis }

  module Compute (S : Semantics.SEMANTICS) = struct
    (* NOT a literal call into [Concat.Compute.pixel]: that reads every
       non-concatenated axis straight off the coordinate it is given, which is
       exactly wrong here whenever [dim] relabels a real axis (see
       [unsqueezed_shape]'s comment) — [out]'s value at, say, native H is the
       operand's real row index, but the RAW operand tensor's own storage
       still has that same row index addressed under native W, so reading it
       under H would silently read the wrong (always extent-1) slot. Folding
       [out] through the SAME [repack_dropped] pairs [Select.Compute] uses —
       direction reversed, matching [unsqueezed_shape] above — recovers the
       coordinate each operand's OWN pre-insertion storage needs; unlike
       [Select]'s [base], [p.axis] is never a fold SOURCE ([kin] ranges over
       [Axis.all] minus [p.axis]), so no operand's real data is ever read off
       it here, regardless of what [p.axis] happens to alias in some other
       operand's own frame. *)
    let source_coord (p : params) (out : Semantics.position S.index Vec6.t) =
      let zero =
        Vec6.make ~n:S.index_zero ~t:S.index_zero ~d:S.index_zero
          ~h:S.index_zero ~w:S.index_zero ~c:S.index_zero
      in
      List.fold_left
        (fun v (kin, oax) -> Vec6.copy out ~src:kin ~dst:oax v)
        zero
        (Aten_shape.repack_dropped ~dropped:[ p.axis ])

    (* What DOES carry over from [Concat.Compute.select_segment] is the
       clamp-and-compare "which segment" idiom, degenerated to the ONE-WIDE
       case every operand is here (its unsqueezed extent along [p.axis] is
       always 1, so "which operand" IS [out]'s coordinate there): an equality
       test rather than [Concat]'s general clamped range, since there is no
       within-segment offset left to compute. *)
    let rec select_by_index (p : params) coord out k = function
      | [] -> invalid_arg "Stack.Compute.pixel: no operands"
      | [ input ] -> S.load input coord
      | input :: rest ->
          let inside =
            S.index_eq (S.of_index (Vec6.get out p.axis)) (S.index_const k)
          in
          S.select inside (S.load input coord)
            (select_by_index p coord out (k + 1) rest)

    let pixel (p : params) ~(xs : S.input list)
        (out : Semantics.position S.index Vec6.t) : S.t =
      select_by_index p (source_coord p out) out 0 xs
  end
end
