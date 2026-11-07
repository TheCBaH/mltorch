(* Loop collapsing over dense rows: [for v1 in [0, n1): for v2 in [0, n2): B]
   becomes [for g in [0, n1 * n2): B'] when every access of [B] that names
   [v1] or [v2] reads or writes its buffer at [v1] on some axis [a] and [v2]
   on the next one, with [n2] that next axis's whole extent. Row-major, the
   element [(v1, v2)] of axes [(a, a + 1)] is the element [g = n2 * v1 + v2]
   of one axis of extent [E(a) * E(a + 1)], so each such buffer is declared
   with the two axes merged (extent 1 at [a]) and accessed at [(0, g)]: the
   same bytes, the same iteration order, the same statements in the same
   number. The vectorizer then sees one long contiguous loop instead of short
   rows.

   The SSA counterpart of [Loop_opt_collapse], which the planned pipelines do
   not see (they lower from the fusion plan). What keeps it exact:
   - The nest is perfect: the outer body is the inner loop and nothing else,
     both start at zero, step by one, have constant trips and carry only the
     effect, so every statement runs once per [(v1, v2)] in the same order.
   - [v1] and [v2] occur nowhere but as those two coordinates (no index
     arithmetic, guard, payload or bound reads them), and only in in-bounds
     loads and stores, which report no coordinate.
   - Every access of a merged buffer, anywhere in the program, is one of
     those: a buffer read elsewhere, by a checked load, a bounds check, a
     contraction or an attention, keeps its axes, and the pair is refused. So
     is a per-channel quantized buffer, whose channel axis is part of its
     meaning.
   - [n2] is the extent of axis [a + 1], so the rows are adjacent: a broadcast
     row ([v1] absent), a transposed pair or a partial row fail the shape
     and the pair stays nested. *)

module Id_map = Map.Make (Int)

let id (v : Ssa_value.t) = (v.Ssa_value.id :> int)
let bid (b : Ssa_id.Buffer.t) = (b :> int)

(* The access of a statement: its buffer, and its coordinate when it is one
   this pass may rewrite (an in-bounds load or a store at a coordinate). *)
type access = Rewritable of Ssa_value.t Expr.Coord.t | Other

let accesses (op : Ssa_op.t) =
  match op with
  | Ssa_op.Load_in_bounds { buffer; at = Ssa_access.Coord c; _ }
  | Ssa_op.Store { buffer; at = Ssa_access.Coord c; _ } ->
      [ (buffer, Rewritable c) ]
  | Ssa_op.Load { buffer; _ }
  | Ssa_op.Load_in_bounds { buffer; _ }
  | Ssa_op.Store { buffer; _ }
  | Ssa_op.Check_access { buffer; _ }
  | Ssa_op.Vec_load { buffer; _ }
  | Ssa_op.Vec_store { buffer; _ } ->
      [ (buffer, Other) ]
  | Ssa_op.Attend { q; k; v; mask; o; _ } ->
      List.map (fun b -> (b, Other)) ([ q; k; v; o ] @ Option.to_list mask)
  | Ssa_op.Contract { a; b; c; _ } -> [ (a, Other); (b, Other); (c, Other) ]
  | _ -> []

let rec fold_instrs f acc (r : Ssa_region.t) =
  List.fold_left
    (fun acc -> function
      | Ssa_stmt.Instr i -> f acc i
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          fold_instrs f acc body
      | Ssa_stmt.If { then_; else_; _ } ->
          fold_instrs f (fold_instrs f acc then_) else_)
    acc r.Ssa_region.body

(* How many accesses each buffer has in [r]. *)
let access_counts r =
  fold_instrs
    (fun m (i : Ssa_instr.t) ->
      List.fold_left
        (fun m (b, _) ->
          Id_map.update (bid b)
            (fun n -> Some (1 + Option.value n ~default:0))
            m)
        m (accesses i.Ssa_instr.op))
    Id_map.empty r

exception Refused

let axis_after a =
  let rec go = function
    | x :: (y :: _ as rest) -> if x = a then Some y else go rest
    | _ -> None
  in
  go Expr.Axis.all

(* The axis of [c] that holds [v], if exactly one does. *)
let axis_of c (v : Ssa_value.t) =
  match
    List.filter (fun a -> Ssa_value.equal (Expr.Coord.get c a) v) Expr.Axis.all
  with
  | [ a ] -> Some a
  | _ -> None

(* The merge [B] needs: buffer id -> axis [a], or [Refused]. *)
let plan ~(buffers : Ssa_buffer.t Id_map.t) ~total ~v1 ~v2 ~n2
    (body : Ssa_region.t) =
  let local = access_counts body in
  let merges =
    fold_instrs
      (fun m (i : Ssa_instr.t) ->
        let named =
          List.exists
            (fun v -> Ssa_value.equal v v1 || Ssa_value.equal v v2)
            (Ssa_op.operands i.Ssa_instr.op)
        in
        if not named then m
        else
          match accesses i.Ssa_instr.op with
          | [ (b, Rewritable c) ] -> (
              let buf =
                match Id_map.find_opt (bid b) buffers with
                | Some buf -> buf
                | None -> raise Refused
              in
              if Ssa_format.per_channel buf.Ssa_buffer.format then raise Refused;
              (* each of [v1], [v2] is one coordinate, the only ones naming
                 them, on adjacent axes, the inner one whole *)
              match (axis_of c v1, axis_of c v2) with
              | Some a, Some a2 when axis_after a = Some a2 ->
                  if
                    not
                      (Int64.equal
                         (Expr.Coord.get buf.Ssa_buffer.extents a2)
                         n2)
                  then raise Refused;
                  (match Id_map.find_opt (bid b) m with
                  | Some a' when a' <> a -> raise Refused
                  | _ -> ());
                  Id_map.add (bid b) a m
              | _ -> raise Refused)
          | _ -> raise Refused)
      Id_map.empty body
  in
  (* a merged buffer is accessed nowhere else *)
  Id_map.iter
    (fun b _ ->
      if Id_map.find_opt b total <> Id_map.find_opt b local then raise Refused)
    merges;
  merges

let instr ~results op =
  Ssa_stmt.Instr
    { Ssa_instr.results; op; token = None; origin = Ssa_origin.Unknown }

let pass (p : Ssa_program.t) =
  let ranges = Ssa_range.analyze p in
  let total = access_counts p.Ssa_program.entry in
  let buffers =
    List.fold_left
      (fun m (b : Ssa_buffer.t) -> Id_map.add (bid b.Ssa_buffer.id) b m)
      Id_map.empty p.Ssa_program.buffers
  in
  let merged = ref Id_map.empty in
  let constant v =
    match Ssa_range.range ranges v with
    | Ssa_range.Range r when Int64.equal r.lo r.hi -> Some r.lo
    | _ -> None
  in
  let trips ~lo ~hi =
    match (constant lo, Ssa_range.trips ranges ~lo ~hi ~step:1L) with
    | Some 0L, Ssa_range.Exactly n when Int64.compare n 0L > 0 -> Some n
    | _ -> None
  in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.For
        ({
           lo;
           hi;
           step = 1L;
           inits = [ _ ];
           results = [ _ ];
           body =
             {
               Ssa_region.params = [ v1; e1 ];
               body =
                 [
                   Ssa_stmt.For
                     {
                       lo = lo2;
                       hi = hi2;
                       step = 1L;
                       inits = [ e1' ];
                       results = [ r2 ];
                       body = inner;
                     };
                 ];
               yields = [ y1 ];
               _;
             };
         } as outer)
      when Ssa_value.equal e1 e1' && Ssa_value.equal y1 r2 -> (
        match
          (trips ~lo ~hi, trips ~lo:lo2 ~hi:hi2, inner.Ssa_region.params)
        with
        | Some n1, Some n2, [ v2; _ ] -> (
            let fits =
              (* the merged trips stay inside the index domain *)
              Int64.compare n1 (Int64.div Ssa_buffer.max_elements n2) <= 0
            in
            match
              if fits then Some (plan ~buffers ~total ~v1 ~v2 ~n2 inner)
              else None
            with
            | exception Refused -> [ s ]
            | None -> [ s ]
            | Some merges when Id_map.is_empty merges -> [ s ]
            | Some merges
              when Id_map.exists (fun b _ -> Id_map.mem b !merged) merges ->
                (* a buffer merges one pair of axes a pass: its declared
                   extents are rewritten once, at the end; the next round
                   sees them and may merge another pair *)
                [ s ]
            | Some merges -> (
                (* the body again, its merged accesses at [(0, g)]; [v1] or
                   [v2] anywhere but there (an inner loop's bound, a yield)
                   refuses the pair *)
                let zero =
                  Ssa_rewrite.fresh t (Ssa_type.Scalar Ssa_type.Index)
                in
                let hi_g =
                  Ssa_rewrite.fresh t (Ssa_type.Scalar Ssa_type.Index)
                in
                let rewrite (op : Ssa_op.t) =
                  let at buffer c =
                    match Id_map.find_opt (bid buffer) merges with
                    | Some a -> Expr.Coord.set c a zero
                    | None -> c
                  in
                  match op with
                  | Ssa_op.Load_in_bounds
                      { buffer; at = Ssa_access.Coord c; decode } ->
                      Ssa_op.Load_in_bounds
                        { buffer; at = Ssa_access.Coord (at buffer c); decode }
                  | Ssa_op.Store
                      { buffer; at = Ssa_access.Coord c; encode; value } ->
                      Ssa_op.Store
                        {
                          buffer;
                          at = Ssa_access.Coord (at buffer c);
                          encode;
                          value;
                        }
                  | op -> op
                in
                let named vs =
                  List.exists
                    (fun v -> Ssa_value.equal v v1 || Ssa_value.equal v v2)
                    vs
                in
                let rec region (r : Ssa_region.t) =
                  if named r.Ssa_region.yields then raise Refused;
                  { r with Ssa_region.body = List.map stmt r.Ssa_region.body }
                and stmt = function
                  | Ssa_stmt.Instr i ->
                      let op = rewrite i.Ssa_instr.op in
                      if List.exists (Ssa_value.equal v1) (Ssa_op.operands op)
                      then raise Refused;
                      Ssa_stmt.Instr { i with Ssa_instr.op }
                  | Ssa_stmt.For f ->
                      if named (f.lo :: f.hi :: f.inits) then raise Refused;
                      Ssa_stmt.For { f with body = region f.body }
                  | Ssa_stmt.Ordered_sum f ->
                      if named [ f.lo; f.hi; f.seed; f.token ] then
                        raise Refused;
                      Ssa_stmt.Ordered_sum { f with body = region f.body }
                  | Ssa_stmt.If f ->
                      if named [ f.cond ] then raise Refused;
                      Ssa_stmt.If
                        {
                          f with
                          then_ = region f.then_;
                          else_ = region f.else_;
                        }
                in
                match region inner with
                | exception Refused -> [ s ]
                | body ->
                    Id_map.iter
                      (fun b a -> merged := Id_map.add b a !merged)
                      merges;
                    Ssa_rewrite.mark_changed t;
                    [
                      instr ~results:[ zero ]
                        (Ssa_op.Const (Ssa_const.Index 0L));
                      instr ~results:[ hi_g ]
                        (Ssa_op.Const (Ssa_const.Index (Int64.mul n1 n2)));
                      Ssa_stmt.For { outer with lo = zero; hi = hi_g; body };
                    ]))
        | _ -> [ s ])
    | Ssa_stmt.For _ | Ssa_stmt.If _ | Ssa_stmt.Instr _ | Ssa_stmt.Ordered_sum _
      ->
        [ s ]
  in
  let q, changed = Ssa_rewrite.program rule p in
  let buffers =
    List.map
      (fun (b : Ssa_buffer.t) ->
        match Id_map.find_opt (bid b.Ssa_buffer.id) !merged with
        | None -> b
        | Some a ->
            let a2 = Option.get (axis_after a) in
            let e = b.Ssa_buffer.extents in
            let joined = Int64.mul (Expr.Coord.get e a) (Expr.Coord.get e a2) in
            {
              b with
              Ssa_buffer.extents =
                Expr.Coord.set (Expr.Coord.set e a 1L) a2 joined;
            })
      q.Ssa_program.buffers
  in
  ({ q with Ssa_program.buffers }, changed)
