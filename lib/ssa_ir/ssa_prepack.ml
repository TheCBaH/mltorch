(* Prepacked weights: a [Contract] whose B is a constant read with strided
   columns (a [Linear]'s weight, [n] rows of [k]) reads instead a packed copy
   of it, [k] rows of [n] contiguous columns, which the host builds once from
   the constant before any invocation runs. The function then loads each B
   row with one unit-stride vector load rather than one access per element.
   A batch-shared B ([batch] stride 0) packs one batch. A constant no other
   access reads afterwards leaves the program, so its bytes are not bound
   twice. *)

module Pack = struct
  type t = {
    source : Ssa_id.Buffer.t;
    target : Ssa_id.Buffer.t;
    format : Contraction.Element.t;
    batches : int64;  (** batches packed: 1 for a batch-shared B *)
    k : int64;
    n : int64;
    from : Contraction.Operand.t;  (** B's strides in the source *)
  }

  (* The source element of packed element [e]: batch, then [k], then column. *)
  let source_offset t e =
    let per = Int64.mul t.k t.n in
    let bb = Int64.div e per and r = Int64.rem e per in
    let kk = Int64.div r t.n and j = Int64.rem r t.n in
    Int64.add
      (Int64.mul bb t.from.Contraction.Operand.batch)
      (Int64.add
         (Int64.mul j t.from.Contraction.Operand.outer)
         (Int64.mul kk t.from.Contraction.Operand.k))

  let elements t = Int64.mul t.batches (Int64.mul t.k t.n)
end

let rec map_region f (r : Ssa_region.t) =
  { r with Ssa_region.body = List.map (map_stmt f) r.Ssa_region.body }

and map_stmt f (s : Ssa_region.t Ssa_stmt.t) =
  match s with
  | Ssa_stmt.Instr i -> Ssa_stmt.Instr (f i)
  | Ssa_stmt.For x -> Ssa_stmt.For { x with body = map_region f x.body }
  | Ssa_stmt.Ordered_sum x ->
      Ssa_stmt.Ordered_sum { x with body = map_region f x.body }
  | Ssa_stmt.If x ->
      Ssa_stmt.If
        { x with then_ = map_region f x.then_; else_ = map_region f x.else_ }

let rec iter_region f (r : Ssa_region.t) =
  List.iter
    (function
      | Ssa_stmt.Instr i -> f i
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          iter_region f body
      | Ssa_stmt.If { then_; else_; _ } ->
          iter_region f then_;
          iter_region f else_)
    r.Ssa_region.body

let program ~constant (p : Ssa_program.t) =
  let next =
    ref
      (1
      + List.fold_left
          (fun m (b : Ssa_buffer.t) -> max m (b.Ssa_buffer.id :> int))
          0 p.Ssa_program.buffers)
  in
  let packs = ref [] in
  let pack (d : Contraction.t) b =
    let from = d.Contraction.b in
    let batches =
      if Int64.equal from.Contraction.Operand.batch 0L then 1L
      else d.Contraction.batch
    in
    let same (q : Pack.t) =
      Ssa_id.Buffer.equal q.Pack.source b
      && q.Pack.from = from && q.Pack.batches = batches
      && q.Pack.k = d.Contraction.k && q.Pack.n = d.Contraction.n
    in
    match List.find_opt same !packs with
    | Some q -> q
    | None ->
        let q =
          {
            Pack.source = b;
            target = Ssa_id.Buffer.of_int !next;
            format = d.Contraction.format;
            batches;
            k = d.Contraction.k;
            n = d.Contraction.n;
            from;
          }
        in
        incr next;
        packs := !packs @ [ q ];
        q
  in
  let rewrite (i : Ssa_instr.t) =
    match i.Ssa_instr.op with
    | Ssa_op.Contract { desc = d; a; b; c }
      when constant b
           && Int64.compare d.Contraction.b.Contraction.Operand.outer 1L <> 0
           && Int64.compare d.Contraction.k 0L > 0 -> (
        let q = pack d b in
        match
          Err.payload
            (Contraction.make ~batch:d.Contraction.batch ~m:d.Contraction.m
               ~n:d.Contraction.n ~k:d.Contraction.k ~a:d.Contraction.a
               ~b:
                 {
                   Contraction.Operand.batch =
                     (if Int64.equal q.Pack.batches 1L then 0L
                      else Int64.mul d.Contraction.k d.Contraction.n);
                   outer = 1L;
                   k = d.Contraction.n;
                 }
               ~c:d.Contraction.c ~format:d.Contraction.format
               ~seed:d.Contraction.seed ~numerics:d.Contraction.numerics)
        with
        | Ok desc ->
            {
              i with
              Ssa_instr.op = Ssa_op.Contract { desc; a; b = q.Pack.target; c };
            }
        | Error _ -> i)
    | _ -> i
  in
  let entry = map_region rewrite p.Ssa_program.entry in
  if !packs = [] then (p, [])
  else
    let used = ref Ssa_id.Buffer.Set.empty in
    iter_region
      (fun i ->
        let e = Ssa_effects.of_op i.Ssa_instr.op in
        used :=
          Ssa_id.Buffer.Set.union !used
            (Ssa_id.Buffer.Set.union e.Ssa_effects.reads e.Ssa_effects.writes))
      entry;
    let source id =
      List.find
        (fun (b : Ssa_buffer.t) -> Ssa_id.Buffer.equal b.Ssa_buffer.id id)
        p.Ssa_program.buffers
    in
    let added =
      List.map
        (fun (q : Pack.t) ->
          {
            (source q.Pack.source) with
            Ssa_buffer.id = q.Pack.target;
            extents =
              Expr.Coord.of_fn (fun a ->
                  match a with
                  | Expr.Axis.H -> q.Pack.batches
                  | Expr.Axis.W -> q.Pack.k
                  | Expr.Axis.C -> q.Pack.n
                  | Expr.Axis.N | Expr.Axis.T | Expr.Axis.D -> 1L);
            role = Ssa_buffer.Input;
          })
        !packs
    in
    ( {
        p with
        Ssa_program.revision = Ssa_id.Revision.succ p.Ssa_program.revision;
        buffers =
          List.filter
            (fun (b : Ssa_buffer.t) ->
              Ssa_id.Buffer.Set.mem b.Ssa_buffer.id !used
              || not
                   (List.exists
                      (fun (q : Pack.t) ->
                        Ssa_id.Buffer.equal q.Pack.source b.Ssa_buffer.id)
                      !packs))
            p.Ssa_program.buffers
          @ added;
        entry;
      },
      !packs )
