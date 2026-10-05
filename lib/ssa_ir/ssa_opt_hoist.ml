(* Loop-invariant code motion.

   A pure operation is total (no failure, no effect), so it may run before a
   loop that might not run: moving it can only compute a value nothing reads.
   Its operands must all be defined outside the loop; operations hoisted from the
   body count as outside for the ones after them.

   A load is not total and it reads memory, so it moves only under a proof for
   each part: it cannot fail (it is the in-bounds form), the loop runs at least
   once (hoisting a read out of a loop that never runs adds a read), and nothing
   in the loop may write what it reads (the alias policy is the caller's). Its
   place on the effect chain moves with it: it consumes the effect the loop was
   given and the loop receives the one it returns, and the body's own chain is
   rejoined across the gap. Reads are not logical work, so no mark changes. *)

module Id_set = Set.Make (Int)

(* Every value defined inside a region, however deeply. *)
let rec defs_of_region acc (r : Ssa_region.t) =
  let add acc (v : Ssa_value.t) = Id_set.add (v.Ssa_value.id :> int) acc in
  let acc = List.fold_left add acc r.Ssa_region.params in
  List.fold_left
    (fun acc s ->
      match s with
      | Ssa_stmt.Instr i -> List.fold_left add acc i.Ssa_instr.results
      | Ssa_stmt.For { results; body; _ }
      | Ssa_stmt.Ordered_sum { results; body; _ } ->
          defs_of_region (List.fold_left add acc results) body
      | Ssa_stmt.If { results; then_; else_; _ } ->
          defs_of_region
            (defs_of_region (List.fold_left add acc results) then_)
            else_)
    acc r.Ssa_region.body

type motion = {
  hoisted : Ssa_region.t Ssa_stmt.t list;  (** in order, before the loop *)
  body : Ssa_region.t;
  entry : Ssa_value.t;  (** the effect the loop now starts from *)
}

(* Pulls what can move out of [body]. [entry] is the effect the loop receives;
   [nonempty] and [policy] gate the loads. *)
let extract t ~policy ~nonempty ~entry (body : Ssa_region.t) =
  let summary = Ssa_effects.of_region body in
  let inner = ref (defs_of_region Id_set.empty body) in
  let invariant (v : Ssa_value.t) =
    not (Id_set.mem (v.Ssa_value.id :> int) !inner)
  in
  let entry = ref entry in
  let hoisted = ref [] in
  let moved (i : Ssa_instr.t) =
    List.iter
      (fun (r : Ssa_value.t) ->
        inner := Id_set.remove (r.Ssa_value.id :> int) !inner)
      i.Ssa_instr.results
  in
  let kept =
    List.filter
      (fun s ->
        match s with
        | Ssa_stmt.Instr i
          when (not (Ssa_op.effectful i.Ssa_instr.op))
               && List.for_all invariant (Ssa_op.operands i.Ssa_instr.op) ->
            hoisted := s :: !hoisted;
            moved i;
            Ssa_rewrite.mark_changed t;
            false
        | Ssa_stmt.Instr
            ({
               Ssa_instr.op = Ssa_op.Load_in_bounds { buffer; _ };
               token = Some body_in;
               results;
               _;
             } as i)
          when nonempty
               && List.for_all invariant (Ssa_op.operands i.Ssa_instr.op)
               && not (Ssa_effects.may_write policy summary buffer) ->
            (* the read goes first on the chain: it takes the effect the loop
               was given, and the loop takes the effect it returns *)
            let out_old = List.nth results (List.length results - 1) in
            let out_new = Ssa_rewrite.fresh t Ssa_type.Effect in
            let load =
              {
                i with
                Ssa_instr.token = Some !entry;
                results =
                  List.filteri (fun k _ -> k < List.length results - 1) results
                  @ [ out_new ];
              }
            in
            (* the body's chain skips the gap the read left *)
            Ssa_rewrite.alias t ~from:out_old ~to_:body_in;
            entry := out_new;
            hoisted := Ssa_stmt.Instr load :: !hoisted;
            moved i;
            false
        | _ -> true)
      body.Ssa_region.body
  in
  {
    hoisted = List.rev !hoisted;
    body = { body with Ssa_region.body = kept };
    entry = !entry;
  }

let pass ~alias (p : Ssa_program.t) =
  let ranges = Ssa_range.analyze p in
  let nonempty ~lo ~hi ~step =
    match Ssa_range.trips ranges ~lo ~hi ~step with
    | Ssa_range.At_least_one -> true
    | Ssa_range.Exactly n -> Int64.compare n 0L > 0
    | Ssa_range.Zero | Ssa_range.Unknown -> false
  in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.For { lo; hi; step; inits; results; body } -> (
        (* the carried effect: the one effect-typed init *)
        let effect_index =
          let rec find k = function
            | [] -> None
            | (v : Ssa_value.t) :: rest ->
                if Ssa_type.equal v.Ssa_value.ty Ssa_type.Effect then Some k
                else find (k + 1) rest
          in
          find 0 inits
        in
        match effect_index with
        | None -> [ s ]
        | Some k ->
            let m =
              extract t ~policy:alias ~nonempty:(nonempty ~lo ~hi ~step)
                ~entry:(List.nth inits k) body
            in
            if m.hoisted = [] then [ s ]
            else
              let inits =
                List.mapi (fun j v -> if j = k then m.entry else v) inits
              in
              (* the body's chain was rejoined by aliases, so every statement
                 and yield of the body is read through them again *)
              let body = Ssa_rewrite.resolve_region t m.body in
              m.hoisted
              @ [ Ssa_stmt.For { lo; hi; step; inits; results; body } ])
    | Ssa_stmt.Ordered_sum { lo; hi; seed; token; results; body } ->
        let m =
          extract t ~policy:alias
            ~nonempty:(nonempty ~lo ~hi ~step:1L)
            ~entry:token body
        in
        if m.hoisted = [] then [ s ]
        else
          let body = Ssa_rewrite.resolve_region t m.body in
          m.hoisted
          @ [
              Ssa_stmt.Ordered_sum
                { lo; hi; seed; token = m.entry; results; body };
            ]
    | Ssa_stmt.Instr _ | Ssa_stmt.If _ -> [ s ]
  in
  Ssa_rewrite.program rule p
