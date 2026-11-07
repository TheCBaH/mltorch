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
   rejoined across the gap. Reads are not logical work, so no mark changes.

   A read of a scratch object ([Local_read]) moves the same way, under four
   conditions: its operands are invariant (the object is allocated outside the
   loop); the loop runs at least once (otherwise the hoisted read could reach a
   cell nothing wrote, a defect the loop never had); its index is a constant
   [Index k] inside the object's [slots] (so it cannot fail); and nothing in the
   loop writes that object (only [Local_write] does, so the cell holds the
   value it had at the loop's entry on every trip). A non-constant index stays
   in the loop: no range proof is attempted. *)

module Id_set = Set.Make (Int)
module Id_map = Map.Make (Int)

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
   [nonempty] and [policy] gate the loads, [nonempty] and [in_bounds] the local
   reads. *)
let extract t ~in_bounds ~policy ~nonempty ~entry (body : Ssa_region.t) =
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
  (* the read goes first on the chain: it takes the effect the loop was
     given, and the loop takes the effect it returns *)
  let first_on_chain ~body_in (i : Ssa_instr.t) =
    let results = i.Ssa_instr.results in
    let out_old = List.nth results (List.length results - 1) in
    let out_new = Ssa_rewrite.fresh t Ssa_type.Effect in
    let read =
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
    hoisted := Ssa_stmt.Instr read :: !hoisted;
    moved i
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
               _;
             } as i)
          when nonempty
               && List.for_all invariant (Ssa_op.operands i.Ssa_instr.op)
               && not (Ssa_effects.may_write policy summary buffer) ->
            first_on_chain ~body_in i;
            false
        | Ssa_stmt.Instr
            ({
               Ssa_instr.op = Ssa_op.Local_read { local; at; _ };
               token = Some body_in;
               _;
             } as i)
          when nonempty
               && List.for_all invariant (Ssa_op.operands i.Ssa_instr.op)
               && in_bounds ~local ~at
               && not (Ssa_effects.writes_local body local) ->
            first_on_chain ~body_in i;
            false
        | _ -> true)
      body.Ssa_region.body
  in
  {
    hoisted = List.rev !hoisted;
    body = { body with Ssa_region.body = kept };
    entry = !entry;
  }

(* Each scratch object's slot count and each constant index, by value id: one
   scan of the program, for the local-read rule's bounds. *)
let locals_and_indices (p : Ssa_program.t) =
  let rec scan acc (r : Ssa_region.t) =
    List.fold_left
      (fun ((slots, indices) as acc) s ->
        match s with
        | Ssa_stmt.Instr
            {
              Ssa_instr.op = Ssa_op.Local_alloc { slots = n; _ };
              results = l :: _;
              _;
            } ->
            (Id_map.add (l.Ssa_value.id :> int) n slots, indices)
        | Ssa_stmt.Instr
            {
              Ssa_instr.op = Ssa_op.Const (Ssa_const.Index k);
              results = [ v ];
              _;
            } ->
            (slots, Id_map.add (v.Ssa_value.id :> int) k indices)
        | Ssa_stmt.Instr _ -> acc
        | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
            scan acc body
        | Ssa_stmt.If { then_; else_; _ } -> scan (scan acc then_) else_)
      acc r.Ssa_region.body
  in
  scan (Id_map.empty, Id_map.empty) p.Ssa_program.entry

let pass ~alias (p : Ssa_program.t) =
  let ranges = Ssa_range.analyze p in
  let slots, indices = locals_and_indices p in
  let in_bounds ~(local : Ssa_value.t) ~(at : Ssa_value.t) =
    match
      ( Id_map.find_opt (local.Ssa_value.id :> int) slots,
        Id_map.find_opt (at.Ssa_value.id :> int) indices )
    with
    | Some n, Some k -> Int64.compare k 0L >= 0 && Int64.compare k n < 0
    | _ -> false
  in
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
              extract t ~in_bounds ~policy:alias
                ~nonempty:(nonempty ~lo ~hi ~step) ~entry:(List.nth inits k)
                body
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
          extract t ~in_bounds ~policy:alias
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
