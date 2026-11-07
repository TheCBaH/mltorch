(* Unroll-and-interleave of vector loops: a vector loop of constant trips
   becomes a main loop whose body is [unroll] consecutive copies of the vector
   body, followed by the original loop for the trips left over. The copies are
   interleaved so the loads of copy [k + 1] issue between the operations of
   copy [k] (one after its last load and one after each operation after it), which keeps
   about three vector groups live for a two-copy add (load A0, load B0, load
   A1, add 0, load B1, store 0, add 1, store 1) and lets a core overlap one
   trip's memory latency with the other's arithmetic.

   What keeps it exact:
   - Each copy is the body at [iv + k * step], so the copies cover the same
     lanes in the same order as [unroll] trips of the original loop, and the
     remainder runs the rest: the statements run the same number of times.
   - Loads stay in order among loads and stores among stores; the effect chain
     is rebuilt in the new order. The one reordering is a load of a later copy
     issued above a store of an earlier one. It is allowed only when every
     store and every load of a buffer that may overlap it (under the alias
     policy) use the same coordinate: then each copy touches only its own
     lanes, the cells a later copy reads are ones no earlier copy writes, which
     is what vectorizing at [unroll] times the lanes would need. A loop with a
     store and a load of one buffer at different coordinates (a recurrence at
     a distance) keeps its trips in order.
   - The body is straight-line and its only effects are vector loads and
     stores and marks: nothing fails, so no failure moves. *)

let id (v : Ssa_value.t) = (v.Ssa_value.id :> int)

let instr ?token ~results op =
  { Ssa_instr.results; op; token; origin = Ssa_origin.Unknown }

let is_load (i : Ssa_instr.t) =
  match i.Ssa_instr.op with Ssa_op.Vec_load _ -> true | _ -> false

(* The body's instructions, if it is straight-line and its effects are only
   what this pass may reorder. *)
let straight (body : Ssa_region.t) =
  let rec go acc = function
    | [] -> Some (List.rev acc)
    | Ssa_stmt.Instr i :: rest -> (
        match i.Ssa_instr.op with
        | Ssa_op.Vec_load _ | Ssa_op.Vec_store _ | Ssa_op.Mark _
        | Ssa_op.Mark_lanes _ ->
            go (i :: acc) rest
        | op when Ssa_op.effectful op -> None
        | _ -> go (i :: acc) rest)
    | (Ssa_stmt.For _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _) :: _ -> None
  in
  go [] body.Ssa_region.body

let has_vector instrs =
  List.exists
    (fun (i : Ssa_instr.t) ->
      match i.Ssa_instr.op with
      | Ssa_op.Vec_load _ | Ssa_op.Vec_store _ | Ssa_op.Mark_lanes _ -> true
      | _ -> false)
    instrs

(* The register budget: a body defining at most this many vector values. Two
   interleaved copies of one then hold about three of Coral's five m4 groups
   live (a residual add defines three); a heavier body (a binary16 decode, a
   GELU) gains little from the overlap and its copies spill, enough to take
   an RV32 frame past its code model. *)
let max_vectors = 4

(* A body whose operations stay what they are through Machine IR lowering:
   binary32 loads and stores with no decode or encode work, and the cheap
   lanewise arithmetic. A decode (binary16, bfloat16, a dequantization) or a
   transcendental expands into a long sequence whose copies would not fit. *)
let light (i : Ssa_instr.t) =
  match i.Ssa_instr.op with
  | Ssa_op.Vec_load { decode = Ssa_op.Decode.F32_exact; _ }
  | Ssa_op.Vec_store { encode = Ssa_op.Encode.F32_exact; _ } ->
      true
  | Ssa_op.Vec_load _ | Ssa_op.Vec_store _ -> false
  | Ssa_op.Lanewise
      ( Ssa_op.Float_binary
          ((Expr.Value.Add | Expr.Value.Mul | Expr.Value.Sub), _, _)
      | Ssa_op.Float_max _ ) ->
      true
  | Ssa_op.Lanewise _ -> false
  | Ssa_op.Const _ | Ssa_op.Index_add_in_domain _
  | Ssa_op.Index_scale_in_domain _ | Ssa_op.Mark _ | Ssa_op.Mark_lanes _
  | Ssa_op.Vec_splat _ ->
      true
  | _ -> false

(* the vectors each copy defines: a splat is invariant (the hoist after this
   pass moves it out), one register every copy shares *)
let vectors instrs =
  List.fold_left
    (fun n (i : Ssa_instr.t) ->
      match i.Ssa_instr.op with
      | Ssa_op.Vec_splat _ -> n
      | _ ->
          n
          + List.length
              (List.filter
                 (fun (v : Ssa_value.t) ->
                   match v.Ssa_value.ty with
                   | Ssa_type.Vec _ | Ssa_type.Mask _ -> true
                   | _ -> false)
                 i.Ssa_instr.results))
    0 instrs

(* Every store and every load of a buffer that may overlap it at the same
   coordinate and steps. *)
let independent ~alias instrs =
  let at = function
    | Ssa_op.Vec_load { buffer; at; steps; _ } -> Some (`Load, buffer, at, steps)
    | Ssa_op.Vec_store { buffer; at; steps; _ } ->
        Some (`Store, buffer, at, steps)
    | _ -> None
  in
  let accesses =
    List.filter_map (fun (i : Ssa_instr.t) -> at i.Ssa_instr.op) instrs
  in
  let same a b =
    List.for_all2 Ssa_value.equal (Expr.Coord.to_list a) (Expr.Coord.to_list b)
  in
  List.for_all
    (fun (k, b, c, s) ->
      k <> `Store
      || List.for_all
           (fun (k', b', c', s') ->
             k' <> `Load
             || (not (Ssa_effects.may_overlap alias b b'))
             || (same c c' && Expr.Coord.to_list s = Expr.Coord.to_list s'))
           accesses)
    accesses

(* The copies' instructions in issue order: copy [k]'s, with one load of copy
   [k + 1] after its last load and after each of its later instructions, a
   load issued early only once its operands are defined. *)
let interleave ~outside copies =
  let defined = Hashtbl.create 64 in
  let ready (i : Ssa_instr.t) =
    List.for_all
      (fun v -> Hashtbl.mem defined (id v) || outside v)
      (Ssa_op.operands i.Ssa_instr.op)
  in
  (* every instruction defines a result (an effect at least), so its result
     ids name it *)
  let key (i : Ssa_instr.t) = List.map id i.Ssa_instr.results in
  let out = ref [] and issued = Hashtbl.create 64 in
  let emit (i : Ssa_instr.t) =
    out := i :: !out;
    Hashtbl.replace issued (key i) ();
    List.iter (fun r -> Hashtbl.replace defined (id r) ()) i.Ssa_instr.results
  in
  let rec go = function
    | [] -> ()
    | c :: rest ->
        let next =
          ref (match rest with n :: _ -> List.filter is_load n | [] -> [])
        in
        let last_load =
          List.fold_left
            (fun acc (i : Ssa_instr.t) ->
              if is_load i then Some (key i) else acc)
            None c
        in
        List.iter
          (fun (i : Ssa_instr.t) ->
            if not (Hashtbl.mem issued (key i)) then (
              emit i;
              if (not (is_load i)) || Some (key i) = last_load then
                match !next with
                | l :: more when ready l ->
                    emit l;
                    next := more
                | _ -> ()))
          c;
        go rest
  in
  go copies;
  List.rev !out

let pass ~unroll ~alias (p : Ssa_program.t) =
  if unroll <= 1 then (p, false)
  else
    let ranges = Ssa_range.analyze p in
    let constant v =
      match Ssa_range.range ranges v with
      | Ssa_range.Range r when Int64.equal r.lo r.hi -> Some r.lo
      | _ -> None
    in
    let index = Ssa_type.Scalar Ssa_type.Index in
    let rule t (s : Ssa_region.t Ssa_stmt.t) =
      match s with
      | Ssa_stmt.For
          ({
             lo;
             hi;
             step;
             inits = [ e0 ];
             results = [ r ];
             body = { Ssa_region.params = [ iv; _ ]; yields = [ _ ]; _ } as body;
           } as f)
        when Int64.compare step 1L > 0 -> (
          match (constant lo, constant hi, straight body) with
          | Some l, Some h, Some instrs
            when has_vector instrs
                 && vectors instrs <= max_vectors
                 && List.for_all light instrs && independent ~alias instrs ->
              let u = Int64.of_int unroll in
              let trips = Int64.div (Int64.sub h l) step in
              let groups = Int64.div trips u in
              if Int64.compare groups 1L < 0 then [ s ]
              else
                let main_hi =
                  Int64.add l (Int64.mul (Int64.mul groups u) step)
                in
                let const k =
                  let v = Ssa_rewrite.fresh t index in
                  ( v,
                    Ssa_stmt.Instr
                      (instr ~results:[ v ] (Ssa_op.Const (Ssa_const.Index k)))
                  )
                in
                let hi_v, hi_s = const main_hi in
                let iv' = Ssa_rewrite.fresh t index in
                let e' = Ssa_rewrite.fresh t Ssa_type.Effect in
                let bases =
                  List.init unroll (fun k ->
                      if k = 0 then (iv', [])
                      else
                        let c, c_s = const (Int64.mul (Int64.of_int k) step) in
                        let b = Ssa_rewrite.fresh t index in
                        ( b,
                          [
                            c_s;
                            Ssa_stmt.Instr
                              (instr ~results:[ b ]
                                 (Ssa_op.Index_add_in_domain (iv', c)));
                          ] ))
                in
                let copies =
                  List.map
                    (fun (b, _) ->
                      let cl = Ssa_clone.create t ~subst:[ (iv, b) ] in
                      List.map
                        (fun i ->
                          match Ssa_clone.stmt cl (Ssa_stmt.Instr i) with
                          | Ssa_stmt.Instr i -> i
                          | _ -> assert false)
                        instrs)
                    bases
                in
                let prelude = List.concat_map snd bases in
                let inside = Hashtbl.create 64 in
                List.iter
                  (fun (i : Ssa_instr.t) ->
                    List.iter
                      (fun v -> Hashtbl.replace inside (id v) ())
                      i.Ssa_instr.results)
                  (List.concat copies);
                let outside v = not (Hashtbl.mem inside (id v)) in
                (* the effect chain, again, in issue order *)
                let cur = ref e' in
                let body_instrs =
                  List.map
                    (fun (i : Ssa_instr.t) ->
                      match i.Ssa_instr.token with
                      | None -> Ssa_stmt.Instr i
                      | Some _ ->
                          let i = { i with Ssa_instr.token = Some !cur } in
                          cur :=
                            List.nth i.Ssa_instr.results
                              (List.length i.Ssa_instr.results - 1);
                          Ssa_stmt.Instr i)
                    (interleave ~outside copies)
                in
                let main_r =
                  if Int64.equal main_hi h then r
                  else Ssa_rewrite.fresh t Ssa_type.Effect
                in
                let region_id =
                  let rid, nx =
                    Ssa_id.Region.Next.alloc t.Ssa_rewrite.next_region
                  in
                  t.Ssa_rewrite.next_region <- nx;
                  rid
                in
                let main =
                  Ssa_stmt.For
                    {
                      lo;
                      hi = hi_v;
                      step = Int64.mul u step;
                      inits = [ e0 ];
                      results = [ main_r ];
                      body =
                        {
                          Ssa_region.id = region_id;
                          params = [ iv'; e' ];
                          body = prelude @ body_instrs;
                          yields = [ !cur ];
                        };
                    }
                in
                Ssa_rewrite.mark_changed t;
                hi_s :: main
                ::
                (if Int64.equal main_hi h then []
                 else [ Ssa_stmt.For { f with lo = hi_v; inits = [ main_r ] } ])
          | _ -> [ s ])
      | Ssa_stmt.For _ | Ssa_stmt.If _ | Ssa_stmt.Instr _
      | Ssa_stmt.Ordered_sum _ ->
          [ s ]
    in
    Ssa_rewrite.program rule p
