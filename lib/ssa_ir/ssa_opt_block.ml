(* Automatic independent-output blocking: unroll-and-jam of a loop whose
   iterations are independent outputs around a common reduction.

   The shape, which is what matmul's lowering produces after the guards pass:

     for w in [lo, hi):            (* one output per iteration *)
       pre                         (* pure setup *)
       s = ordered_sum k: term     (* a reduction over a domain that does not
                                      depend on w *)
       post                        (* the stores, at addresses that differ with w *)

   becomes, for a group of G outputs, a loop of full groups

     for w in [lo, lo + q*G) step G:
       pre_0 .. pre_{G-1}
       (s_0 .. s_{G-1}) = for k: acc_j <- acc_j + term_j     (one loop, G accumulators)
       post_0 .. post_{G-1}

   followed by the scalar loop over the remaining [hi - lo] mod G outputs, which
   is the original body unchanged. Each output keeps its own accumulator and adds
   its terms in the original order, so its value is bit-for-bit what it was; the
   logical work (a mark per term per output) is the same; only the order in which
   different outputs are computed changes, which is unobservable once nothing
   in the loop can fail and the stores are to distinct addresses. A read the
   clones now repeat (the term's operand that does not depend on w) is shared by
   the load-sharing pass afterwards.

   Legality is checked, not assumed, and a refusal says which condition failed. *)

module Id_set = Set.Make (Int)

type refusal =
  | Carries_values
  | Domain_end
  | Meter_or_locals
  | May_fail
  | Reads_written_buffer
  | Shape
  | Stores_not_independent
  | Sum_bounds_vary
  | Too_few_trips
  | Trips_unknown
  | Unprofitable

let refusal_name = function
  | Carries_values -> "the loop carries a value besides the effect"
  | Domain_end -> "the blocked index would leave the index domain"
  | Meter_or_locals -> "the body touches the scan meter or a scratch object"
  | May_fail -> "the body still contains an operation that can fail"
  | Reads_written_buffer -> "the body reads a buffer it may write"
  | Shape -> "the body is not setup, one ordered sum, then stores"
  | Stores_not_independent ->
      "the stores are not at addresses that differ per iteration"
  | Sum_bounds_vary -> "the reduction's bounds vary with the iteration"
  | Too_few_trips -> "fewer iterations than one group"
  | Trips_unknown -> "the trip count is not a constant"
  | Unprofitable -> "no operand is shared across the group"

type candidate = {
  region : Ssa_id.Region.t;
  trips : int64;
  shared_reads : int;
      (** reads of the reduction that do not depend on the output *)
  decision : (int, refusal) result;  (** the group size, or why not *)
}

type shape = {
  lo : int64;
  hi : int64;
  trips : int64;
  iv : Ssa_value.t;
  effect_param : Ssa_value.t;
  pre : Ssa_region.t Ssa_stmt.t list;
  sum_lo : Ssa_value.t;
  sum_hi : Ssa_value.t;
  seed : Ssa_value.t;
  sum_results : Ssa_value.t list;
  sum_body : Ssa_region.t;
  post : Ssa_region.t Ssa_stmt.t list;
  shared_reads : int;
}

let rec defs_of_stmts acc stmts =
  let add acc (v : Ssa_value.t) = Id_set.add (v.Ssa_value.id :> int) acc in
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
    acc stmts

and defs_of_region acc (r : Ssa_region.t) =
  let acc =
    List.fold_left
      (fun acc (v : Ssa_value.t) -> Id_set.add (v.Ssa_value.id :> int) acc)
      acc r.Ssa_region.params
  in
  defs_of_stmts acc r.Ssa_region.body

(* The values derived from [iv] by instructions of the body. *)
let derived_from iv stmts =
  let derived = ref (Id_set.singleton (iv.Ssa_value.id :> int)) in
  let rec go stmts =
    List.iter
      (fun s ->
        match s with
        | Ssa_stmt.Instr i ->
            if
              List.exists
                (fun (v : Ssa_value.t) ->
                  Id_set.mem (v.Ssa_value.id :> int) !derived)
                (Ssa_instr.operands i)
            then
              List.iter
                (fun (r : Ssa_value.t) ->
                  derived := Id_set.add (r.Ssa_value.id :> int) !derived)
                i.Ssa_instr.results
        | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
            go body.Ssa_region.body
        | Ssa_stmt.If { then_; else_; _ } ->
            go then_.Ssa_region.body;
            go else_.Ssa_region.body)
      stmts
  in
  go stmts;
  !derived

(* Is the loop a candidate of the shape above, with every condition checked? *)
let recognize ~policy ranges (s : Ssa_region.t Ssa_stmt.t) :
    (shape, refusal) result =
  match s with
  | Ssa_stmt.For { lo; hi; step; inits; results = _; body } -> (
      match (body.Ssa_region.params, inits, body.Ssa_region.yields) with
      | [ iv; effect_param ], [ _ ], [ _ ] -> (
          if not (Int64.equal step 1L) then Error Shape
          else
            match Ssa_range.trips ranges ~lo ~hi ~step with
            | Ssa_range.Exactly trips -> (
                let lo_c, hi_c =
                  match
                    (Ssa_range.range ranges lo, Ssa_range.range ranges hi)
                  with
                  | Ssa_range.Range l, Ssa_range.Range h -> (l.lo, h.lo)
                  | _ -> (0L, 0L)
                in
                let summary = Ssa_effects.of_region body in
                if summary.Ssa_effects.may_fail then Error May_fail
                else if summary.Ssa_effects.meter || summary.Ssa_effects.locals
                then Error Meter_or_locals
                else
                  let stmts = body.Ssa_region.body in
                  let sums, others =
                    List.partition
                      (function Ssa_stmt.Ordered_sum _ -> true | _ -> false)
                      stmts
                  in
                  match sums with
                  | [ Ssa_stmt.Ordered_sum sum ] -> (
                      let only_instrs =
                        List.for_all
                          (function Ssa_stmt.Instr _ -> true | _ -> false)
                          others
                      in
                      if not only_instrs then Error Shape
                      else
                        let rec split pre = function
                          | (Ssa_stmt.Ordered_sum _ as x) :: post ->
                              Some (List.rev pre, x, post)
                          | x :: rest -> split (x :: pre) rest
                          | [] -> None
                        in
                        match split [] stmts with
                        | None -> Error Shape
                        | Some (pre, _, post) ->
                            let pre_pure =
                              List.for_all
                                (function
                                  | Ssa_stmt.Instr i ->
                                      not (Ssa_op.effectful i.Ssa_instr.op)
                                  | _ -> false)
                                pre
                            in
                            if not pre_pure then Error Shape
                            else
                              let inner = defs_of_stmts Id_set.empty stmts in
                              let outside (v : Ssa_value.t) =
                                not (Id_set.mem (v.Ssa_value.id :> int) inner)
                              in
                              let constant (v : Ssa_value.t) =
                                match Ssa_range.range ranges v with
                                | Ssa_range.Range r -> Int64.equal r.lo r.hi
                                | Ssa_range.Empty -> false
                              in
                              let invariant v = outside v || constant v in
                              if not (invariant sum.lo && invariant sum.hi) then
                                Error Sum_bounds_vary
                              else
                                let derived = derived_from iv stmts in
                                let is_iv (v : Ssa_value.t) =
                                  Ssa_value.equal v iv
                                in
                                let depends (v : Ssa_value.t) =
                                  Id_set.mem (v.Ssa_value.id :> int) derived
                                in
                                let stores =
                                  List.filter_map
                                    (function
                                      | Ssa_stmt.Instr
                                          {
                                            Ssa_instr.op =
                                              Ssa_op.Store { at; buffer; _ };
                                            _;
                                          } ->
                                          Some (buffer, at)
                                      | _ -> None)
                                    post
                                in
                                let independent (_, at) =
                                  match at with
                                  | Ssa_access.Coord c ->
                                      let components = Expr.Coord.to_list c in
                                      List.exists is_iv components
                                      && List.for_all
                                           (fun v -> is_iv v || not (depends v))
                                           components
                                  | Ssa_access.Flat _ -> false
                                in
                                if
                                  stores = []
                                  || not (List.for_all independent stores)
                                then Error Stores_not_independent
                                else
                                  let reads =
                                    Ssa_id.Buffer.Set.elements
                                      summary.Ssa_effects.reads
                                  in
                                  let clash =
                                    List.exists
                                      (fun b ->
                                        Ssa_effects.may_write policy summary b)
                                      reads
                                  in
                                  if clash then Error Reads_written_buffer
                                  else if
                                    Int64.compare (Int64.add hi_c 64L)
                                      Ssa_const.index_max
                                    > 0
                                  then Error Domain_end
                                  else
                                    (* the reads the reduction repeats for every
                                       output: loads whose operands do not
                                       depend on the output *)
                                    let shared =
                                      List.length
                                        (List.filter
                                           (function
                                             | Ssa_stmt.Instr
                                                 {
                                                   Ssa_instr.op =
                                                     ( Ssa_op.Load_in_bounds _
                                                     | Ssa_op.Load _ ) as op;
                                                   _;
                                                 } ->
                                                 not
                                                   (List.exists depends
                                                      (Ssa_op.operands op))
                                             | _ -> false)
                                           sum.body.Ssa_region.body)
                                    in
                                    Ok
                                      {
                                        lo = lo_c;
                                        hi = hi_c;
                                        trips;
                                        iv;
                                        effect_param;
                                        pre;
                                        sum_lo = sum.lo;
                                        sum_hi = sum.hi;
                                        seed = sum.seed;
                                        sum_results = sum.results;
                                        sum_body = sum.body;
                                        post;
                                        shared_reads = shared;
                                      })
                  | _ -> Error Shape)
            | Ssa_range.Zero | Ssa_range.At_least_one | Ssa_range.Unknown ->
                Error Trips_unknown)
      | _ -> Error Carries_values)
  | Ssa_stmt.Instr _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _ -> Error Shape

(* The group size for a candidate: the largest of 2, 4 and 8 that the trip count
   allows and the register estimate (G accumulators, G varying reads, the shared
   ones) keeps within sixteen values, if it saves any read at all. Four is one
   candidate, not the answer. *)
let choose (shape : shape) =
  if shape.shared_reads = 0 then Error Unprofitable
  else
    let pressure g = (2 * g) + shape.shared_reads in
    let sizes = [ 8; 4; 2 ] in
    match
      List.find_opt
        (fun g ->
          Int64.compare shape.trips (Int64.of_int g) >= 0 && pressure g <= 16)
        sizes
    with
    | Some g -> Ok g
    | None -> Error Too_few_trips

type group = Auto | Fixed of int

let decide ~group shape =
  match group with
  | Auto -> choose shape
  | Fixed g ->
      if Int64.compare shape.trips (Int64.of_int g) < 0 then Error Too_few_trips
      else if shape.shared_reads = 0 then Error Unprofitable
      else Ok g

(* Every loop of the program and what the pass would do with it. *)
let analyze ~policy ~group (p : Ssa_program.t) =
  let ranges = Ssa_range.analyze p in
  let out = ref [] in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt s =
    (match s with
    | Ssa_stmt.For { body; _ } -> (
        match recognize ~policy ranges s with
        | Ok shape ->
            out :=
              {
                region = body.Ssa_region.id;
                trips = shape.trips;
                shared_reads = shape.shared_reads;
                decision = decide ~group shape;
              }
              :: !out
        | Error Trips_unknown | Error Carries_values | Error Shape -> ()
        | Error r ->
            out :=
              {
                region = body.Ssa_region.id;
                trips = 0L;
                shared_reads = 0;
                decision = Error r;
              }
              :: !out)
    | Ssa_stmt.Instr _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _ -> ());
    match s with
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } -> region body
    | Ssa_stmt.If { then_; else_; _ } ->
        region then_;
        region else_
    | Ssa_stmt.Instr _ -> ()
  in
  region p.Ssa_program.entry;
  List.rev !out

(* ---- the transformation ---------------------------------------------------- *)

let instr ?token ~results op =
  Ssa_stmt.Instr { Ssa_instr.results; op; token; origin = Ssa_origin.Unknown }

let const t ty c =
  let v = Ssa_rewrite.fresh t ty in
  (v, instr ~results:[ v ] (Ssa_op.Const c))

let index_ty = Ssa_type.Scalar Ssa_type.Index

let build t ~g (shape : shape) (original : Ssa_region.t Ssa_stmt.t) ~inits
    ~results =
  let q = Int64.div shape.trips (Int64.of_int g) in
  let full_hi = Int64.add shape.lo (Int64.mul q (Int64.of_int g)) in
  let lo_v, lo_i = const t index_ty (Ssa_const.Index shape.lo) in
  let fhi_v, fhi_i = const t index_ty (Ssa_const.Index full_hi) in
  (* the full-group loop's own region *)
  let iv' = Ssa_rewrite.fresh t index_ty in
  let eff' = Ssa_rewrite.fresh t Ssa_type.Effect in
  let kiv' = Ssa_rewrite.fresh t index_ty in
  let jam_eff_param = Ssa_rewrite.fresh t Ssa_type.Effect in
  let seed_ty = shape.seed.Ssa_value.ty in
  let acc_params = List.init g (fun _ -> Ssa_rewrite.fresh t seed_ty) in
  let out_effect = Ssa_rewrite.fresh t Ssa_type.Effect in
  let sums = List.init g (fun _ -> Ssa_rewrite.fresh t seed_ty) in
  let sum_value = List.hd shape.sum_results in
  let sum_effect = List.nth shape.sum_results 1 in
  let sum_params = shape.sum_body.Ssa_region.params in
  let kiv, sum_effect_param =
    match sum_params with
    | [ k; e ] -> (k, e)
    | _ -> invalid_arg "Ssa_opt_block: the reduction's parameters"
  in
  let term_value = List.hd shape.sum_body.Ssa_region.yields in
  let term_effect = List.nth shape.sum_body.Ssa_region.yields 1 in
  (* clone j: the original body with its output index moved by j *)
  let pres = ref []
  and terms = ref []
  and seeds = ref []
  and updates = ref [] in
  let clones =
    List.init g (fun j ->
        let w_stmts, w =
          if j = 0 then ([], iv')
          else
            let jv, ji = const t index_ty (Ssa_const.Index (Int64.of_int j)) in
            let w = Ssa_rewrite.fresh t index_ty in
            ( [ ji; instr ~results:[ w ] (Ssa_op.Index_add_in_domain (iv', jv)) ],
              w )
        in
        (w_stmts, Ssa_clone.create t ~subst:[ (shape.iv, w) ]))
  in
  let cur = ref jam_eff_param in
  let term_stmts_per_clone =
    List.mapi
      (fun j (w_stmts, cl) ->
        pres := !pres @ w_stmts @ Ssa_clone.stmts cl shape.pre;
        seeds := !seeds @ [ Ssa_clone.value cl shape.seed ];
        (* the reduction's own index is shared by every clone *)
        Hashtbl.replace cl.Ssa_clone.renaming (kiv.Ssa_value.id :> int) kiv';
        Hashtbl.replace cl.Ssa_clone.renaming
          (sum_effect_param.Ssa_value.id :> int)
          !cur;
        let body = Ssa_clone.stmts cl shape.sum_body.Ssa_region.body in
        let term = Ssa_clone.value cl term_value in
        cur := Ssa_clone.value cl term_effect;
        let acc = List.nth acc_params j in
        let next = Ssa_rewrite.fresh t seed_ty in
        updates :=
          !updates
          @ [
              ( next,
                instr ~results:[ next ]
                  (Ssa_op.Float_binary (Expr.Value.Add, acc, term)) );
            ];
        body)
      clones
  in
  terms := List.concat term_stmts_per_clone;
  let jam_yields_effect = !cur in
  let jam_body =
    {
      Ssa_region.id = Ssa_clone.fresh_region (Ssa_clone.create t ~subst:[]);
      params = (kiv' :: acc_params) @ [ jam_eff_param ];
      body = !terms @ List.map snd !updates;
      yields = List.map fst !updates @ [ jam_yields_effect ];
    }
  in
  let sum_lo = Ssa_clone.value (snd (List.hd clones)) shape.sum_lo in
  let sum_hi = Ssa_clone.value (snd (List.hd clones)) shape.sum_hi in
  let jam =
    Ssa_stmt.For
      {
        lo = sum_lo;
        hi = sum_hi;
        step = 1L;
        inits = !seeds @ [ eff' ];
        results = sums @ [ out_effect ];
        body = jam_body;
      }
  in
  (* the stores, each after the reduction, in group order along the chain *)
  let chain = ref out_effect in
  let posts =
    List.concat
      (List.mapi
         (fun j (_, cl) ->
           Hashtbl.replace cl.Ssa_clone.renaming
             (sum_value.Ssa_value.id :> int)
             (List.nth sums j);
           Hashtbl.replace cl.Ssa_clone.renaming
             (sum_effect.Ssa_value.id :> int)
             !chain;
           let stmts = Ssa_clone.stmts cl shape.post in
           List.iter
             (function
               | Ssa_stmt.Instr i -> (
                   match (i.Ssa_instr.token, List.rev i.Ssa_instr.results) with
                   | Some _, out :: _ -> chain := out
                   | _ -> ())
               | _ -> ())
             stmts;
           stmts)
         clones)
  in
  let full_body =
    {
      Ssa_region.id = Ssa_clone.fresh_region (Ssa_clone.create t ~subst:[]);
      params = [ iv'; eff' ];
      body = !pres @ [ jam ] @ posts;
      yields = [ !chain ];
    }
  in
  let e_full = Ssa_rewrite.fresh t Ssa_type.Effect in
  let full =
    Ssa_stmt.For
      {
        lo = lo_v;
        hi = fhi_v;
        step = Int64.of_int g;
        inits;
        results = [ e_full ];
        body = full_body;
      }
  in
  let tail =
    if Int64.equal full_hi shape.hi then None
    else
      let thi_v, thi_i = const t index_ty (Ssa_const.Index shape.hi) in
      let tlo_v, tlo_i = const t index_ty (Ssa_const.Index full_hi) in
      let cl = Ssa_clone.create t ~subst:[] in
      let body =
        match original with
        | Ssa_stmt.For f -> Ssa_clone.region cl f.body
        | _ -> invalid_arg "Ssa_opt_block: the original is a loop"
      in
      let e_tail = Ssa_rewrite.fresh t Ssa_type.Effect in
      Some
        ( [ tlo_i; thi_i ],
          Ssa_stmt.For
            {
              lo = tlo_v;
              hi = thi_v;
              step = 1L;
              inits = [ e_full ];
              results = [ e_tail ];
              body;
            },
          e_tail )
  in
  let final_effect, tail_stmts =
    match tail with
    | None -> (e_full, [])
    | Some (consts, loop, e) -> (e, consts @ [ loop ])
  in
  (match results with
  | [ r ] -> Ssa_rewrite.alias t ~from:r ~to_:final_effect
  | _ -> invalid_arg "Ssa_opt_block: the loop's results");
  [ lo_i; fhi_i; full ] @ tail_stmts

let pass ~policy ~group (p : Ssa_program.t) =
  let ranges = Ssa_range.analyze p in
  let rule t (s : Ssa_region.t Ssa_stmt.t) =
    match s with
    | Ssa_stmt.For { inits; results; _ } -> (
        match recognize ~policy ranges s with
        | Ok shape -> (
            match decide ~group shape with
            | Ok g -> build t ~g shape s ~inits ~results
            | Error _ -> [ s ])
        | Error _ -> [ s ])
    | Ssa_stmt.Instr _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _ -> [ s ]
  in
  Ssa_rewrite.program rule p
