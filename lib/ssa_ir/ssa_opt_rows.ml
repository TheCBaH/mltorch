module Id_set = Ssa_opt_block.Id_set

type inner = {
  lo : Ssa_value.t;
  hi : Ssa_value.t;
  step : int64;
  iv : Ssa_value.t;
  pre : Ssa_region.t Ssa_stmt.t list;
  sum_lo : Ssa_value.t;
  sum_hi : Ssa_value.t;
  seed : Ssa_value.t;
  sum_results : Ssa_value.t list;
  sum_body : Ssa_region.t;
  post : Ssa_region.t Ssa_stmt.t list;
}

type shape = {
  trips : int64;
  lo : int64;
  hi : int64;
  h : Ssa_value.t;
  setup : Ssa_region.t Ssa_stmt.t list;
  inner : inner;
}

let pure_instrs stmts =
  List.for_all
    (function
      | Ssa_stmt.Instr i -> not (Ssa_op.effectful i.Ssa_instr.op) | _ -> false)
    stmts

let instrs_only stmts =
  List.for_all (function Ssa_stmt.Instr _ -> true | _ -> false) stmts

type refusal =
  | Bounds_vary
  | May_fail
  | Meter_or_locals
  | Reads_written_buffer
  | Shape
  | Stores_not_independent
  | Too_few_rows
  | Trips_unknown

let refusal_name = function
  | Bounds_vary -> "bounds vary with the row"
  | May_fail -> "the body can fail"
  | Meter_or_locals -> "the body touches the meter or a scratch object"
  | Reads_written_buffer -> "the body reads a buffer it may write"
  | Shape -> "not setup, one vector loop"
  | Stores_not_independent -> "stores are not per row"
  | Too_few_rows -> "fewer rows than one block"
  | Trips_unknown -> "the row count is not a constant"

(* The loop and what it holds, or the condition that failed. *)
let recognize ~policy ranges (s : Ssa_region.t Ssa_stmt.t) :
    (shape, refusal) result =
  let ( let* ) = Result.bind in
  let need why = function Some x -> Ok x | None -> Error why in
  match s with
  | Ssa_stmt.For { lo; hi; step = 1L; inits = [ _ ]; body; results = _ } -> (
      let* h =
        need Shape
          (match body.Ssa_region.params with [ h; _ ] -> Some h | _ -> None)
      in
      let* trips =
        need Trips_unknown
          (match Ssa_range.trips ranges ~lo ~hi ~step:1L with
          | Ssa_range.Exactly n -> Some n
          | Ssa_range.Zero | Ssa_range.At_least_one | Ssa_range.Unknown -> None)
      in
      let* lo_c, hi_c =
        need Trips_unknown
          (match (Ssa_range.range ranges lo, Ssa_range.range ranges hi) with
          | Ssa_range.Range l, Ssa_range.Range r -> Some (l.lo, r.lo)
          | _ -> None)
      in
      (* [setup], one loop, nothing else *)
      let rec split acc = function
        | (Ssa_stmt.For _ as v) :: [] -> Some (List.rev acc, v)
        | (Ssa_stmt.Instr _ as i) :: rest -> split (i :: acc) rest
        | _ -> None
      in
      let* setup, v = need Shape (split [] body.Ssa_region.body) in
      if not (pure_instrs setup) then Error Shape
      else
        match v with
        | Ssa_stmt.For
            {
              lo = vlo;
              hi = vhi;
              step;
              inits = [ _ ];
              results = [ _ ];
              body = vbody;
            } -> (
            let* iv =
              need Shape
                (match vbody.Ssa_region.params with
                | [ i; _ ] -> Some i
                | _ -> None)
            in
            let summary = Ssa_effects.of_region vbody in
            if not (Ssa_vector_body.has_vector vbody) then Error Shape
            else if summary.Ssa_effects.may_fail then Error May_fail
            else if summary.Ssa_effects.meter || summary.Ssa_effects.locals then
              Error Meter_or_locals
            else
              let rec split3 pre = function
                | (Ssa_stmt.Ordered_sum _ as x) :: post ->
                    Some (List.rev pre, x, post)
                | (Ssa_stmt.Instr _ as x) :: rest -> split3 (x :: pre) rest
                | _ -> None
              in
              let* pre, sum, post =
                need Shape (split3 [] vbody.Ssa_region.body)
              in
              match sum with
              | Ssa_stmt.Ordered_sum
                  {
                    lo = sum_lo;
                    hi = sum_hi;
                    seed;
                    results;
                    body = sum_body;
                    _;
                  } ->
                  if not (pure_instrs pre && instrs_only post) then Error Shape
                  else
                    let derived =
                      Ssa_opt_block.derived_from h (setup @ pre @ post)
                    in
                    let is_h (v : Ssa_value.t) = Ssa_value.equal v h in
                    let depends (v : Ssa_value.t) =
                      Id_set.mem (v.Ssa_value.id :> int) derived
                    in
                    let stores =
                      List.filter_map
                        (function
                          | Ssa_stmt.Instr
                              {
                                Ssa_instr.op =
                                  ( Ssa_op.Store { at = Ssa_access.Coord c; _ }
                                  | Ssa_op.Vec_store { at = c; _ } );
                                _;
                              } ->
                              Some (Expr.Coord.to_list c)
                          | _ -> None)
                        post
                    in
                    let independent components =
                      List.exists is_h components
                      && List.for_all
                           (fun v -> is_h v || not (depends v))
                           components
                    in
                    if
                      depends sum_lo || depends sum_hi || depends vlo
                      || depends vhi
                    then Error Bounds_vary
                    else if
                      Ssa_id.Buffer.Set.exists
                        (fun b -> Ssa_effects.may_write policy summary b)
                        summary.Ssa_effects.reads
                    then Error Reads_written_buffer
                    else if stores = [] || not (List.for_all independent stores)
                    then Error Stores_not_independent
                    else
                      Ok
                        {
                          trips;
                          lo = lo_c;
                          hi = hi_c;
                          h;
                          setup;
                          inner =
                            {
                              lo = vlo;
                              hi = vhi;
                              step;
                              iv;
                              pre;
                              sum_lo;
                              sum_hi;
                              seed;
                              sum_results = results;
                              sum_body;
                              post;
                            };
                        }
              | _ -> Error Shape)
        | _ -> Error Shape)
  | _ -> Error Shape

let index_ty = Ssa_type.Scalar Ssa_type.Index

let build t ~factor (shape : shape) (original : Ssa_region.t Ssa_stmt.t) ~inits
    ~results =
  let q = Int64.div shape.trips (Int64.of_int factor) in
  let full_hi = Int64.add shape.lo (Int64.mul q (Int64.of_int factor)) in
  let lo_v, lo_i = Ssa_opt_block.const t index_ty (Ssa_const.Index shape.lo) in
  let fhi_v, fhi_i = Ssa_opt_block.const t index_ty (Ssa_const.Index full_hi) in
  let h' = Ssa_rewrite.fresh t index_ty in
  let eff' = Ssa_rewrite.fresh t Ssa_type.Effect in
  let v_iv' = Ssa_rewrite.fresh t index_ty in
  let v_eff' = Ssa_rewrite.fresh t Ssa_type.Effect in
  (* row r: the row index moved by r, its setup, and the loop's own index shared *)
  let rows =
    List.init factor (fun r ->
        let w_stmts, w =
          if r = 0 then ([], h')
          else
            let rv, ri =
              Ssa_opt_block.const t index_ty (Ssa_const.Index (Int64.of_int r))
            in
            let w = Ssa_rewrite.fresh t index_ty in
            ( [
                ri;
                Ssa_opt_block.instr ~results:[ w ]
                  (Ssa_op.Index_add_in_domain (h', rv));
              ],
              w )
        in
        let cl = Ssa_clone.create t ~subst:[ (shape.h, w) ] in
        Hashtbl.replace cl.Ssa_clone.renaming
          (shape.inner.iv.Ssa_value.id :> int)
          v_iv';
        let setup = Ssa_clone.stmts cl shape.setup in
        (w_stmts @ setup, cl))
  in
  let inner = shape.inner in
  let body, chain =
    Ssa_opt_block.jam t
      ~clones:(List.map (fun (_, cl) -> ([], cl)) rows)
      ~pre:inner.pre ~sum_lo:inner.sum_lo ~sum_hi:inner.sum_hi ~seed:inner.seed
      ~sum_results:inner.sum_results ~sum_body:inner.sum_body ~post:inner.post
      ~eff_in:v_eff'
  in
  let first = snd (List.hd rows) in
  let v_results = Ssa_rewrite.fresh t Ssa_type.Effect in
  let vloop =
    Ssa_stmt.For
      {
        lo = Ssa_clone.value first inner.lo;
        hi = Ssa_clone.value first inner.hi;
        step = inner.step;
        inits = [ eff' ];
        results = [ v_results ];
        body =
          {
            Ssa_region.id =
              Ssa_clone.fresh_region (Ssa_clone.create t ~subst:[]);
            params = [ v_iv'; v_eff' ];
            body;
            yields = [ chain ];
          };
      }
  in
  let outer_body =
    {
      Ssa_region.id = Ssa_clone.fresh_region (Ssa_clone.create t ~subst:[]);
      params = [ h'; eff' ];
      body = List.concat_map fst rows @ [ vloop ];
      yields = [ v_results ];
    }
  in
  let e_full = Ssa_rewrite.fresh t Ssa_type.Effect in
  let full =
    Ssa_stmt.For
      {
        lo = lo_v;
        hi = fhi_v;
        step = Int64.of_int factor;
        inits;
        results = [ e_full ];
        body = outer_body;
      }
  in
  let rest, final_effect =
    if Int64.equal full_hi shape.hi then ([], e_full)
    else
      let thi_v, thi_i =
        Ssa_opt_block.const t index_ty (Ssa_const.Index shape.hi)
      in
      let tlo_v, tlo_i =
        Ssa_opt_block.const t index_ty (Ssa_const.Index full_hi)
      in
      let cl = Ssa_clone.create t ~subst:[] in
      let body =
        match original with
        | Ssa_stmt.For f -> Ssa_clone.region cl f.body
        | _ -> invalid_arg "Ssa_opt_rows: the original is a loop"
      in
      let e_rest = Ssa_rewrite.fresh t Ssa_type.Effect in
      ( [
          tlo_i;
          thi_i;
          Ssa_stmt.For
            {
              lo = tlo_v;
              hi = thi_v;
              step = 1L;
              inits = [ e_full ];
              results = [ e_rest ];
              body;
            };
        ],
        e_rest )
  in
  (match results with
  | [ r ] -> Ssa_rewrite.alias t ~from:r ~to_:final_effect
  | _ -> invalid_arg "Ssa_opt_rows: the loop's results");
  [ lo_i; fhi_i; full ] @ rest

let program ~rows ~policy (p : Ssa_program.t) =
  let blocked = ref 0 in
  if rows < 2 then (p, 0)
  else
    let ranges = Ssa_range.analyze p in
    let rule t (s : Ssa_region.t Ssa_stmt.t) =
      match s with
      | Ssa_stmt.For { inits; results; _ } -> (
          match recognize ~policy ranges s with
          | Ok shape ->
              (* a body that carries several accumulators leaves registers for
                 fewer rows; this one carries one *)
              let factor = Stdlib.min (Int64.to_int shape.trips) rows in
              if factor < 2 then [ s ]
              else (
                incr blocked;
                build t ~factor shape s ~inits ~results)
          | Error _ -> [ s ])
      | Ssa_stmt.Instr _ | Ssa_stmt.If _ | Ssa_stmt.Ordered_sum _ -> [ s ]
    in
    let q, _ = Ssa_rewrite.program rule p in
    (q, !blocked)

(* Every loop that is a candidate by its outer shape and what became of it. *)
let analyze ~rows ~policy (p : Ssa_program.t) =
  let ranges = Ssa_range.analyze p in
  let out = ref [] in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt (s : Ssa_region.t Ssa_stmt.t) =
    (match s with
    | Ssa_stmt.For { body; _ } -> (
        match recognize ~policy ranges s with
        | Ok shape ->
            let factor = Stdlib.min (Int64.to_int shape.trips) rows in
            out :=
              ( body.Ssa_region.id,
                if factor < 2 then Error Too_few_rows else Ok factor )
              :: !out
        | Error Shape -> ()
        | Error r -> out := (body.Ssa_region.id, Error r) :: !out)
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
