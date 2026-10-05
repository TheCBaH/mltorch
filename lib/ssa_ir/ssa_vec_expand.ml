(* Lane values are kept beside the program: a vector or mask value maps to its
   per-lane scalar values, and a scalar value an extract names is renamed to the
   lane it reads. Every other value keeps its identity. Fresh ids continue the
   program's own counters. *)

type t = {
  lanes : (int, Ssa_value.t array) Hashtbl.t;
  alias : (int, Ssa_value.t) Hashtbl.t;
  mutable next_value : Ssa_id.Value.Next.t;
}

let fresh t ty =
  let id, next = Ssa_id.Value.Next.alloc t.next_value in
  t.next_value <- next;
  { Ssa_value.id; ty }

let lane_type : Ssa_type.t -> Ssa_type.t = function
  | Ssa_type.Vec (s, _) -> Ssa_type.Scalar s
  | Ssa_type.Mask _ -> Ssa_type.Scalar Ssa_type.Pred
  | (Ssa_type.Effect | Ssa_type.Local | Ssa_type.Scalar _) as ty -> ty

let width : Ssa_type.t -> int option = function
  | Ssa_type.Vec (_, l) | Ssa_type.Mask l -> Some (Ssa_type.Lanes.to_int l)
  | Ssa_type.Effect | Ssa_type.Local | Ssa_type.Scalar _ -> None

let is_vector (v : Ssa_value.t) = width v.Ssa_value.ty <> None

let resolve t (v : Ssa_value.t) =
  match Hashtbl.find_opt t.alias (v.Ssa_value.id :> int) with
  | Some w -> w
  | None -> v

let lanes_of t (v : Ssa_value.t) =
  match Hashtbl.find_opt t.lanes (v.Ssa_value.id :> int) with
  | Some a -> a
  | None -> invalid_arg "Ssa_vec_expand: a vector with no lanes"

let bind t (v : Ssa_value.t) a =
  Hashtbl.replace t.lanes (v.Ssa_value.id :> int) a

(* A vector definition's fresh lane values, registered; a scalar's own value. *)
let define_lanes t (v : Ssa_value.t) =
  match width v.Ssa_value.ty with
  | Some n ->
      let a = Array.init n (fun _ -> fresh t (lane_type v.Ssa_value.ty)) in
      bind t v a;
      Array.to_list a
  | None -> [ v ]

(* The values standing for a use of [v] where a scalar program carries it. *)
let flatten t (v : Ssa_value.t) =
  if is_vector v then Array.to_list (lanes_of t v) else [ resolve t v ]

let instr ?token ~results op =
  Ssa_stmt.Instr { Ssa_instr.results; op; token; origin = Ssa_origin.Unknown }

let const t c =
  let v = fresh t (Ssa_const.ty c) in
  (v, instr ~results:[ v ] (Ssa_op.Const c))

(* The coordinate of lane [k]: each axis moved by [k * step], by a pure sum the
   verifier re-proves. *)
let lane_coord t (at : Ssa_value.t Expr.Coord.t) (steps : int64 Expr.Coord.t) k
    =
  let pre = ref [] in
  let c =
    Expr.Coord.mapi
      (fun axis (v : Ssa_value.t) ->
        let v = resolve t v in
        let delta = Int64.mul (Int64.of_int k) (Expr.Coord.get steps axis) in
        if Int64.equal delta 0L then v
        else
          let d, d_i = const t (Ssa_const.Index delta) in
          let sum = fresh t v.Ssa_value.ty in
          pre :=
            instr ~results:[ sum ] (Ssa_op.Index_add_in_domain (v, d))
            :: d_i :: !pre;
          sum)
      at
  in
  (c, List.rev !pre)

let expand_instr t (i : Ssa_instr.t) : Ssa_region.t Ssa_stmt.t list =
  let token = Option.map (resolve t) i.Ssa_instr.token in
  let result = match i.Ssa_instr.results with r :: _ -> Some r | [] -> None in
  let last_effect =
    List.nth i.Ssa_instr.results (List.length i.Ssa_instr.results - 1)
  in
  (* a chain of effectful scalar instructions, ending in the original effect *)
  let chain n step =
    let cur = ref (Option.get token) in
    let out = ref [] in
    for k = 0 to n - 1 do
      let next = if k = n - 1 then last_effect else fresh t Ssa_type.Effect in
      let pre, make = step k in
      out := List.rev_append pre !out;
      out := make ~token:!cur ~chained:next :: !out;
      cur := next
    done;
    List.rev !out
  in
  match i.Ssa_instr.op with
  | Ssa_op.Lanewise inner ->
      let lanes = define_lanes t (Option.get result) in
      List.mapi
        (fun k lane ->
          instr ~results:[ lane ]
            (Ssa_op.map_operands (fun v -> (lanes_of t v).(k)) inner))
        lanes
  | Ssa_op.Mark_lanes { mark; lanes } ->
      chain (Ssa_type.Lanes.to_int lanes) (fun _ ->
          ( [],
            fun ~token ~chained ->
              instr ~token ~results:[ chained ] (Ssa_op.Mark mark) ))
  | Ssa_op.Vec_extract { lane; vector } ->
      Hashtbl.replace t.alias
        ((Option.get result).Ssa_value.id :> int)
        (lanes_of t vector).(Ssa_type.Lane.to_int lane);
      []
  | Ssa_op.Vec_insert { lane; vector; element } ->
      let a = Array.copy (lanes_of t vector) in
      a.(Ssa_type.Lane.to_int lane) <- resolve t element;
      bind t (Option.get result) a;
      []
  | Ssa_op.Vec_iota { base; step; lanes = _ } ->
      let out = define_lanes t (Option.get result) in
      let b64 = fresh t (Ssa_type.Scalar Ssa_type.I64) in
      let widen =
        instr ~results:[ b64 ]
          (Ssa_op.Convert (Ssa_op.Convert.Index_to_i64, resolve t base))
      in
      widen
      :: List.concat
           (List.mapi
              (fun k lane ->
                let d, d_i =
                  const t (Ssa_const.I64 (Int64.mul (Int64.of_int k) step))
                in
                let sum = fresh t (Ssa_type.Scalar Ssa_type.I64) in
                [
                  d_i;
                  instr ~results:[ sum ]
                    (Ssa_op.I64_arith (Ssa_op.I64_op.Add, b64, d));
                  instr ~results:[ lane ]
                    (Ssa_op.Convert (Ssa_op.Convert.I64_to_f64, sum));
                ])
              out)
  | Ssa_op.Vec_splat { element; lanes } ->
      bind t (Option.get result)
        (Array.make (Ssa_type.Lanes.to_int lanes) (resolve t element));
      []
  | Ssa_op.Vec_load { buffer; at; steps; decode; lanes } ->
      let out = Array.of_list (define_lanes t (Option.get result)) in
      chain (Ssa_type.Lanes.to_int lanes) (fun k ->
          let c, pre = lane_coord t at steps k in
          ( pre,
            fun ~token ~chained ->
              instr ~token
                ~results:[ out.(k); chained ]
                (Ssa_op.Load_in_bounds
                   { buffer; at = Ssa_access.Coord c; decode }) ))
  | Ssa_op.Vec_store { buffer; at; steps; encode; value; lanes } ->
      let values = lanes_of t value in
      chain (Ssa_type.Lanes.to_int lanes) (fun k ->
          let c, pre = lane_coord t at steps k in
          ( pre,
            fun ~token ~chained ->
              instr ~token ~results:[ chained ]
                (Ssa_op.Store
                   {
                     buffer;
                     at = Ssa_access.Coord c;
                     encode;
                     value = values.(k);
                   }) ))
  | ( Ssa_op.Check_access _ | Ssa_op.Check_gather _ | Ssa_op.Check_local _
    | Ssa_op.Check_scan _ | Ssa_op.Const _ | Ssa_op.Convert _
    | Ssa_op.Float_binary _ | Ssa_op.Float_compare _ | Ssa_op.Float_max _
    | Ssa_op.Float_to_i64 _ | Ssa_op.Float_unary _ | Ssa_op.I64_arith _
    | Ssa_op.I64_compare _ | Ssa_op.I64_div _ | Ssa_op.Index_add _
    | Ssa_op.Index_add_in_domain _ | Ssa_op.Index_ceil_div _
    | Ssa_op.Index_clamp_low _ | Ssa_op.Index_compare _
    | Ssa_op.Index_floor_div _ | Ssa_op.Index_max _ | Ssa_op.Index_min _
    | Ssa_op.Index_of_i64 _ | Ssa_op.Index_scale _
    | Ssa_op.Index_scale_in_domain _ | Ssa_op.Load _ | Ssa_op.Load_in_bounds _
    | Ssa_op.Local_alloc _ | Ssa_op.Local_read _ | Ssa_op.Local_write _
    | Ssa_op.Mark _ | Ssa_op.Meter_charge | Ssa_op.Meter_release _
    | Ssa_op.Meter_reserve _ | Ssa_op.Meter_reset | Ssa_op.Pool_better _
    | Ssa_op.Pred_not _ | Ssa_op.Pred_or _ | Ssa_op.Select _ | Ssa_op.Store _ )
    as op ->
      [
        Ssa_stmt.Instr
          { i with Ssa_instr.op = Ssa_op.map_operands (resolve t) op; token };
      ]

let rec stmt t : Ssa_region.t Ssa_stmt.t -> Ssa_region.t Ssa_stmt.t list =
  function
  | Ssa_stmt.Instr i -> expand_instr t i
  | Ssa_stmt.For f ->
      let lo = resolve t f.lo and hi = resolve t f.hi in
      let inits = List.concat_map (flatten t) f.inits in
      let params =
        match f.body.Ssa_region.params with
        | iv :: carried -> iv :: List.concat_map (define_lanes t) carried
        | [] -> invalid_arg "Ssa_vec_expand: a loop without an induction value"
      in
      let body = region t f.body ~params in
      let results = List.concat_map (define_lanes t) f.results in
      [ Ssa_stmt.For { f with lo; hi; inits; results; body } ]
  | Ssa_stmt.If f ->
      let cond = resolve t f.cond in
      let then_ = region t f.then_ ~params:[] in
      let else_ = region t f.else_ ~params:[] in
      let results = List.concat_map (define_lanes t) f.results in
      [ Ssa_stmt.If { cond; results; then_; else_ } ]
  | Ssa_stmt.Ordered_sum f -> (
      let lo = resolve t f.lo and hi = resolve t f.hi in
      let token = resolve t f.token in
      let iv, eff =
        match f.body.Ssa_region.params with
        | [ iv; eff ] -> (iv, eff)
        | _ -> invalid_arg "Ssa_vec_expand: a sum's parameters"
      in
      let seed = f.seed in
      match width seed.Ssa_value.ty with
      | None ->
          let seed = resolve t seed in
          let body = region t f.body ~params:[ iv; eff ] in
          [ Ssa_stmt.Ordered_sum { f with lo; hi; seed; token; body } ]
      | Some n ->
          (* one accumulator per lane, each the left fold of its own terms *)
          let seeds = Array.to_list (lanes_of t seed) in
          let lane_ty = lane_type seed.Ssa_value.ty in
          let accs = List.init n (fun _ -> fresh t lane_ty) in
          let results =
            List.concat_map (define_lanes t) [ List.hd f.results ]
          in
          let effect_result = List.nth f.results 1 in
          let body_stmts = List.concat_map (stmt t) f.body.Ssa_region.body in
          let term, term_effect =
            match f.body.Ssa_region.yields with
            | [ term; e ] -> (term, e)
            | _ -> invalid_arg "Ssa_vec_expand: a sum's yields"
          in
          let terms = lanes_of t term in
          let sums = List.init n (fun _ -> fresh t lane_ty) in
          let adds =
            List.mapi
              (fun k acc ->
                instr
                  ~results:[ List.nth sums k ]
                  (Ssa_op.Float_binary (Expr.Value.Add, acc, terms.(k))))
              accs
          in
          let body =
            {
              Ssa_region.id = f.body.Ssa_region.id;
              params = (iv :: accs) @ [ eff ];
              body = body_stmts @ adds;
              yields = sums @ [ resolve t term_effect ];
            }
          in
          [
            Ssa_stmt.For
              {
                lo;
                hi;
                step = 1L;
                inits = seeds @ [ token ];
                results = results @ [ effect_result ];
                body;
              };
          ])

and region t (r : Ssa_region.t) ~params =
  let body = List.concat_map (stmt t) r.Ssa_region.body in
  {
    Ssa_region.id = r.Ssa_region.id;
    params;
    body;
    yields = List.concat_map (flatten t) r.Ssa_region.yields;
  }

let program (p : Ssa_program.t) =
  (match Err.payload (Ssa_verify.check p) with
  | Ok () -> ()
  | Error e -> invalid_arg (Fmt.str "Ssa_vec_expand: %a" Ssa_verify.pp_error e));
  let t =
    {
      lanes = Hashtbl.create 64;
      alias = Hashtbl.create 16;
      next_value = p.Ssa_program.next_value;
    }
  in
  let entry =
    region t p.Ssa_program.entry ~params:p.Ssa_program.entry.Ssa_region.params
  in
  {
    p with
    Ssa_program.entry;
    revision = Ssa_id.Revision.succ p.Ssa_program.revision;
    next_value = t.next_value;
  }
