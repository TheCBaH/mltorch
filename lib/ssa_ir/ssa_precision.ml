type t = {
  env : (int, Ssa_value.t) Hashtbl.t;
  mutable next_value : Ssa_id.Value.Next.t;
}

let f32 = Ssa_type.Scalar Ssa_type.F32
let f64 = Ssa_type.Scalar Ssa_type.F64

let fresh t ty =
  let id, next = Ssa_id.Value.Next.alloc t.next_value in
  t.next_value <- next;
  { Ssa_value.id; ty }

let get t (v : Ssa_value.t) =
  match Hashtbl.find_opt t.env (v.Ssa_value.id :> int) with
  | Some w -> w
  | None -> v

let set t (v : Ssa_value.t) w = Hashtbl.replace t.env (v.Ssa_value.id :> int) w

(* The definition standing for [v] in the rewritten program: a fresh value when
   its type changes, the value itself otherwise. *)
let define t (v : Ssa_value.t) =
  if Ssa_type.equal v.Ssa_value.ty f64 then (
    let w = fresh t f32 in
    set t v w;
    w)
  else v

let instr ?token ~results op =
  Ssa_stmt.Instr { Ssa_instr.results; op; token; origin = Ssa_origin.Unknown }

let convert t c (a : Ssa_value.t) into =
  let r = fresh t into in
  (r, instr ~results:[ r ] (Ssa_op.Convert (c, a)))

let rec stmt t : Ssa_region.t Ssa_stmt.t -> Ssa_region.t Ssa_stmt.t list =
  function
  | Ssa_stmt.Instr i -> instr_ t i
  | Ssa_stmt.For f ->
      let lo = get t f.lo and hi = get t f.hi in
      let inits = List.map (get t) f.inits in
      let params = List.map (define t) f.body.Ssa_region.params in
      let body = region t f.body ~params in
      let results = List.map (define t) f.results in
      [ Ssa_stmt.For { f with lo; hi; inits; results; body } ]
  | Ssa_stmt.If f ->
      let cond = get t f.cond in
      let then_ = region t f.then_ ~params:[] in
      let else_ = region t f.else_ ~params:[] in
      let results = List.map (define t) f.results in
      [ Ssa_stmt.If { cond; results; then_; else_ } ]
  | Ssa_stmt.Ordered_sum f ->
      let lo = get t f.lo and hi = get t f.hi in
      let seed = get t f.seed and token = get t f.token in
      let params = List.map (define t) f.body.Ssa_region.params in
      let body = region t f.body ~params in
      let results = List.map (define t) f.results in
      [ Ssa_stmt.Ordered_sum { lo; hi; seed; token; results; body } ]

and region t (r : Ssa_region.t) ~params =
  let body = List.concat_map (stmt t) r.Ssa_region.body in
  {
    Ssa_region.id = r.Ssa_region.id;
    params;
    body;
    yields = List.map (get t) r.Ssa_region.yields;
  }

and instr_ t (i : Ssa_instr.t) =
  let token = Option.map (get t) i.Ssa_instr.token in
  let keep op results =
    [ instr ?token ~results (Ssa_op.map_operands (get t) op) ]
  in
  let results () = List.map (define t) i.Ssa_instr.results in
  let narrowed_load make =
    (* the load keeps its binary64 decode; its value is narrowed once *)
    match i.Ssa_instr.results with
    | [ value; chain ] when Ssa_type.equal value.Ssa_value.ty f64 ->
        let wide = fresh t f64 in
        let narrow, narrow_i = convert t Ssa_op.Convert.F64_to_f32 wide f32 in
        set t value narrow;
        [ instr ?token ~results:[ wide; chain ] (make ()); narrow_i ]
    | _ -> keep (make ()) i.Ssa_instr.results
  in
  let widened (a : Ssa_value.t) =
    convert t Ssa_op.Convert.F32_to_f64 (get t a) f64
  in
  match i.Ssa_instr.op with
  | Ssa_op.Const (Ssa_const.F64 x) ->
      let r = define t (List.hd i.Ssa_instr.results) in
      [
        instr ~results:[ r ]
          (Ssa_op.Const (Ssa_const.F32 (Ssa_const.round_f32 x)));
      ]
  | Ssa_op.Convert (Ssa_op.Convert.F32_to_f64, a) ->
      (* a binary32 widened to binary64 is the binary32 itself now *)
      set t (List.hd i.Ssa_instr.results) (get t a);
      []
  | Ssa_op.Convert (Ssa_op.Convert.F64_to_f32, a) ->
      set t (List.hd i.Ssa_instr.results) (get t a);
      []
  | Ssa_op.Convert (Ssa_op.Convert.I64_to_f64, a) ->
      let r = define t (List.hd i.Ssa_instr.results) in
      [
        instr ~results:[ r ]
          (Ssa_op.Convert (Ssa_op.Convert.I64_to_f32, get t a));
      ]
  | Ssa_op.Convert (Ssa_op.Convert.Index_to_f64, a) ->
      let r = define t (List.hd i.Ssa_instr.results) in
      let wide, wide_i =
        convert t Ssa_op.Convert.Index_to_i64 (get t a)
          (Ssa_type.Scalar Ssa_type.I64)
      in
      [
        wide_i;
        instr ~results:[ r ] (Ssa_op.Convert (Ssa_op.Convert.I64_to_f32, wide));
      ]
  | Ssa_op.Float_to_i64 a ->
      let wide, wide_i = widened a in
      [
        wide_i;
        instr ~results:i.Ssa_instr.results ?token (Ssa_op.Float_to_i64 wide);
      ]
  | Ssa_op.Load { buffer; at; decode } ->
      narrowed_load (fun () ->
          Ssa_op.map_operands (get t) (Ssa_op.Load { buffer; at; decode }))
  | Ssa_op.Load_in_bounds { buffer; at; decode } ->
      narrowed_load (fun () ->
          Ssa_op.map_operands (get t)
            (Ssa_op.Load_in_bounds { buffer; at; decode }))
  | Ssa_op.Store { buffer; at; encode; value } -> (
      match encode with
      | Ssa_op.Encode.I64 -> keep i.Ssa_instr.op i.Ssa_instr.results
      | Ssa_op.Encode.Bool_nonzero | Ssa_op.Encode.F32_round ->
          let wide, wide_i = widened value in
          [
            wide_i;
            instr ?token ~results:i.Ssa_instr.results
              (Ssa_op.Store
                 {
                   buffer;
                   at = Ssa_access.map (get t) at;
                   encode;
                   value = wide;
                 });
          ])
  | Ssa_op.Local_write { local; at; value } ->
      let wide, wide_i = widened value in
      [
        wide_i;
        instr ?token ~results:i.Ssa_instr.results
          (Ssa_op.Local_write
             { local = get t local; at = get t at; value = wide });
      ]
  | Ssa_op.Local_read { local; at } ->
      let wide = fresh t f64 in
      let chain = List.nth i.Ssa_instr.results 1 in
      let narrow, narrow_i = convert t Ssa_op.Convert.F64_to_f32 wide f32 in
      set t (List.hd i.Ssa_instr.results) narrow;
      [
        instr ?token ~results:[ wide; chain ]
          (Ssa_op.Local_read { local = get t local; at = get t at });
        narrow_i;
      ]
  | Ssa_op.Lanewise _ | Ssa_op.Mark_lanes _ | Ssa_op.Vec_extract _
  | Ssa_op.Vec_insert _ | Ssa_op.Vec_iota _ | Ssa_op.Vec_load _
  | Ssa_op.Vec_splat _ | Ssa_op.Vec_store _ ->
      invalid_arg "Ssa_precision: choose the precision before vectorizing"
  | ( Ssa_op.Check_access _ | Ssa_op.Check_gather _ | Ssa_op.Check_local _
    | Ssa_op.Check_scan _ | Ssa_op.Const _ | Ssa_op.Convert _
    | Ssa_op.Float_binary _ | Ssa_op.Float_compare _ | Ssa_op.Float_fma _
    | Ssa_op.Float_max _ | Ssa_op.Float_unary _ | Ssa_op.I64_arith _
    | Ssa_op.I64_compare _ | Ssa_op.I64_div _ | Ssa_op.Index_add _
    | Ssa_op.Index_add_in_domain _ | Ssa_op.Index_ceil_div _
    | Ssa_op.Index_clamp_low _ | Ssa_op.Index_compare _
    | Ssa_op.Index_floor_div _ | Ssa_op.Index_max _ | Ssa_op.Index_min _
    | Ssa_op.Index_of_i64 _ | Ssa_op.Index_scale _
    | Ssa_op.Index_scale_in_domain _ | Ssa_op.Local_alloc _ | Ssa_op.Mark _
    | Ssa_op.Meter_charge | Ssa_op.Meter_release _ | Ssa_op.Meter_reserve _
    | Ssa_op.Meter_reset | Ssa_op.Pool_better _ | Ssa_op.Pred_not _
    | Ssa_op.Pred_or _ | Ssa_op.Select _ ) as op ->
      keep op (results ())

let to_f32 (p : Ssa_program.t) =
  (match Err.payload (Ssa_verify.check p) with
  | Ok () -> ()
  | Error e -> invalid_arg (Fmt.str "Ssa_precision: %a" Ssa_verify.pp_error e));
  (match Ssa_numerics.admit p with
  | Ok () -> ()
  | Error r ->
      invalid_arg (Fmt.str "Ssa_precision: %a" Ssa_numerics.Refusal.pp r));
  let t = { env = Hashtbl.create 64; next_value = p.Ssa_program.next_value } in
  let entry =
    region t p.Ssa_program.entry ~params:p.Ssa_program.entry.Ssa_region.params
  in
  {
    p with
    Ssa_program.entry;
    revision = Ssa_id.Revision.succ p.Ssa_program.revision;
    next_value = t.next_value;
  }
