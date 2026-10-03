(* Lowering of a vector loop to 128-bit WebAssembly SIMD.

   A logical vector of [lanes] binary64 values is [lanes / 2] registers of two
   lanes each (the "halves"); a vector expression is lowered one half at a time,
   each half a straight-line sequence leaving one [v128]. Vector temporaries live
   in locals, one per half. Nothing here is a physical instruction in the
   vector program: this file is the one place a lane count meets the two lanes of
   an [f64x2].

   Strict means a lane is its scalar iteration: [f64x2] arithmetic is the scalar
   [f64] operation per lane (no fused multiply-add, no reassociation); a
   [Round_f32] is [f32x4.demote_f64x2_zero] then [f64x2.promote_low_f32x4]; the
   maximum is [f64x2.max], which propagates NaN and orders [-0 < +0] like
   [f64.max]; a transcendental, which has no vector instruction, is expanded
   lane by lane through the same scalar callee. *)

open Loop_wasm_ctx
open Loop_wasm_value
module V = Loop_vector

let vn op = I.Numeric op

(* The sixteen bytes of a vector holding [x] in both lanes. *)
let f64_pair_bytes x =
  let bits = Int64.bits_of_float x in
  let one =
    String.init 8 (fun k ->
        Char.chr
          (Int64.to_int (Int64.shift_right_logical bits (8 * k)) land 0xFF))
  in
  one ^ one

type vctx = {
  halves : int;
  temps : (int, int array) Hashtbl.t;
  mutable splats : (float Loop_expr.t * int) list;
      (** a splatted scalar expression and the local holding its vector *)
}

module F = Loop_vector_facts

(* A splat computed once, before the loop, is one that no enclosing inner loop's
   variable reaches; any other is evaluated where it is used. *)
let hoistable inner e =
  not (List.exists (fun v -> F.expr_depends v Loop_temp.Set.empty e) inner)

let temp_local st vc t h =
  let regs =
    match Hashtbl.find_opt vc.temps (V.Temp.to_int t) with
    | Some r -> r
    | None ->
        let r = Array.init vc.halves (fun _ -> fresh st Wasm_type.V128) in
        Hashtbl.add vc.temps (V.Temp.to_int t) r;
        r
  in
  regs.(h)

let splat_instrs st vc e =
  match List.find_opt (fun (e', _) -> e' == e) vc.splats with
  | Some (_, l) -> [ get l ]
  | None -> num st e @ [ vn Wasm_op.F64x2_splat ]

(* The two scalar lanes of half [h]: lane indices [2h] and [2h + 1]. *)
let lane_indices h = (2 * h, (2 * h) + 1)
let arg0 = { Wasm.Mem_arg.align = 0; offset = 0 }

(* The offset of lane [k] of an access, as an index expression. *)
let lane_offset (a : V.Access.t) k =
  if k = 0 then a.V.Access.offset
  else
    Loop_index.Add (a.V.Access.offset, Loop_index.Const (k * a.V.Access.stride))

(* A pair of scalar [f64] values already computed lane by lane into a vector:
   [first] pushes lane 0's value, [second] lane 1's. *)
let pair first second =
  first
  @ [ vn Wasm_op.F64x2_splat ]
  @ second
  @ [ I.Simd_lane (Wasm.Simd_lane.F64x2_replace, 1) ]

let rec half st vc h (e : V.t) : I.t list =
  match e with
  | V.Const x -> [ I.V128_const (f64_pair_bytes x) ]
  | V.Splat s -> splat_instrs st vc s
  | V.Binary (op, a, b) ->
      let a = half st vc h a in
      let b = half st vc h b in
      a @ b
      @ [
          vn
            (match op with
            | Expr.Value.Add -> Wasm_op.F64x2_add
            | Expr.Value.Div -> Wasm_op.F64x2_div
            | Expr.Value.Mul -> Wasm_op.F64x2_mul
            | Expr.Value.Sub -> Wasm_op.F64x2_sub);
        ]
  | V.Float_max (a, b) ->
      let a = half st vc h a in
      let b = half st vc h b in
      a @ b @ [ vn Wasm_op.F64x2_max ]
  | V.Round_f32 a ->
      half st vc h a
      @ [
          vn Wasm_op.F32x4_demote_f64x2_zero; vn Wasm_op.F64x2_promote_low_f32x4;
        ]
  | V.Unary (op, a) -> (
      let a = half st vc h a in
      match op with
      | Expr.Value.Sqrt -> a @ [ vn Wasm_op.F64x2_sqrt ]
      | Expr.Value.Trunc -> a @ [ vn Wasm_op.F64x2_trunc ]
      | Expr.Value.Cos | Expr.Value.Erf | Expr.Value.Exp | Expr.Value.Log
      | Expr.Value.Sin ->
          let callee =
            match op with
            | Expr.Value.Cos -> R.Callee.Cos
            | Expr.Value.Erf -> R.Callee.Erf
            | Expr.Value.Exp -> R.Callee.Exp
            | Expr.Value.Log -> R.Callee.Log
            | _ -> R.Callee.Sin
          in
          let s = fresh st Wasm_type.V128 in
          let lane k =
            [
              get s;
              I.Simd_lane (Wasm.Simd_lane.F64x2_extract, k);
              call st callee;
            ]
          in
          a @ [ set s ] @ pair (lane 0) (lane 1))
  | V.Select (m, a, b) ->
      let a = half st vc h a in
      let b = half st vc h b in
      a @ b @ half_mask st vc h m @ [ vn Wasm_op.V128_bitselect ]
  | V.Temp t -> [ get (temp_local st vc t h) ]
  | V.Index_value { base; step } ->
      let k0, k1 = lane_indices h in
      let lane k =
        index st (Loop_index.Add (base, Loop_index.Const (k * step)))
        @ [ n Wasm_op.F64_convert_i32_s ]
      in
      pair (lane k0) (lane k1)
  | V.Load a -> load st h a

and half_mask st vc h (m : V.mask) : I.t list =
  match m with
  | V.Not m -> half_mask st vc h m @ [ vn Wasm_op.V128_not ]
  | V.Or (a, b) ->
      half_mask st vc h a @ half_mask st vc h b @ [ vn Wasm_op.V128_or ]
  | V.Value_eq (a, b) ->
      half st vc h a @ half st vc h b @ [ vn Wasm_op.F64x2_eq ]
  | V.Value_lt (a, b) ->
      half st vc h a @ half st vc h b @ [ vn Wasm_op.F64x2_lt ]
  | V.Pool_better (best, value) ->
      (* The candidate wins on strict greater-than or on NaN. *)
      let b = fresh st Wasm_type.V128 and v = fresh st Wasm_type.V128 in
      half st vc h best
      @ [ set b ]
      @ half st vc h value
      @ [
          set v;
          get v;
          get b;
          vn Wasm_op.F64x2_gt;
          get v;
          get v;
          vn Wasm_op.F64x2_ne;
          vn Wasm_op.V128_or;
        ]

(* Half [h] of a load: lanes [2h] and [2h + 1] of the access. A contiguous run of
   binary32 or int32 cells is one 64-bit load widened; a contiguous run of
   binary64 is one register load; everything else (a broadcast, a stride, a
   format with a scalar decode) is two scalar loads put into a vector. *)
and load st h (a : V.Access.t) =
  let b = a.V.Access.buffer in
  let k0, k1 = lane_indices h in
  let at k = cell_address st b (Flat (lane_offset a k)) in
  let scalar k = load_cell st b (Flat (lane_offset a k)) in
  if a.V.Access.stride = 0 then scalar 0 @ [ vn Wasm_op.F64x2_splat ]
  else if a.V.Access.stride <> 1 then pair (scalar k0) (scalar k1)
  else
    match fmt_of b with
    | "f32" ->
        at k0
        @ [ I.Simd_load (Wasm.Simd_load.Load64_zero, arg0) ]
        @ [ vn Wasm_op.F64x2_promote_low_f32x4 ]
    | "f64" -> at k0 @ [ I.Simd_load (Wasm.Simd_load.Load, arg0) ]
    | "i32" ->
        at k0
        @ [ I.Simd_load (Wasm.Simd_load.Load64_zero, arg0) ]
        @ [ vn Wasm_op.F64x2_convert_low_i32x4_s ]
    | _ -> pair (scalar k0) (scalar k1)

let store st vc (a : V.Access.t) (value : V.stored) =
  let b = a.V.Access.buffer in
  List.concat
    (List.init vc.halves (fun h ->
         let k0, k1 = lane_indices h in
         let at k = cell_address st b (Flat (lane_offset a k)) in
         match value with
         | V.F32 e when a.V.Access.stride = 1 ->
             at k0 @ half st vc h e
             @ [
                 vn Wasm_op.F32x4_demote_f64x2_zero;
                 I.Simd_store (Wasm.Simd_store.Store64_lane, arg0, 0);
               ]
         | V.F32 e ->
             (* Strided: each lane narrows and stores on its own. *)
             let s = fresh st Wasm_type.V128 in
             half st vc h e
             @ [ set s ]
             @ List.concat_map
                 (fun (k, lane) ->
                   at k
                   @ [
                       get s;
                       I.Simd_lane (Wasm.Simd_lane.F64x2_extract, lane);
                       n Wasm_op.F32_demote_f64;
                       I.Store
                         ( Wasm.Store.F32_store,
                           { Wasm.Mem_arg.align = 2; offset = 0 } );
                     ])
                 [ (k0, 0); (k1, 1) ]
         | V.Bool e ->
             let s = fresh st Wasm_type.V128 in
             half st vc h e
             @ [ set s ]
             @ List.concat_map
                 (fun (k, lane) ->
                   at k
                   @ [
                       get s;
                       I.Simd_lane (Wasm.Simd_lane.F64x2_extract, lane);
                       f64 0.;
                       n Wasm_op.F64_ne;
                       I.Store (Wasm.Store.I32_store8, arg0);
                     ])
                 [ (k0, 0); (k1, 1) ]))

(* The scalar expressions a loop splats and may hoist, each once, in order of
   appearance; [inner] is the variables of the inner loops around the point. *)
let rec collect_splats ~inner acc (e : V.t) =
  match e with
  | V.Splat s ->
      if List.exists (fun x -> x == s) acc || not (hoistable inner s) then acc
      else acc @ [ s ]
  | V.Binary (_, a, b) | V.Float_max (a, b) ->
      collect_splats ~inner (collect_splats ~inner acc a) b
  | V.Round_f32 a | V.Unary (_, a) -> collect_splats ~inner acc a
  | V.Select (m, a, b) ->
      collect_splats ~inner
        (collect_splats ~inner (mask_splats ~inner acc m) a)
        b
  | V.Const _ | V.Index_value _ | V.Load _ | V.Temp _ -> acc

and mask_splats ~inner acc (m : V.mask) =
  match m with
  | V.Not m -> mask_splats ~inner acc m
  | V.Or (a, b) -> mask_splats ~inner (mask_splats ~inner acc a) b
  | V.Pool_better (a, b) | V.Value_eq (a, b) | V.Value_lt (a, b) ->
      collect_splats ~inner (collect_splats ~inner acc a) b

let rec stmt_splats ~inner acc (s : V.stmt) =
  match s with
  | V.Assign (_, e) -> collect_splats ~inner acc e
  | V.Store { value = V.F32 e | V.Bool e; _ } -> collect_splats ~inner acc e
  | V.Index_assign _ | V.Mark _ -> acc
  | V.Inner { var; body; _ } ->
      List.fold_left (stmt_splats ~inner:(var :: inner)) acc body

let rec stmt st vc (s : V.stmt) =
  match s with
  | V.Assign (t, e) ->
      List.concat
        (List.init vc.halves (fun h ->
             half st vc h e @ [ set (temp_local st vc t h) ]))
  | V.Store { access; value } -> store st vc access value
  | V.Index_assign (t, i) -> index st i @ [ set (xtemp st t) ]
  | V.Mark m -> (
      (* A counting build bumps the mark's word once per lane; the default build
         emits nothing. *)
      match st.mark_base with
      | None -> []
      | Some base ->
          let at = base + (4 * Loop_mark.index m) in
          [
            i32 at;
            i32 at;
            I.Load (Wasm.Load.I32_load, { Wasm.Mem_arg.align = 2; offset = 0 });
            i32 (2 * vc.halves);
            n Wasm_op.I32_add;
            I.Store
              (Wasm.Store.I32_store, { Wasm.Mem_arg.align = 2; offset = 0 });
          ])
  | V.Inner { var = v; lo; hi = hi_ix; body } ->
      let name = var st v in
      let lo = index st lo in
      let hi, limit =
        match hi_ix with
        | Loop_index.Const _ | Loop_index.Var _ -> ([], index st hi_ix)
        | _ ->
            let b = bound st v in
            (index st hi_ix @ [ set b ], [ get b ])
      in
      let body = List.concat_map (stmt st vc) body in
      lo
      @ [ set name ]
      @ hi
      @ [
          I.Block
            ( None,
              [
                I.Loop
                  ( None,
                    [ get name ]
                    @ limit
                    @ [ n Wasm_op.I32_ge_s; I.Br_if 1 ]
                    @ body
                    @ [ get name; i32 1; n Wasm_op.I32_add; set name; I.Br 0 ]
                  );
              ] );
        ]

let loop st ~scalar_stmt (l : V.loop) : I.t list =
  let halves = l.V.lanes / 2 in
  let lo, hi =
    match (l.V.lo, l.V.hi) with
    | Loop_index.Const lo, Loop_index.Const hi -> (lo, hi)
    | _ -> invalid_arg "Loop_wasm_vector: a vector loop without constant bounds"
  in
  let vc = { halves; temps = Hashtbl.create 8; splats = [] } in
  let splat_exprs = List.fold_left (stmt_splats ~inner:[]) [] l.V.body in
  (* Each splatted scalar is computed once, before the loop: it depends on
     neither the loop variable nor anything the loop assigns or stores. *)
  let prelude =
    List.concat_map
      (fun e ->
        let local = fresh st Wasm_type.V128 in
        vc.splats <- vc.splats @ [ (e, local) ];
        num st e @ [ vn Wasm_op.F64x2_splat; set local ])
      splat_exprs
  in
  let body = List.concat_map (stmt st vc) l.V.body in
  let var_local = var st l.V.var in
  let trips = max 0 (hi - lo) / l.V.lanes in
  let stop = lo + (trips * l.V.lanes) in
  let remainder =
    match l.V.scalar with
    | Loop_stmt.For f ->
        scalar_stmt (Loop_stmt.For { f with lo = Loop_index.Const stop })
    | s -> scalar_stmt s
  in
  prelude
  @ [ int_const st lo; set var_local ]
  @ [
      I.Block
        ( None,
          [
            I.Loop
              ( None,
                [
                  get var_local;
                  int_const st stop;
                  n Wasm_op.I32_ge_s;
                  I.Br_if 1;
                ]
                @ body
                @ [
                    get var_local;
                    i32 l.V.lanes;
                    n Wasm_op.I32_add;
                    set var_local;
                    I.Br 0;
                  ] );
          ] );
    ]
  @ remainder
