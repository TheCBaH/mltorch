(* Lowering of a vector loop to 128-bit WebAssembly SIMD.

   A logical vector of [lanes] values is [lanes / lpr] registers of [lpr] lanes
   each (the "halves", whatever their number): two [f64] lanes in a binary64
   kernel, four [f32] lanes in a binary32 one. A vector expression is lowered
   one register at a time, each a straight-line sequence leaving one [v128].
   Vector temporaries live in locals, one per register. Nothing here is a
   physical instruction in the vector program: this file is the one place a lane
   count meets the lanes of an [f64x2] or [f32x4].

   Strict means a lane is its scalar iteration: [f64x2] arithmetic is the scalar
   [f64] operation per lane (no fused multiply-add, no reassociation); a
   [Round_f32] is [f32x4.demote_f64x2_zero] then [f64x2.promote_low_f32x4]; the
   maximum is [f64x2.max], which propagates NaN and orders [-0 < +0] like
   [f64.max]; a transcendental, which has no vector instruction, is expanded
   lane by lane through the same scalar callee. In a binary32 kernel the same
   holds with [f32x4]: each lane is the scalar [f32] operation, [Round_f32] is
   the identity, and a transcendental widens one lane, calls the binary64
   import and narrows the result. *)

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
  lpr : int;  (** lanes per register: 2 ([f64x2]) or 4 ([f32x4]) *)
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
  | None ->
      num st e
      @ [ vn (if st.f32 then Wasm_op.F32x4_splat else Wasm_op.F64x2_splat) ]

(* The first lane of register [h]: the lanes of the register are [lane0 .. lane0
   + lpr - 1]. *)
let lane0 vc h = vc.lpr * h
let arg0 = { Wasm.Mem_arg.align = 0; offset = 0 }

(* The offset of lane [k] of an access, as an index expression. *)
let lane_offset (a : V.Access.t) k =
  if k = 0 then a.V.Access.offset
  else
    Loop_index.Add (a.V.Access.offset, Loop_index.Const (k * a.V.Access.stride))

(* The scalar working values already computed lane by lane into a vector:
   element [k] of [lanes] pushes lane [k]'s value. *)
let gather st lanes =
  match lanes with
  | [] -> invalid_arg "Loop_wasm_vector.gather: no lanes"
  | first :: rest ->
      let splat, replace =
        if st.f32 then (Wasm_op.F32x4_splat, Wasm.Simd_lane.F32x4_replace)
        else (Wasm_op.F64x2_splat, Wasm.Simd_lane.F64x2_replace)
      in
      first
      @ [ vn splat ]
      @ List.concat
          (List.mapi (fun k l -> l @ [ I.Simd_lane (replace, k + 1) ]) rest)

let extract st =
  if st.f32 then Wasm.Simd_lane.F32x4_extract else Wasm.Simd_lane.F64x2_extract

let f32_quad_bytes x =
  let bits = Int32.bits_of_float x in
  let one =
    String.init 4 (fun k ->
        Char.chr
          (Int32.to_int (Int32.shift_right_logical bits (8 * k)) land 0xFF))
  in
  one ^ one ^ one ^ one

let rec half st vc h (e : V.t) : I.t list =
  match e with
  | V.Const x ->
      [ I.V128_const (if st.f32 then f32_quad_bytes x else f64_pair_bytes x) ]
  | V.Splat s -> splat_instrs st vc s
  | V.Binary (op, a, b) ->
      let a = half st vc h a in
      let b = half st vc h b in
      a @ b
      @ [
          vn
            (match (st.f32, op) with
            | false, Expr.Value.Add -> Wasm_op.F64x2_add
            | false, Expr.Value.Div -> Wasm_op.F64x2_div
            | false, Expr.Value.Mul -> Wasm_op.F64x2_mul
            | false, Expr.Value.Sub -> Wasm_op.F64x2_sub
            | true, Expr.Value.Add -> Wasm_op.F32x4_add
            | true, Expr.Value.Div -> Wasm_op.F32x4_div
            | true, Expr.Value.Mul -> Wasm_op.F32x4_mul
            | true, Expr.Value.Sub -> Wasm_op.F32x4_sub);
        ]
  | V.Fma (a, b, c) ->
      if st.relaxed_madd && st.f32 then
        let a = half st vc h a in
        let b = half st vc h b in
        let c = half st vc h c in
        a @ b @ c @ [ vn Wasm_op.F32x4_relaxed_madd ]
      else
        invalid_arg
          "Loop_wasm_vector: standard WebAssembly SIMD has no fused \
           multiply-add (plan for relaxed SIMD)"
  | V.Float_max (a, b) ->
      let a = half st vc h a in
      let b = half st vc h b in
      a @ b @ [ vn (if st.f32 then Wasm_op.F32x4_max else Wasm_op.F64x2_max) ]
  | V.Round_f32 a ->
      (* the identity in binary32: the lanes are already single precision *)
      if st.f32 then half st vc h a
      else
        half st vc h a
        @ [
            vn Wasm_op.F32x4_demote_f64x2_zero;
            vn Wasm_op.F64x2_promote_low_f32x4;
          ]
  | V.Unary (op, a) -> (
      let a = half st vc h a in
      match op with
      | Expr.Value.Sqrt ->
          a @ [ vn (if st.f32 then Wasm_op.F32x4_sqrt else Wasm_op.F64x2_sqrt) ]
      | Expr.Value.Trunc ->
          a
          @ [ vn (if st.f32 then Wasm_op.F32x4_trunc else Wasm_op.F64x2_trunc) ]
      | Expr.Value.Cos | Expr.Value.Erf | Expr.Value.Exp | Expr.Value.Log
      | Expr.Value.Sin ->
          let callee =
            match (st.f32, op) with
            | true, Expr.Value.Erf -> R.Callee.Erf_f32
            | _, Expr.Value.Cos -> R.Callee.Cos
            | _, Expr.Value.Erf -> R.Callee.Erf
            | _, Expr.Value.Exp -> R.Callee.Exp
            | _, Expr.Value.Log -> R.Callee.Log
            | _ -> R.Callee.Sin
          in
          let s = fresh st Wasm_type.V128 in
          (* Binary32: the binary64 import on the widened lane, rounded once. *)
          let wide = st.f32 && callee <> R.Callee.Erf_f32 in
          let lane k =
            [ get s; I.Simd_lane (extract st, k) ]
            @ (if wide then [ n Wasm_op.F64_promote_f32 ] else [])
            @ [ call st callee ]
            @ if wide then [ n Wasm_op.F32_demote_f64 ] else []
          in
          a @ [ set s ] @ gather st (List.init vc.lpr lane))
  | V.Select (m, a, b) ->
      let a = half st vc h a in
      let b = half st vc h b in
      a @ b @ half_mask st vc h m @ [ vn Wasm_op.V128_bitselect ]
  | V.Temp t -> [ get (temp_local st vc t h) ]
  | V.Index_value { base; step } ->
      let k0 = lane0 vc h in
      let lane k =
        index st (Loop_index.Add (base, Loop_index.Const (k * step)))
        @ [ n Wasm_op.F64_convert_i32_s ]
        @ if st.f32 then [ n Wasm_op.F32_demote_f64 ] else []
      in
      gather st (List.init vc.lpr (fun k -> lane (k0 + k)))
  | V.Load a -> load st vc h a

and half_mask st vc h (m : V.mask) : I.t list =
  let eq, lt, gt, ne =
    if st.f32 then
      (Wasm_op.F32x4_eq, Wasm_op.F32x4_lt, Wasm_op.F32x4_gt, Wasm_op.F32x4_ne)
    else (Wasm_op.F64x2_eq, Wasm_op.F64x2_lt, Wasm_op.F64x2_gt, Wasm_op.F64x2_ne)
  in
  match m with
  | V.Not m -> half_mask st vc h m @ [ vn Wasm_op.V128_not ]
  | V.Or (a, b) ->
      half_mask st vc h a @ half_mask st vc h b @ [ vn Wasm_op.V128_or ]
  | V.Value_eq (a, b) -> half st vc h a @ half st vc h b @ [ vn eq ]
  | V.Value_lt (a, b) -> half st vc h a @ half st vc h b @ [ vn lt ]
  | V.Pool_better (best, value) ->
      (* The candidate wins on strict greater-than or on NaN. *)
      let b = fresh st Wasm_type.V128 and v = fresh st Wasm_type.V128 in
      half st vc h best
      @ [ set b ]
      @ half st vc h value
      @ [ set v; get v; get b; vn gt; get v; get v; vn ne; vn Wasm_op.V128_or ]

(* Register [h] of a load: its [lpr] lanes of the access. A contiguous run of
   binary32 cells is one 128-bit load in a binary32 kernel, or one 64-bit load
   widened in a binary64 one; a contiguous run of int32 is one 64-bit load
   widened; a contiguous run of binary64 is one register load; everything else (a
   broadcast, a stride, a format with a scalar decode) is scalar loads put into a
   vector. *)
and load st vc h (a : V.Access.t) =
  let b = a.V.Access.buffer in
  let k0 = lane0 vc h in
  let at k = cell_address st b (Flat (lane_offset a k)) in
  let scalar k = load_cell st b (Flat (lane_offset a k)) in
  let lanes = List.init vc.lpr (fun i -> scalar (k0 + i)) in
  if a.V.Access.stride = 0 then
    scalar 0
    @ [ vn (if st.f32 then Wasm_op.F32x4_splat else Wasm_op.F64x2_splat) ]
  else if a.V.Access.stride <> 1 then gather st lanes
  else
    match (fmt_of b, st.f32) with
    | "f32", true -> at k0 @ [ I.Simd_load (Wasm.Simd_load.Load, arg0) ]
    | "f32", false ->
        at k0
        @ [ I.Simd_load (Wasm.Simd_load.Load64_zero, arg0) ]
        @ [ vn Wasm_op.F64x2_promote_low_f32x4 ]
    | "f64", false -> at k0 @ [ I.Simd_load (Wasm.Simd_load.Load, arg0) ]
    | "i32", false ->
        at k0
        @ [ I.Simd_load (Wasm.Simd_load.Load64_zero, arg0) ]
        @ [ vn Wasm_op.F64x2_convert_low_i32x4_s ]
    | _ -> gather st lanes

let store st vc (a : V.Access.t) (value : V.stored) =
  let b = a.V.Access.buffer in
  List.concat
    (List.init vc.halves (fun h ->
         let k0 = lane0 vc h in
         let at k = cell_address st b (Flat (lane_offset a k)) in
         let lanes = List.init vc.lpr (fun i -> k0 + i) in
         match value with
         | V.F32 e when a.V.Access.stride = 1 && st.f32 ->
             at k0 @ half st vc h e
             @ [ I.Simd_store (Wasm.Simd_store.Store, arg0, 0) ]
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
             @ List.concat
                 (List.mapi
                    (fun lane k ->
                      at k
                      @ [ get s; I.Simd_lane (extract st, lane) ]
                      @ (if st.f32 then [] else [ n Wasm_op.F32_demote_f64 ])
                      @ [
                          I.Store
                            ( Wasm.Store.F32_store,
                              { Wasm.Mem_arg.align = 2; offset = 0 } );
                        ])
                    lanes)
         | V.Bool e ->
             let s = fresh st Wasm_type.V128 in
             half st vc h e
             @ [ set s ]
             @ List.concat
                 (List.mapi
                    (fun lane k ->
                      at k
                      @ [
                          get s;
                          I.Simd_lane (extract st, lane);
                          fconst st 0.;
                          n (if st.f32 then Wasm_op.F32_ne else Wasm_op.F64_ne);
                          I.Store (Wasm.Store.I32_store8, arg0);
                        ])
                    lanes)))

(* The scalar expressions a loop splats and may hoist, each once, in order of
   appearance; [inner] is the variables of the inner loops around the point. *)
let rec collect_splats ~inner acc (e : V.t) =
  match e with
  | V.Splat s ->
      if List.exists (fun x -> x == s) acc || not (hoistable inner s) then acc
      else acc @ [ s ]
  | V.Binary (_, a, b) | V.Float_max (a, b) ->
      collect_splats ~inner (collect_splats ~inner acc a) b
  | V.Fma (a, b, c) ->
      collect_splats ~inner
        (collect_splats ~inner (collect_splats ~inner acc a) b)
        c
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
            i32 (vc.lpr * vc.halves);
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
  let lpr = if st.f32 then 4 else 2 in
  let halves = l.V.lanes / lpr in
  let lo, hi =
    match (l.V.lo, l.V.hi) with
    | Loop_index.Const lo, Loop_index.Const hi -> (lo, hi)
    | _ -> invalid_arg "Loop_wasm_vector: a vector loop without constant bounds"
  in
  let vc = { halves; lpr; temps = Hashtbl.create 8; splats = [] } in
  let splat_exprs = List.fold_left (stmt_splats ~inner:[]) [] l.V.body in
  (* Each splatted scalar is computed once, before the loop: it depends on
     neither the loop variable nor anything the loop assigns or stores. *)
  let prelude =
    List.concat_map
      (fun e ->
        let local = fresh st Wasm_type.V128 in
        vc.splats <- vc.splats @ [ (e, local) ];
        num st e
        @ [
            vn (if st.f32 then Wasm_op.F32x4_splat else Wasm_op.F64x2_splat);
            set local;
          ])
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

(* A scheduled sum ({!Loop_vector.Reduction}): [parts] accumulators of [halves]
   registers each over the main rounds, the leftover vectors, the adjacent-pair
   trees (accumulators first, as register adds, then lanes, as scalar adds), the
   sequential tail and the seed, in exactly the order the definition gives. *)
let reduction st (r : V.Reduction.t) : I.t list =
  let lanes = r.V.Reduction.lanes and parts = r.V.Reduction.parts in
  let lpr = if st.f32 then 4 else 2 in
  if lanes mod lpr <> 0 then
    invalid_arg
      "Loop_wasm_vector: a reduction whose lanes the registers cannot hold";
  let halves = lanes / lpr in
  let lo = r.V.Reduction.lo in
  let terms = r.V.Reduction.hi - lo in
  let full = terms / lanes in
  let main = full / parts and extra = full mod parts in
  let tail = terms - (full * lanes) in
  let vc = { halves; lpr; temps = Hashtbl.create 8; splats = [] } in
  let splat_exprs = collect_splats ~inner:[] [] r.V.Reduction.term in
  let prelude =
    List.concat_map
      (fun e ->
        let local = fresh st Wasm_type.V128 in
        vc.splats <- vc.splats @ [ (e, local) ];
        num st e
        @ [
            vn (if st.f32 then Wasm_op.F32x4_splat else Wasm_op.F64x2_splat);
            set local;
          ])
      splat_exprs
  in
  let accs =
    Array.init parts (fun _ ->
        Array.init halves (fun _ -> fresh st Wasm_type.V128))
  in
  let var_local = var st r.V.Reduction.var in
  let add_v = vn (if st.f32 then Wasm_op.F32x4_add else Wasm_op.F64x2_add) in
  let zero =
    I.V128_const (if st.f32 then f32_quad_bytes 0. else f64_pair_bytes 0.)
  in
  let init =
    List.concat
      (List.init parts (fun j ->
           List.concat (List.init halves (fun h -> [ zero; set accs.(j).(h) ]))))
  in
  let accumulate j delta =
    let term = V.shift r.V.Reduction.var delta r.V.Reduction.term in
    List.concat
      (List.init halves (fun h ->
           [ get accs.(j).(h) ]
           @ half st vc h term
           @ [ add_v; set accs.(j).(h) ]))
  in
  let rounds =
    if main = 0 then []
    else
      [ int_const st lo; set var_local ]
      @ [
          I.Block
            ( None,
              [
                I.Loop
                  ( None,
                    [
                      get var_local;
                      int_const st (lo + (main * parts * lanes));
                      n Wasm_op.I32_ge_s;
                      I.Br_if 1;
                    ]
                    @ List.concat
                        (List.init parts (fun j -> accumulate j (j * lanes)))
                    @ [
                        get var_local;
                        i32 (parts * lanes);
                        n Wasm_op.I32_add;
                        set var_local;
                        I.Br 0;
                      ] );
              ] );
        ]
  in
  let leftover =
    List.concat
      (List.init extra (fun e ->
           [
             int_const st (lo + (main * parts * lanes) + (e * lanes));
             set var_local;
           ]
           @ accumulate e 0))
  in
  (* tree over a list of instruction sequences, each leaving one value *)
  let rec tree op = function
    | [] -> invalid_arg "Loop_wasm_vector.tree"
    | [ x ] -> x
    | xs ->
        let rec pairs = function
          | a :: b :: rest -> (a @ b @ [ op ]) :: pairs rest
          | rest -> rest
        in
        tree op (pairs xs)
  in
  let combined = Array.init halves (fun _ -> fresh st Wasm_type.V128) in
  let combine =
    List.concat
      (List.init halves (fun h ->
           tree add_v (List.init parts (fun j -> [ get accs.(j).(h) ]))
           @ [ set combined.(h) ]))
  in
  let scalar_add = n (if st.f32 then Wasm_op.F32_add else Wasm_op.F64_add) in
  let horizontal_local = fresh st (ftype st) in
  let horizontal =
    tree scalar_add
      (List.init lanes (fun l ->
           [ get combined.(l / lpr); I.Simd_lane (extract st, l mod lpr) ]))
    @ [ set horizontal_local ]
  in
  let tail_local = fresh st (ftype st) in
  let tail_stmts =
    [ fconst st 0.; set tail_local ]
    @ List.concat
        (List.init tail (fun u ->
             [
               int_const st (lo + (full * lanes) + u);
               set var_local;
               get tail_local;
             ]
             @ num st
                 (Loop_vector_expand.lane_expr ~var:r.V.Reduction.var
                    ~base:(Loop_index.Var r.V.Reduction.var)
                    ~temp:(fun _ _ ->
                      invalid_arg
                        "Loop_wasm_vector: a reduction term has no temporaries")
                    r.V.Reduction.term 0)
             @ [ scalar_add; set tail_local ]))
  in
  prelude @ init @ rounds @ leftover @ combine @ horizontal @ tail_stmts
  @ [
      fconst st r.V.Reduction.seed;
      get horizontal_local;
      get tail_local;
      scalar_add;
      scalar_add;
      set (ftemp st r.V.Reduction.acc);
    ]
