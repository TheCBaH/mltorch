open Ssa_ir
open Ssa_wasm_ctx
open Ssa_wasm_op

(* Writes a single-register result and returns the instructions that do. *)
let set_one st (res : Ssa_value.t) instrs =
  let r = (define st res).(0) in
  instrs @ [ set r ]

(* Writes a result held in several registers: one instruction list each. *)
let set_chunks ?(shape = Wide) st (res : Ssa_value.t) chunks =
  let regs = define ~shape st res in
  List.concat (List.mapi (fun i c -> c @ [ set regs.(i) ]) chunks)

let per_register (v : Ssa_value.t) = lanes_per_register (elem_of v)
let chunk_count st (v : Ssa_value.t) = Array.length (regs st v)

(* ---- lane-wise operations ------------------------------------------------------- *)

let lanewise st (inner : Ssa_op.t) (res : Ssa_value.t) =
  let src v i = get (regs st v).(i) in
  let chunks_of v f = List.init (chunk_count st v) f in
  match inner with
  | Ssa_op.Convert (Ssa_op.Convert.F32_to_f64, a) ->
      let count = lanes_of a / 2 in
      let lane k =
        [ src a (k / 4); I.Simd_lane (Wasm.Simd_lane.F32x4_extract, k mod 4) ]
      in
      set_chunks st res
        (List.init count (fun i ->
             if 2 * i mod 4 = 0 then
               [ src a (2 * i / 4); vn Wasm_op.F64x2_promote_low_f32x4 ]
             else
               lane (2 * i)
               @ [ n Wasm_op.F64_promote_f32; vn Wasm_op.F64x2_splat ]
               @ lane ((2 * i) + 1)
               @ [
                   n Wasm_op.F64_promote_f32;
                   I.Simd_lane (Wasm.Simd_lane.F64x2_replace, 1);
                 ]))
  | Ssa_op.Convert (Ssa_op.Convert.F64_to_f32, a) ->
      let count = lanes_of a / 4 in
      let lane k =
        [ src a (k / 2); I.Simd_lane (Wasm.Simd_lane.F64x2_extract, k mod 2) ]
      in
      set_chunks st res
        (List.init count (fun j ->
             [ src a (2 * j); vn Wasm_op.F32x4_demote_f64x2_zero ]
             @ lane ((4 * j) + 2)
             @ [
                 n Wasm_op.F32_demote_f64;
                 I.Simd_lane (Wasm.Simd_lane.F32x4_replace, 2);
               ]
             @ lane ((4 * j) + 3)
             @ [
                 n Wasm_op.F32_demote_f64;
                 I.Simd_lane (Wasm.Simd_lane.F32x4_replace, 3);
               ]))
  | Ssa_op.Float_binary (o, a, b) ->
      let e = elem_of a in
      set_chunks st res
        (chunks_of a (fun i -> [ src a i; src b i; vn (vbin e o) ]))
  | Ssa_op.Float_compare (c, a, b) ->
      let e = elem_of a in
      let op =
        match (e, c) with
        | Ssa_type.F32, Ssa_op.Compare.Eq -> Wasm_op.F32x4_eq
        | Ssa_type.F32, Ssa_op.Compare.Lt -> Wasm_op.F32x4_lt
        | _, Ssa_op.Compare.Eq -> Wasm_op.F64x2_eq
        | _, Ssa_op.Compare.Lt -> Wasm_op.F64x2_lt
      in
      set_chunks ~shape:(shape_of_elem e) st res
        (chunks_of a (fun i -> [ src a i; src b i; vn op ]))
  | Ssa_op.Float_max (a, b) ->
      let op =
        if elem_of a = Ssa_type.F32 then Wasm_op.F32x4_max
        else Wasm_op.F64x2_max
      in
      set_chunks st res (chunks_of a (fun i -> [ src a i; src b i; vn op ]))
  | Ssa_op.Float_fma (a, b, c) ->
      if elem_of a = Ssa_type.F32 && st.relaxed_madd then
        set_chunks st res
          (chunks_of a (fun i ->
               [ src a i; src b i; src c i; vn Wasm_op.F32x4_relaxed_madd ]))
      else
        refuse
          (`Unsupported_operation "a fused multiply-add outside relaxed SIMD")
  | Ssa_op.Float_unary (op, a) -> (
      let f32 = elem_of a = Ssa_type.F32 in
      match op with
      | Expr.Value.Sqrt ->
          set_chunks st res
            (chunks_of a (fun i ->
                 [
                   src a i;
                   vn (if f32 then Wasm_op.F32x4_sqrt else Wasm_op.F64x2_sqrt);
                 ]))
      | Expr.Value.Trunc ->
          set_chunks st res
            (chunks_of a (fun i ->
                 [
                   src a i;
                   vn (if f32 then Wasm_op.F32x4_trunc else Wasm_op.F64x2_trunc);
                 ]))
      | ( Expr.Value.Cos | Expr.Value.Erf | Expr.Value.Exp | Expr.Value.Log
        | Expr.Value.Sin ) as op
        when f32 ->
          (* the module's own four-lane sequence, once per register *)
          let callee =
            match op with
            | Expr.Value.Cos -> R.Callee.Cos_f32x4
            | Expr.Value.Erf -> R.Callee.Erf_f32x4
            | Expr.Value.Exp -> R.Callee.Exp_f32x4
            | Expr.Value.Log -> R.Callee.Log_f32x4
            | _ -> R.Callee.Sin_f32x4
          in
          set_chunks st res (chunks_of a (fun i -> [ src a i; call st callee ]))
      | Expr.Value.Cos | Expr.Value.Erf | Expr.Value.Exp | Expr.Value.Log
      | Expr.Value.Sin ->
          (* no vector instruction: each lane runs the scalar callee *)
          let per = per_register a in
          let lane k =
            [
              src a (k / per);
              I.Simd_lane
                ( (if f32 then Wasm.Simd_lane.F32x4_extract
                   else Wasm.Simd_lane.F64x2_extract),
                  k mod per );
            ]
            @ unary st ~f32 op
          in
          let lanes = List.init (lanes_of a) lane in
          set_chunks st res (gather (elem_of a) lanes))
  | Ssa_op.Pool_better (best, value) ->
      let e = elem_of value in
      let gt, ne =
        if e = Ssa_type.F32 then (Wasm_op.F32x4_gt, Wasm_op.F32x4_ne)
        else (Wasm_op.F64x2_gt, Wasm_op.F64x2_ne)
      in
      set_chunks ~shape:(shape_of_elem e) st res
        (chunks_of value (fun i ->
             [
               src value i;
               src best i;
               vn gt;
               src value i;
               src value i;
               vn ne;
               vn Wasm_op.V128_or;
             ]))
  | Ssa_op.Pred_not a ->
      set_chunks ~shape:(mask_shape st a) st res
        (chunks_of a (fun i -> [ src a i; vn Wasm_op.V128_not ]))
  | Ssa_op.Pred_or (a, b) ->
      let shape = mask_shape st a in
      let b_chunks = reshape_mask st b ~target:shape in
      set_chunks ~shape st res
        (List.mapi
           (fun i bc -> [ src a i ] @ bc @ [ vn Wasm_op.V128_or ])
           b_chunks)
  | Ssa_op.Select (p, a, b) -> (
      match res.Ssa_value.ty with
      | Ssa_type.Mask _ ->
          let shape = mask_shape st a in
          let p_chunks = reshape_mask st p ~target:shape in
          set_chunks ~shape st res
            (List.mapi
               (fun i pc ->
                 [ src a i; src b i ] @ pc @ [ vn Wasm_op.V128_bitselect ])
               p_chunks)
      | _ ->
          let p_chunks =
            reshape_mask st p ~target:(shape_of_elem (elem_of a))
          in
          set_chunks st res
            (List.mapi
               (fun i pc ->
                 [ src a i; src b i ] @ pc @ [ vn Wasm_op.V128_bitselect ])
               p_chunks))
  | _ -> refuse (`Unsupported_operation "an operation with no lane-wise form")

(* ---- one instruction ------------------------------------------------------------ *)

let first_result (i : Ssa_instr.t) =
  match i.Ssa_instr.results with
  | r :: _ -> r
  | [] -> invalid_arg "Ssa_wasm: an instruction without a result"

let meter_check st ~limits (op : Ssa_op.t) =
  let remaining, live = meter st in
  match op with
  | Ssa_op.Meter_charge ->
      [
        get remaining;
        I.I64_const 0L;
        n Wasm_op.I64_le_s;
        I.If
          ( None,
            meter_failure st F.Meter.Updates_exhausted
              (Expr.Scan_limits.max_updates limits),
            [] );
        get remaining;
        I.I64_const 1L;
        n Wasm_op.I64_sub;
        set remaining;
      ]
  | Ssa_op.Meter_release width ->
      [
        get live; I.I64_const (Int64.mul 2L width); n Wasm_op.I64_sub; set live;
      ]
  | Ssa_op.Meter_reserve width ->
      let amount = Int64.mul 2L width in
      let max_state = Expr.Scan_limits.max_state limits in
      [
        get live;
        I.I64_const amount;
        n Wasm_op.I64_add;
        I.I64_const (Int64.of_int max_state);
        n Wasm_op.I64_gt_s;
        I.If
          ( None,
            meter_failure st F.Meter.State_over_limit (Int64.of_int max_state),
            [] );
        get live;
        I.I64_const amount;
        n Wasm_op.I64_add;
        set live;
      ]
  | Ssa_op.Meter_reset ->
      [
        I.I64_const (Expr.Scan_limits.max_updates limits);
        set remaining;
        I.I64_const 0L;
        set live;
      ]
  | _ -> invalid_arg "Ssa_wasm.meter_check"

let instr st ~limits (i : Ssa_instr.t) : I.t list =
  let op = i.Ssa_instr.op in
  let r = read st in
  let one instrs = set_one st (first_result i) instrs in
  match op with
  | Ssa_op.Mark _ | Ssa_op.Mark_lanes _ -> []
  | Ssa_op.Contract _ -> refuse (`Unsupported_operation "a contraction")
  | Ssa_op.Check_access { buffer; at } -> (
      let b = find_buffer st buffer in
      match at with
      | Ssa_access.Coord c -> check_coord st b c
      | Ssa_access.Flat _ -> [])
  | Ssa_op.Check_gather { raw; extent } ->
      (* the raw index, in [-extent, extent) *)
      r raw
      @ [ I.I64_const (Int64.neg extent); n Wasm_op.I64_lt_s ]
      @ r raw
      @ [
          I.I64_const extent;
          n Wasm_op.I64_ge_s;
          n Wasm_op.I32_or;
          I.If
            ( None,
              fail st F.Kind.Gather_index_out_of_range
                [ (0, r raw); (1, [ I.I64_const extent ]) ],
              [] );
        ]
  | Ssa_op.Check_local { var = lv; at; extent } ->
      let s =
        site st
          (LF.Local_out_of_range
             {
               local = lv;
               index = Loop_ir.Loop_index.Const 0;
               extent = Int64.to_int extent;
             })
      in
      r at
      @ [
          index_const extent;
          n Wasm_op.I32_ge_u;
          I.If (None, fail st F.Kind.Unbound_local [ (0, i64_of_int s) ], []);
        ]
  | Ssa_op.Check_scan { var = lv; row; lane; row_extent; lane_extent } ->
      let record which s extent =
        fail st F.Kind.Scan_projection
          [
            (0, i64_of_int (match which with `Lane -> 0 | `Row -> 1));
            (1, i64_of_int (if Option.is_some lv then 1 else 0));
            (2, wide (r row));
            (3, wide (r lane));
            (4, [ I.I64_const extent ]);
            (5, i64_of_int s);
          ]
      in
      let row_site =
        site st
          (LF.Scan_row_out_of_range
             {
               local = lv;
               row = Loop_ir.Loop_index.Const 0;
               lane = Loop_ir.Loop_index.Const 0;
               extent = Int64.to_int row_extent;
             })
      in
      let lane_site =
        site st
          (LF.Scan_lane_out_of_range
             {
               local = lv;
               row = Loop_ir.Loop_index.Const 0;
               lane = Loop_ir.Loop_index.Const 0;
               extent = Int64.to_int lane_extent;
             })
      in
      r row
      @ [
          index_const row_extent;
          n Wasm_op.I32_ge_u;
          I.If (None, record `Row row_site row_extent, []);
        ]
      @ r lane
      @ [
          index_const lane_extent;
          n Wasm_op.I32_ge_u;
          I.If (None, record `Lane lane_site lane_extent, []);
        ]
  | Ssa_op.Const c ->
      one
        [
          (match c with
          | Ssa_const.F32 x -> f32 x
          | Ssa_const.F64 x -> f64 x
          | Ssa_const.I64 k -> I.I64_const k
          | Ssa_const.Index k -> index_const k
          | Ssa_const.Pred b -> i32 (if b then 1 else 0));
        ]
  | Ssa_op.Convert (c, a) ->
      one
        (r a
        @
        match c with
        | Ssa_op.Convert.F32_to_f64 -> [ n Wasm_op.F64_promote_f32 ]
        | Ssa_op.Convert.F64_to_f32 -> [ n Wasm_op.F32_demote_f64 ]
        | Ssa_op.Convert.I64_to_f32 -> [ n Wasm_op.F32_convert_i64_s ]
        | Ssa_op.Convert.I64_to_f64 -> [ n Wasm_op.F64_convert_i64_s ]
        | Ssa_op.Convert.Index_to_f64 -> [ n Wasm_op.F64_convert_i32_s ]
        | Ssa_op.Convert.Index_to_i64 -> [ n Wasm_op.I64_extend_i32_s ])
  | Ssa_op.Float_binary (o, a, b) ->
      one (r a @ r b @ [ n (float_op ~f32:(is_f32 (first_result i)) o) ])
  | Ssa_op.Float_compare (c, a, b) ->
      let ty = if is_f32 a then `F32 else `F64 in
      one (r a @ r b @ [ n (compare_op ~ty c) ])
  | Ssa_op.Float_fma _ ->
      refuse (`Unsupported_operation "a scalar fused multiply-add")
  | Ssa_op.Float_max (a, b) ->
      one
        (r a @ r b
        @ [
            n
              (if is_f32 (first_result i) then Wasm_op.F32_max
               else Wasm_op.F64_max);
          ])
  | Ssa_op.Float_to_i64 a ->
      let x = fresh st Wasm_type.F64 in
      let checked =
        r a
        @ [
            set x;
            get x;
            f64 (-9223372036854775808.);
            n Wasm_op.F64_ge;
            get x;
            f64 9223372036854775808.;
            n Wasm_op.F64_lt;
            n Wasm_op.I32_and;
            n Wasm_op.I32_eqz;
            I.If (None, from_float_failure st x, []);
          ]
      in
      checked @ one [ get x; n Wasm_op.I64_trunc_sat_f64_s ]
  | Ssa_op.Float_unary (o, a) ->
      one (r a @ unary st ~f32:(is_f32 (first_result i)) o)
  | Ssa_op.I64_arith (o, a, b) ->
      one
        (r a @ r b
        @ [
            n
              (match o with
              | Ssa_op.I64_op.Add -> Wasm_op.I64_add
              | Ssa_op.I64_op.Mul -> Wasm_op.I64_mul
              | Ssa_op.I64_op.Sub -> Wasm_op.I64_sub);
          ])
  | Ssa_op.I64_compare (c, a, b) ->
      one (r a @ r b @ [ n (compare_op ~ty:`I64 c) ])
  | Ssa_op.Index_compare (c, a, b) ->
      one (r a @ r b @ [ n (compare_op ~ty:`I32 c) ])
  | Ssa_op.I64_div (a, b) ->
      r b
      @ [
          n Wasm_op.I64_eqz;
          I.If (None, fail st F.Kind.I64_division_by_zero [], []);
        ]
      @ r a
      @ [ I.I64_const Int64.min_int; n Wasm_op.I64_eq ]
      @ r b
      @ [
          I.I64_const (-1L);
          n Wasm_op.I64_eq;
          n Wasm_op.I32_and;
          I.If (None, fail st F.Kind.I64_division_overflow [], []);
        ]
      @ one (r a @ r b @ [ call st R.Callee.I64_div ])
  | Ssa_op.Index_add (a, b) ->
      let lhs = wide (r a) and rhs = wide (r b) in
      outside_int32 (lhs @ rhs @ [ n Wasm_op.I64_add ])
      @ [
          I.If
            ( None,
              fail st F.Kind.Index_overflow
                [ (0, i64_of_int 0); (1, lhs); (2, rhs) ],
              [] );
        ]
      @ one (r a @ r b @ [ n Wasm_op.I32_add ])
  | Ssa_op.Index_add_in_domain (a, b) -> one (r a @ r b @ [ n Wasm_op.I32_add ])
  | Ssa_op.Index_ceil_div (k, a) ->
      one (r a @ [ index_const k; call st R.Callee.Ceil_div ])
  | Ssa_op.Index_clamp_low a ->
      one (r a @ [ i32 0 ] @ r a @ [ i32 0; n Wasm_op.I32_gt_s; I.Select ])
  | Ssa_op.Index_floor_div (k, a) ->
      one (r a @ [ index_const k; call st R.Callee.Floor_div ])
  | Ssa_op.Index_max (a, b) ->
      one (r a @ r b @ r a @ r b @ [ n Wasm_op.I32_gt_s; I.Select ])
  | Ssa_op.Index_min (a, b) ->
      one (r a @ r b @ r a @ r b @ [ n Wasm_op.I32_lt_s; I.Select ])
  | Ssa_op.Index_of_i64 a -> one (r a @ [ n Wasm_op.I32_wrap_i64 ])
  | Ssa_op.Index_scale (k, a) ->
      let lhs = [ I.I64_const k ] and rhs = wide (r a) in
      outside_int32 (lhs @ rhs @ [ n Wasm_op.I64_mul ])
      @ [
          I.If
            ( None,
              fail st F.Kind.Index_overflow
                [ (0, i64_of_int 1); (1, lhs); (2, rhs) ],
              [] );
        ]
      @ one ([ index_const k ] @ r a @ [ n Wasm_op.I32_mul ])
  | Ssa_op.Index_scale_in_domain (k, a) ->
      one ([ index_const k ] @ r a @ [ n Wasm_op.I32_mul ])
  | Ssa_op.Lanewise inner -> lanewise st inner (first_result i)
  | Ssa_op.Load { buffer; at; decode = d }
  | Ssa_op.Load_in_bounds { buffer; at; decode = d } ->
      let b = find_buffer st buffer in
      let check =
        match (op, at) with
        | Ssa_op.Load _, Ssa_access.Coord c -> check_coord st b c
        | _ -> []
      in
      let address = cell_address st b (access_index st b at) in
      check @ one (decode st b d ~address ~access:at)
  | Ssa_op.Local_alloc { slots; var = lv } ->
      let res = first_result i in
      let off = reserve_local st (Int64.mul 8L slots) in
      let a = (define st res).(0) in
      Hashtbl.replace st.locals (res.Ssa_value.id :> int) (slots, lv);
      let count = Int64.to_int slots in
      let init = [ get 0; i32 off; n Wasm_op.I32_add; set a ] in
      if count = 0 then init
      else
        let c = fresh st Wasm_type.I32 in
        init
        @ [
            i32 0;
            set c;
            I.Block
              ( None,
                [
                  I.Loop
                    ( None,
                      [
                        get c;
                        i32 count;
                        n Wasm_op.I32_ge_s;
                        I.Br_if 1;
                        get a;
                        get c;
                        i32 3;
                        n Wasm_op.I32_shl;
                        n Wasm_op.I32_add;
                        (* zero bytes, written as an integer: a cell is f32 or
                           f64 only once something writes it *)
                        I.I64_const 0L;
                        I.Store (Wasm.Store.I64_store, arg 3 0);
                        get c;
                        i32 1;
                        n Wasm_op.I32_add;
                        set c;
                        I.Br 0;
                      ] );
                ] );
          ]
  | Ssa_op.Local_read { local; at; cell } ->
      let check =
        match Hashtbl.find_opt st.locals (local.Ssa_value.id :> int) with
        | Some (slots, Some lv) ->
            let s =
              site st
                (LF.Local_out_of_range
                   {
                     local = lv;
                     index = Loop_ir.Loop_index.Const 0;
                     extent = Int64.to_int slots;
                   })
            in
            r at
            @ [
                index_const slots;
                n Wasm_op.I32_ge_u;
                I.If
                  (None, fail st F.Kind.Unbound_local [ (0, i64_of_int s) ], []);
              ]
        | Some (_, None) | None -> []
      in
      check
      @ one
          (local_address st local at
          @
          match cell with
          | Ssa_op.Cell.F64 -> [ I.Load (Wasm.Load.F64_load, arg 3 0) ]
          | Ssa_op.Cell.F32 -> [ I.Load (Wasm.Load.F32_load, arg 2 0) ])
  | Ssa_op.Local_write { local; at; value } ->
      local_address st local at @ r value
      @ [
          (if is_f32 value then I.Store (Wasm.Store.F32_store, arg 2 0)
           else I.Store (Wasm.Store.F64_store, arg 3 0));
        ]
  | Ssa_op.Meter_charge | Ssa_op.Meter_release _ | Ssa_op.Meter_reserve _
  | Ssa_op.Meter_reset ->
      meter_check st ~limits op
  | Ssa_op.Pool_better (best, value) ->
      let e = if is_f32 value then `F32 else `F64 in
      let gt, ne =
        match e with
        | `F32 -> (Wasm_op.F32_gt, Wasm_op.F32_ne)
        | `F64 -> (Wasm_op.F64_gt, Wasm_op.F64_ne)
      in
      one
        (r value @ r best
        @ [ n gt ]
        @ r value @ r value
        @ [ n ne; n Wasm_op.I32_or ])
  | Ssa_op.Pred_not a -> one (r a @ [ n Wasm_op.I32_eqz ])
  | Ssa_op.Pred_or (a, b) -> one (r a @ r b @ [ n Wasm_op.I32_or ])
  | Ssa_op.Select (p, a, b) -> one (r a @ r b @ r p @ [ I.Select ])
  | Ssa_op.Store { buffer; at; encode; value } ->
      let b = find_buffer st buffer in
      cell_address st b (access_index st b at) @ r value @ encode_store encode
  | Ssa_op.Vec_extract { lane; vector } -> (
      let k = Ssa_type.Lane.to_int lane in
      let res = first_result i in
      match vector.Ssa_value.ty with
      | Ssa_type.Mask _ -> (
          let regs = regs st vector in
          match mask_shape st vector with
          | Wide ->
              one
                [
                  get regs.(k / 2);
                  I.Simd_lane (Wasm.Simd_lane.I64x2_extract, k mod 2);
                  I.I64_const 0L;
                  n Wasm_op.I64_ne;
                ]
          | Narrow ->
              one
                [
                  get regs.(k / 4);
                  I.Simd_lane (Wasm.Simd_lane.I32x4_extract, k mod 4);
                  i32 0;
                  n Wasm_op.I32_ne;
                ])
      | _ ->
          ignore res;
          let e = elem_of vector in
          let per = lanes_per_register e in
          one
            [
              get (regs st vector).(k / per);
              I.Simd_lane (extract_lane e, k mod per);
            ])
  | Ssa_op.Vec_insert { lane; vector; element } ->
      let k = Ssa_type.Lane.to_int lane in
      let res = first_result i in
      let shape =
        match res.Ssa_value.ty with
        | Ssa_type.Mask _ -> mask_shape st vector
        | _ -> Wide
      in
      let dst = define ~shape st res in
      let src = regs st vector in
      let copy =
        List.concat
          (List.mapi (fun c d -> [ get src.(c); set d ]) (Array.to_list dst))
      in
      let patch =
        match res.Ssa_value.ty with
        | Ssa_type.Mask _ -> (
            match shape with
            | Wide ->
                [ get dst.(k / 2); I.I64_const 0L ]
                @ r element
                @ [
                    n Wasm_op.I64_extend_i32_u;
                    n Wasm_op.I64_sub;
                    I.Simd_lane (Wasm.Simd_lane.I64x2_replace, k mod 2);
                    set dst.(k / 2);
                  ]
            | Narrow ->
                [ get dst.(k / 4); i32 0 ]
                @ r element
                @ [
                    n Wasm_op.I32_sub;
                    I.Simd_lane (Wasm.Simd_lane.I32x4_replace, k mod 4);
                    set dst.(k / 4);
                  ])
        | _ ->
            let e = elem_of res in
            let per = lanes_per_register e in
            [ get dst.(k / per) ]
            @ r element
            @ [ I.Simd_lane (replace_lane e, k mod per); set dst.(k / per) ]
      in
      copy @ patch
  | Ssa_op.Vec_iota { base; step; lanes } ->
      let res = first_result i in
      if elem_of res <> Ssa_type.F64 then
        refuse (`Unsupported_operation "a binary32 iota");
      let lane k =
        wide (r base)
        @ [
            I.I64_const (Int64.mul (Int64.of_int k) step);
            n Wasm_op.I64_add;
            n Wasm_op.F64_convert_i64_s;
          ]
      in
      set_chunks st res
        (gather Ssa_type.F64 (List.init (Ssa_type.Lanes.to_int lanes) lane))
  | Ssa_op.Vec_splat { element; lanes = _ } -> (
      let res = first_result i in
      match res.Ssa_value.ty with
      | Ssa_type.Mask _ ->
          let count = Array.length (define ~shape:Wide st res) in
          let regs = regs st res in
          List.concat
            (List.init count (fun c ->
                 [ I.I64_const 0L ] @ r element
                 @ [
                     n Wasm_op.I64_extend_i32_u;
                     n Wasm_op.I64_sub;
                     vn Wasm_op.I64x2_splat;
                     set regs.(c);
                   ]))
      | _ ->
          let e = elem_of res in
          let count = List.length (local_types st res ~shape:Wide) in
          set_chunks st res
            (List.init count (fun _ -> r element @ [ vn (splat_op e) ])))
  | Ssa_op.Vec_load { buffer; at; steps; decode = d; lanes } ->
      let b = find_buffer st buffer in
      (* a per-channel dequantization reads lane 0's channel for every lane *)
      if
        Ssa_format.per_channel b.Ssa_buffer.format
        && not (Int64.equal steps.Expr.Coord.c 0L)
      then
        refuse
          (`Unsupported_operation
             "a vector load across the channels of a per-channel quantized \
              buffer");
      let res = first_result i in
      let delta = lane_delta b steps in
      let o = fresh st Wasm_type.I32 in
      let count = Ssa_type.Lanes.to_int lanes in
      let start = coord_offset st b at @ [ set o ] in
      let element k =
        [
          get o;
          index_const (Int64.mul (Int64.of_int k) delta);
          n Wasm_op.I32_add;
        ]
      in
      let lane k =
        decode st b d
          ~address:(cell_address st b (element k))
          ~access:(Ssa_access.Coord at)
      in
      (* registers of the result's element: a fast form for the access where
         one exists, else lane by lane *)
      let chunks =
        match elem_of res with
        | Ssa_type.F64 ->
            List.init (count / 2) (fun i ->
                match (d, Int64.equal delta 1L) with
                | Ssa_op.Decode.F32_to_f64, true ->
                    cell_address st b (element (2 * i))
                    @ [
                        I.Simd_load (Wasm.Simd_load.Load64_zero, arg 2 0);
                        vn Wasm_op.F64x2_promote_low_f32x4;
                      ]
                | Ssa_op.Decode.F64_to_f64, true ->
                    cell_address st b (element (2 * i))
                    @ [ I.Simd_load (Wasm.Simd_load.Load, arg 3 0) ]
                | _ ->
                    List.hd
                      (gather Ssa_type.F64 [ lane (2 * i); lane ((2 * i) + 1) ]))
        | Ssa_type.F32 ->
            let per = lanes_per_register Ssa_type.F32 in
            let splat_i32 k = [ i32 k; vn Wasm_op.I32x4_splat ] in
            let splat_f32 x = [ f32 x; vn Wasm_op.F32x4_splat ] in
            (* four narrow cells widened to i32 lanes *)
            let narrow load extend first =
              cell_address st b (element first)
              @ [ I.Simd_load (load, arg 0 0) ]
              @ List.map vn extend
            in
            List.init (count / per) (fun i ->
                match (d, delta, Ssa_format.quant b.Ssa_buffer.format) with
                | ( (Ssa_op.Decode.I8_dequant32 | Ssa_op.Decode.I16_dequant32),
                    1L,
                    Some (Ssa_format.Per_tensor { scale; zero_point }) ) ->
                    (* [q] exact in binary32, minus the zero point, times the
                       scale: each rounded once, as the scalar decode *)
                    (match d with
                      | Ssa_op.Decode.I8_dequant32 ->
                          narrow Wasm.Simd_load.Load32_zero
                            [
                              Wasm_op.I16x8_extend_low_i8x16_s;
                              Wasm_op.I32x4_extend_low_i16x8_s;
                            ]
                            (per * i)
                      | _ ->
                          narrow Wasm.Simd_load.Load64_zero
                            [ Wasm_op.I32x4_extend_low_i16x8_s ]
                            (per * i))
                    @ [ vn Wasm_op.F32x4_convert_i32x4_s ]
                    @ splat_f32 zero_point
                    @ [ vn Wasm_op.F32x4_sub ]
                    @ splat_f32 scale
                    @ [ vn Wasm_op.F32x4_mul ]
                | Ssa_op.Decode.Bf16_to_f32, 1L, _ ->
                    (* the high half of each binary32 *)
                    narrow Wasm.Simd_load.Load64_zero
                      [ Wasm_op.I32x4_extend_low_i16x8_u ]
                      (per * i)
                    @ [ i32 16; vn Wasm_op.I32x4_shl ]
                | Ssa_op.Decode.F16_to_f32, 1L, _ ->
                    (* binary16 by integer lanes: a normal value rebiased, an
                       infinity or NaN with the binary32 exponent, a
                       subnormal its mantissa times 2^-24; the sign last *)
                    let h = fresh st Wasm_type.V128
                    and e = fresh st Wasm_type.V128
                    and m = fresh st Wasm_type.V128
                    and sign = fresh st Wasm_type.V128 in
                    narrow Wasm.Simd_load.Load64_zero
                      [ Wasm_op.I32x4_extend_low_i16x8_u ]
                      (per * i)
                    @ [ set h ]
                    @ [ get h; i32 15; vn Wasm_op.I32x4_shr_u; i32 31 ]
                    @ [ vn Wasm_op.I32x4_shl; set sign ]
                    @ [ get h; i32 10; vn Wasm_op.I32x4_shr_u ]
                    @ splat_i32 0x1f
                    @ [ vn Wasm_op.V128_and; set e ]
                    @ [ get h ]
                    @ splat_i32 0x3ff
                    @ [ vn Wasm_op.V128_and; set m ]
                    (* subnormal or zero *)
                    @ [ get m; vn Wasm_op.F32x4_convert_i32x4_s ]
                    @ splat_f32 (Float.ldexp 1. (-24))
                    @ [ vn Wasm_op.F32x4_mul; get sign; vn Wasm_op.V128_or ]
                    (* infinity or NaN *)
                    @ [ get sign ]
                    @ splat_i32 0x7f80_0000
                    @ [
                        vn Wasm_op.V128_or; get m; i32 13; vn Wasm_op.I32x4_shl;
                      ]
                    @ [ vn Wasm_op.V128_or ]
                    (* normal *)
                    @ [ get sign; get e ]
                    @ splat_i32 112
                    @ [ vn Wasm_op.I32x4_add; i32 23; vn Wasm_op.I32x4_shl ]
                    @ [
                        vn Wasm_op.V128_or; get m; i32 13; vn Wasm_op.I32x4_shl;
                      ]
                    @ [ vn Wasm_op.V128_or ]
                    @ [ get e ]
                    @ splat_i32 0x1f
                    @ [ vn Wasm_op.I32x4_eq; vn Wasm_op.V128_bitselect ]
                    @ [ get e ]
                    @ splat_i32 0
                    @ [ vn Wasm_op.I32x4_eq; vn Wasm_op.V128_bitselect ]
                | Ssa_op.Decode.F32_exact, 1L, _ ->
                    cell_address st b (element (per * i))
                    @ [ I.Simd_load (Wasm.Simd_load.Load, arg 2 0) ]
                | Ssa_op.Decode.F32_exact, 0L, _ ->
                    cell_address st b (element 0)
                    @ [ I.Simd_load (Wasm.Simd_load.Load32_splat, arg 2 0) ]
                | _ ->
                    List.hd
                      (gather Ssa_type.F32
                         (List.init per (fun j -> lane ((per * i) + j)))))
        | Ssa_type.I64 | Ssa_type.Index | Ssa_type.Pred ->
            refuse (`Unsupported_operation "a vector load of a non-float")
      in
      start @ set_chunks st res chunks
  | Ssa_op.Vec_store { buffer; at; steps; encode; value; lanes } -> (
      let b = find_buffer st buffer in
      let delta = lane_delta b steps in
      let o = fresh st Wasm_type.I32 in
      let count = Ssa_type.Lanes.to_int lanes in
      let regs = regs st value in
      let element k =
        [
          get o;
          index_const (Int64.mul (Int64.of_int k) delta);
          n Wasm_op.I32_add;
        ]
      in
      coord_offset st b at
      @ [ set o ]
      @
      match (encode, Int64.equal delta 1L, elem_of value) with
      | Ssa_op.Encode.F32_exact, true, Ssa_type.F32 ->
          List.concat
            (List.init
               (count / per_register value)
               (fun i ->
                 cell_address st b (element (per_register value * i))
                 @ [
                     get regs.(i);
                     I.Simd_store (Wasm.Simd_store.Store, arg 2 0, 0);
                   ]))
      | Ssa_op.Encode.F32_round, true, Ssa_type.F64 ->
          List.concat
            (List.init (count / 2) (fun i ->
                 cell_address st b (element (2 * i))
                 @ [
                     get regs.(i);
                     vn Wasm_op.F32x4_demote_f64x2_zero;
                     I.Simd_store (Wasm.Simd_store.Store64_lane, arg 2 0, 0);
                   ]))
      | _ ->
          let per = per_register value in
          let e = elem_of value in
          List.concat
            (List.init count (fun k ->
                 cell_address st b (element k)
                 @ [
                     get regs.(k / per); I.Simd_lane (extract_lane e, k mod per);
                   ]
                 @ encode_store encode)))
