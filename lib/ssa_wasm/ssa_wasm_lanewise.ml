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

(* ---- lane-wise operations ------------------------------------------------------- *)

let lanewise st (inner : Ssa_op.t) (res : Ssa_value.t) =
  let src v i = get (regs st v).(i) in
  let chunks_of v f = Array.to_list (Array.mapi (fun i _ -> f i) (regs st v)) in
  match inner with
  | Ssa_op.Convert (Ssa_op.Convert.F32_to_f64, a) ->
      let per_in = lanes_per_register Ssa_type.F32
      and per_out = lanes_per_register Ssa_type.F64 in
      let lane k =
        [
          src a (Ssa_type.Lane.chunk k ~per:per_in :> int);
          I.Simd_lane
            ( Wasm.Simd_lane.F32x4_extract,
              (Ssa_type.Lane.slot k ~per:per_in :> int) );
        ]
      in
      set_chunks st res
        (List.map
           (fun i ->
             let first = Ssa_type.Lane.first i ~per:per_out in
             if
               Ssa_type.Lane.equal
                 (Ssa_type.Lane.slot first ~per:per_in)
                 Ssa_type.Lane.zero
             then
               [
                 src a (Ssa_type.Lane.chunk first ~per:per_in :> int);
                 vn Wasm_op.F64x2_promote_low_f32x4;
               ]
             else
               lane first
               @ [ n Wasm_op.F64_promote_f32; vn Wasm_op.F64x2_splat ]
               @ lane (Ssa_type.Lane.add first ~by:(Ssa_type.Lane.of_int 1))
               @ [
                   n Wasm_op.F64_promote_f32;
                   I.Simd_lane (Wasm.Simd_lane.F64x2_replace, 1);
                 ])
           (Ssa_type.Chunk.all (lanes_of a) ~per:per_out))
  | Ssa_op.Convert (Ssa_op.Convert.F64_to_f32, a) ->
      let per_in = lanes_per_register Ssa_type.F64
      and per_out = lanes_per_register Ssa_type.F32 in
      let lane k =
        [
          src a (Ssa_type.Lane.chunk k ~per:per_in :> int);
          I.Simd_lane
            ( Wasm.Simd_lane.F64x2_extract,
              (Ssa_type.Lane.slot k ~per:per_in :> int) );
        ]
      in
      let past first n = Ssa_type.Lane.add first ~by:(Ssa_type.Lane.of_int n) in
      set_chunks st res
        (List.map
           (fun j ->
             let first = Ssa_type.Lane.first j ~per:per_out in
             [
               src a (Ssa_type.Lane.chunk first ~per:per_in :> int);
               vn Wasm_op.F32x4_demote_f64x2_zero;
             ]
             @ lane (past first 2)
             @ [
                 n Wasm_op.F32_demote_f64;
                 I.Simd_lane (Wasm.Simd_lane.F32x4_replace, 2);
               ]
             @ lane (past first 3)
             @ [
                 n Wasm_op.F32_demote_f64;
                 I.Simd_lane (Wasm.Simd_lane.F32x4_replace, 3);
               ])
           (Ssa_type.Chunk.all (lanes_of a) ~per:per_out))
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
        refuse st
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
      | Expr.Value.Round_bf16 -> invalid_arg Ssa_wasm_op.bf16_unary
      | Expr.Value.Cos | Expr.Value.Erf | Expr.Value.Exp | Expr.Value.Log
      | Expr.Value.Sin ->
          (* no vector instruction: each lane runs the scalar callee *)
          let per = per_register a in
          let lane k =
            [
              src a (Ssa_type.Lane.chunk k ~per :> int);
              I.Simd_lane
                ( (if f32 then Wasm.Simd_lane.F32x4_extract
                   else Wasm.Simd_lane.F64x2_extract),
                  (Ssa_type.Lane.slot k ~per :> int) );
            ]
            @ unary st ~f32 op
          in
          let lanes = List.map lane (Ssa_type.Lane.all (lanes_of a)) in
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
  | _ ->
      refuse st (`Unsupported_operation "an operation with no lane-wise form")
