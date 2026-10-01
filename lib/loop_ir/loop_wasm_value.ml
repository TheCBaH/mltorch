open Loop_wasm_ctx

let rec index st : Loop_index.t -> I.t list = function
  | Loop_index.Add (a, b) ->
      let a = index st a in
      let b = index st b in
      a @ b @ [ n Wasm_op.I32_add ]
  | Loop_index.Ceil_div_pos (a, d) ->
      let a = index st a in
      a @ [ positive st d; call st R.Callee.Ceil_div ]
  | Loop_index.Clamp_low a ->
      let a = index st a in
      let t = fresh st Wasm_type.I32 in
      a @ [ I.Local_tee t; i32 0; get t; i32 0; n Wasm_op.I32_gt_s; I.Select ]
  | Loop_index.Const k -> [ int_const st k ]
  | Loop_index.Floor_div_pos (a, d) ->
      let a = index st a in
      a @ [ positive st d; call st R.Callee.Floor_div ]
  | Loop_index.Max (a, b) -> select2 st a b Wasm_op.I32_gt_s
  | Loop_index.Min (a, b) -> select2 st a b Wasm_op.I32_lt_s
  | Loop_index.Scale (k, a) ->
      let a = index st a in
      [ int_const st k ] @ a @ [ n Wasm_op.I32_mul ]
  | Loop_index.Temp t -> [ get (xtemp st t) ]
  | Loop_index.Var v -> [ get (var st v) ]

and positive st d =
  if d > 0 then int_const st d else refuse st (`Index_constant_out_of_range d)

(* [a] then [b] each once, then the one [cmp] picks wins. *)
and select2 st a b cmp =
  let a = index st a in
  let b = index st b in
  let ta = fresh st Wasm_type.I32 in
  let tb = fresh st Wasm_type.I32 in
  a
  @ [ set ta ]
  @ b
  @ [ set tb; get ta; get tb; get ta; get tb; n cmp; I.Select ]

(* One node per checked operation, post-order, as [Loop_js.overflow_nodes]:
   the first to leave the [int32] domain is the one reported. Each node's value
   is formed in [i64] from operands that already passed their own check, so
   neither the value nor the check can wrap. *)
type overflow_node = {
  op : int;
  value : I.t list;
  lhs : I.t list;
  rhs : I.t list;
}

let overflow_nodes st i =
  let wide l = l @ [ n Wasm_op.I64_extend_i32_s ] in
  let rec go acc (i : Loop_index.t) =
    match i with
    | Loop_index.Add (a, b) ->
        let acc = go (go acc a) b in
        let lhs = wide (index st a) in
        let rhs = wide (index st b) in
        { op = 0; value = lhs @ rhs @ [ n Wasm_op.I64_add ]; lhs; rhs } :: acc
    | Loop_index.Scale (k, a) ->
        let acc = go acc a in
        let lhs = [ I.I64_const (Int64.of_int k) ] in
        let rhs = wide (index st a) in
        { op = 1; value = lhs @ rhs @ [ n Wasm_op.I64_mul ]; lhs; rhs } :: acc
    | Loop_index.Ceil_div_pos (a, _)
    | Loop_index.Clamp_low a
    | Loop_index.Floor_div_pos (a, _) ->
        go acc a
    | Loop_index.Max (a, b) | Loop_index.Min (a, b) -> go (go acc a) b
    | Loop_index.Const _ | Loop_index.Temp _ | Loop_index.Var _ -> acc
  in
  List.rev (go [] i)

(* [value] outside [-2^31, 2^31): shifted into [0, 2^32) it is above 2^32 - 1
   when read as unsigned. *)
let outside_int32 value =
  value
  @ [
      I.I64_const 0x8000_0000L;
      n Wasm_op.I64_add;
      I.I64_const 0xFFFF_FFFFL;
      n Wasm_op.I64_gt_u;
    ]

(* Zero-extends the 0/1 of a comparison where an [i64] is wanted. *)
let wide_index st i = index st i @ [ n Wasm_op.I64_extend_i32_s ]

type addr = At of Loop_index.coord | Flat of Loop_index.t

let addr_index (b : Loop_buffer.t) = function
  | At c -> offset_index b c
  | Flat i -> i

(* The byte address of cell [addr]: the buffer's base plus the cell offset
   scaled by the cell width. The memory plan bounds every buffer so this stays
   in a positive [int32]; the check that the cell is in range is a [Fail_if]. *)
let cell_address st b addr =
  let off = index st (addr_index b addr) in
  let log2 = cell_log2 b in
  [ get (buffer_param st b) ]
  @ off
  @ (if log2 = 0 then [] else [ i32 log2; n Wasm_op.I32_shl ])
  @ [ n Wasm_op.I32_add ]

let quant_of (b : Loop_buffer.t) =
  match b.Loop_buffer.sg.Tensor_sig.quant with
  | Some q -> q
  | None -> invalid_arg "Loop_wasm: a quantized buffer without parameters"

let load_cell st (b : Loop_buffer.t) addr =
  let at = cell_address st b addr in
  let load l = at @ [ I.Load (l, arg (Wasm.Load.natural_align l) 0) ] in
  match fmt_of b with
  | "bf16" ->
      load Wasm.Load.I32_load16_u
      @ [
          i32 16;
          n Wasm_op.I32_shl;
          n Wasm_op.F32_reinterpret_i32;
          n Wasm_op.F64_promote_f32;
        ]
  | "bool" ->
      load Wasm.Load.I32_load8_u
      @ [ i32 0; n Wasm_op.I32_ne; n Wasm_op.F64_convert_i32_u ]
  | "f16" -> load Wasm.Load.I32_load16_u @ [ call st R.Callee.F16_to_float ]
  | "f32" -> load Wasm.Load.F32_load @ [ n Wasm_op.F64_promote_f32 ]
  | "f64" -> load Wasm.Load.F64_load
  | "i32" -> load Wasm.Load.I32_load @ [ n Wasm_op.F64_convert_i32_s ]
  | "i64" -> load Wasm.Load.I64_load @ [ n Wasm_op.F64_convert_i64_s ]
  | ("i16" | "i8") as f -> (
      let l =
        if f = "i8" then Wasm.Load.I32_load8_s else Wasm.Load.I32_load16_s
      in
      let q = quant_of b in
      match Quant.channel_count q with
      | None ->
          let scale, zero = Quant.params q ~c:(Dim.index 0) in
          [ f64 scale ]
          @ load l
          @ [
              n Wasm_op.F64_convert_i32_s;
              f64 (float_of_int zero);
              n Wasm_op.F64_sub;
              n Wasm_op.F64_mul;
            ]
      | Some _ ->
          let c =
            match addr with
            | At c -> c
            | Flat _ ->
                invalid_arg "Loop_wasm: a flat load of a per-channel buffer"
          in
          let scales, zeros =
            Hashtbl.find st.tables (Tensor_id.to_int b.Loop_buffer.id)
          in
          let table base =
            [ i32 base ]
            @ index st (Expr.Coord.get c Expr.Axis.C)
            @ [
                i32 3;
                n Wasm_op.I32_shl;
                n Wasm_op.I32_add;
                I.Load (Wasm.Load.F64_load, arg 3 0);
              ]
          in
          table scales @ load l
          @ [ n Wasm_op.F64_convert_i32_s ]
          @ table zeros
          @ [ n Wasm_op.F64_sub; n Wasm_op.F64_mul ])
  | f -> invalid_arg ("Loop_wasm: no decode for format " ^ f)

let load_i64 st b addr =
  cell_address st b addr @ [ I.Load (Wasm.Load.I64_load, arg 3 0) ]

let array_address st a i =
  let off = Hashtbl.find st.arrays (Loop_array.to_int a) in
  [ i32 off ] @ index st i @ [ i32 3; n Wasm_op.I32_shl; n Wasm_op.I32_add ]

let rec num st : float Loop_expr.t -> I.t list = function
  | Loop_expr.Array_get (a, i) ->
      array_address st a i @ [ I.Load (Wasm.Load.F64_load, arg 3 0) ]
  | Loop_expr.Binary (op, a, b) ->
      let a = num st a in
      let b = num st b in
      a @ b
      @ [
          n
            (match op with
            | Expr.Value.Add -> Wasm_op.F64_add
            | Expr.Value.Div -> Wasm_op.F64_div
            | Expr.Value.Mul -> Wasm_op.F64_mul
            | Expr.Value.Sub -> Wasm_op.F64_sub);
        ]
  | Loop_expr.Const x -> [ f64 x ]
  | Loop_expr.Float_max (a, b) ->
      let a = num st a in
      let b = num st b in
      a @ b @ [ n Wasm_op.F64_max ]
  | Loop_expr.I64_to_float a -> big st a @ [ n Wasm_op.F64_convert_i64_s ]
  | Loop_expr.Load (b, c) -> load_cell st b (At c)
  | Loop_expr.Load_flat (b, i) -> load_cell st b (Flat i)
  | Loop_expr.Round_f32 a ->
      num st a @ [ n Wasm_op.F32_demote_f64; n Wasm_op.F64_promote_f32 ]
  | Loop_expr.Select (p, a, b) ->
      let p = pred st p in
      let a = num st a in
      let b = num st b in
      p @ [ I.If (Some Wasm_type.F64, a, b) ]
  | Loop_expr.Temp (Loop_carrier.Float, t) -> [ get (ftemp st t) ]
  | Loop_expr.Unary (op, a) -> (
      let a = num st a in
      match op with
      | Expr.Value.Cos -> a @ [ call st R.Callee.Cos ]
      | Expr.Value.Erf -> a @ [ call st R.Callee.Erf ]
      | Expr.Value.Exp -> a @ [ call st R.Callee.Exp ]
      | Expr.Value.Log -> a @ [ call st R.Callee.Log ]
      | Expr.Value.Sin -> a @ [ call st R.Callee.Sin ]
      | Expr.Value.Sqrt -> a @ [ n Wasm_op.F64_sqrt ]
      | Expr.Value.Trunc -> a @ [ n Wasm_op.F64_trunc ])
  | Loop_expr.Value_of_index i -> index st i @ [ n Wasm_op.F64_convert_i32_s ]

and big st : int64 Loop_expr.t -> I.t list = function
  | Loop_expr.Float_to_i64 a -> num st a @ [ n Wasm_op.I64_trunc_sat_f64_s ]
  | Loop_expr.I64_binary (op, a, b) -> (
      let a = big st a in
      let b = big st b in
      match op with
      | Expr.Value.I64_add -> a @ b @ [ n Wasm_op.I64_add ]
      | Expr.Value.I64_div -> a @ b @ [ call st R.Callee.I64_div ]
      | Expr.Value.I64_mul -> a @ b @ [ n Wasm_op.I64_mul ]
      | Expr.Value.I64_sub -> a @ b @ [ n Wasm_op.I64_sub ])
  | Loop_expr.I64_const k -> [ I.I64_const k ]
  | Loop_expr.I64_of_index i -> wide_index st i
  | Loop_expr.Load_i64 (b, c) -> load_i64 st b (At c)
  | Loop_expr.Load_i64_flat (b, i) -> load_i64 st b (Flat i)
  | Loop_expr.Select (p, a, b) ->
      let p = pred st p in
      let a = big st a in
      let b = big st b in
      p @ [ I.If (Some Wasm_type.I64, a, b) ]
  | Loop_expr.Temp (Loop_carrier.Int64, t) -> [ get (itemp st t) ]

and pred st : Loop_expr.pred -> I.t list = function
  | Loop_bool.I64_eq (a, b) ->
      let a = big st a in
      let b = big st b in
      a @ b @ [ n Wasm_op.I64_eq ]
  | Loop_bool.I64_lt (a, b) ->
      let a = big st a in
      let b = big st b in
      a @ b @ [ n Wasm_op.I64_lt_s ]
  | Loop_bool.Index_eq (a, b) ->
      let a = index st a in
      let b = index st b in
      a @ b @ [ n Wasm_op.I32_eq ]
  | Loop_bool.Index_lt (a, b) ->
      let a = index st a in
      let b = index st b in
      a @ b @ [ n Wasm_op.I32_lt_s ]
  | Loop_bool.Index_overflows i -> (
      match overflow_nodes st i with
      | [] -> [ i32 0 ]
      | first :: rest ->
          List.fold_left
            (fun acc node ->
              acc @ outside_int32 node.value @ [ n Wasm_op.I32_or ])
            (outside_int32 first.value)
            rest)
  | Loop_bool.Not p -> pred st p @ [ n Wasm_op.I32_eqz ]
  | Loop_bool.Or (p, q) ->
      let p = pred st p in
      let q = pred st q in
      p @ [ I.If (Some Wasm_type.I32, [ i32 1 ], q) ]
  | Loop_bool.Out_of_range (i, extent) ->
      index st i @ [ int_const st extent; n Wasm_op.I32_ge_u ]
  | Loop_bool.Pool_better (best, value) ->
      (* The candidate wins on strict greater-than or on NaN. *)
      let best = num st best in
      let value = num st value in
      let b = fresh st Wasm_type.F64 in
      let v = fresh st Wasm_type.F64 in
      best
      @ [ set b ]
      @ value
      @ [
          set v;
          get v;
          get b;
          n Wasm_op.F64_gt;
          get v;
          get v;
          n Wasm_op.F64_ne;
          n Wasm_op.I32_or;
        ]
  | Loop_bool.Value_eq (a, b) ->
      let a = num st a in
      let b = num st b in
      a @ b @ [ n Wasm_op.F64_eq ]
  | Loop_bool.Value_lt (a, b) ->
      let a = num st a in
      let b = num st b in
      a @ b @ [ n Wasm_op.F64_lt ]
