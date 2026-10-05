open Ssa_ir
open Ssa_wasm_ctx

let is_f32 (v : Ssa_value.t) =
  Ssa_type.equal v.Ssa_value.ty (Ssa_type.Scalar Ssa_type.F32)

let elem_of (v : Ssa_value.t) =
  match v.Ssa_value.ty with
  | Ssa_type.Vec (s, _) -> s
  | _ -> invalid_arg "Ssa_wasm: a vector operand"

let lanes_of (v : Ssa_value.t) =
  match v.Ssa_value.ty with
  | Ssa_type.Vec (_, l) | Ssa_type.Mask l -> Ssa_type.Lanes.to_int l
  | _ -> invalid_arg "Ssa_wasm: a vector operand"

(* ---- scalar arithmetic ---------------------------------------------------------- *)

let float_op ~f32 (op : Expr.Value.binary_op) =
  match (f32, op) with
  | false, Expr.Value.Add -> Wasm_op.F64_add
  | false, Expr.Value.Div -> Wasm_op.F64_div
  | false, Expr.Value.Mul -> Wasm_op.F64_mul
  | false, Expr.Value.Sub -> Wasm_op.F64_sub
  | true, Expr.Value.Add -> Wasm_op.F32_add
  | true, Expr.Value.Div -> Wasm_op.F32_div
  | true, Expr.Value.Mul -> Wasm_op.F32_mul
  | true, Expr.Value.Sub -> Wasm_op.F32_sub

(* Binary32: the binary64 import on the widened argument, rounded once; a square
   root and a truncation are exact in [f32]. The argument is on the stack. *)
let unary st ~f32 (op : Expr.Value.unary_op) =
  if f32 then
    let wide c =
      [ n Wasm_op.F64_promote_f32; call st c; n Wasm_op.F32_demote_f64 ]
    in
    match op with
    | Expr.Value.Cos -> wide R.Callee.Cos
    | Expr.Value.Erf -> [ call st R.Callee.Erf_f32 ]
    | Expr.Value.Exp -> wide R.Callee.Exp
    | Expr.Value.Log -> wide R.Callee.Log
    | Expr.Value.Sin -> wide R.Callee.Sin
    | Expr.Value.Sqrt -> [ n Wasm_op.F32_sqrt ]
    | Expr.Value.Trunc -> [ n Wasm_op.F32_trunc ]
  else
    match op with
    | Expr.Value.Cos -> [ call st R.Callee.Cos ]
    | Expr.Value.Erf -> [ call st R.Callee.Erf ]
    | Expr.Value.Exp -> [ call st R.Callee.Exp ]
    | Expr.Value.Log -> [ call st R.Callee.Log ]
    | Expr.Value.Sin -> [ call st R.Callee.Sin ]
    | Expr.Value.Sqrt -> [ n Wasm_op.F64_sqrt ]
    | Expr.Value.Trunc -> [ n Wasm_op.F64_trunc ]

let compare_op ~ty (c : Ssa_op.Compare.t) =
  match (ty, c) with
  | `F32, Ssa_op.Compare.Eq -> Wasm_op.F32_eq
  | `F32, Ssa_op.Compare.Lt -> Wasm_op.F32_lt
  | `F64, Ssa_op.Compare.Eq -> Wasm_op.F64_eq
  | `F64, Ssa_op.Compare.Lt -> Wasm_op.F64_lt
  | `I32, Ssa_op.Compare.Eq -> Wasm_op.I32_eq
  | `I32, Ssa_op.Compare.Lt -> Wasm_op.I32_lt_s
  | `I64, Ssa_op.Compare.Eq -> Wasm_op.I64_eq
  | `I64, Ssa_op.Compare.Lt -> Wasm_op.I64_lt_s

(* ---- failures ------------------------------------------------------------------- *)

(* [Value.i64_of_float]'s three rejections: NaN, an infinity, or a finite value
   beyond the [int64] range. *)
let from_float_failure st x =
  let only kind =
    [ i32 (W.kind_index kind); call st R.Callee.Fail_set; i32 1; I.Return ]
  in
  [ get x; get x; n Wasm_op.F64_ne ]
  @ [
      I.If
        ( None,
          only F.Kind.I64_from_float_nan,
          [
            get x;
            n Wasm_op.F64_abs;
            f64 Float.infinity;
            n Wasm_op.F64_eq;
            I.If
              ( None,
                only F.Kind.I64_from_float_infinite,
                fail st F.Kind.I64_from_float_out_of_range
                  [ (0, [ get x; n Wasm_op.I64_reinterpret_f64 ]) ] );
          ] );
    ]

(* The first axis, in order, whose component is outside the buffer's shape names
   the failure; none is a defect of the program, recorded as such. *)
let coord_failure st (b : Ssa_buffer.t) (comps : int list) =
  let exts = extents b in
  let rec chain k = function
    | [] -> [ i32 (W.kind_index F.Kind.Defect); call st R.Callee.Fail_set ]
    | (l, extent) :: rest ->
        [
          get l;
          index_const extent;
          n Wasm_op.I32_ge_u;
          I.If
            ( None,
              [
                i32 Loop_ir.Loop_wasm_ctx.error_address;
                I.I64_const (Int64.of_int k);
                I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset 1));
                i32 Loop_ir.Loop_wasm_ctx.error_address;
              ]
              @ wide [ get l ]
              @ [ I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset 2)) ],
              chain (k + 1) rest );
        ]
  in
  [ i32 (W.kind_index F.Kind.Coord_out_of_range); call st R.Callee.Fail_set ]
  @ [
      i32 Loop_ir.Loop_wasm_ctx.error_address;
      I.I64_const (Int64.of_int (b.Ssa_buffer.id :> int));
      I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset 0));
    ]
  @ List.concat
      (List.mapi
         (fun k l ->
           [ i32 Loop_ir.Loop_wasm_ctx.error_address ]
           @ wide [ get l ]
           @ [ I.Store (Wasm.Store.I64_store, arg 3 (W.slot_offset (3 + k))) ])
         comps)
  @ chain 0 (List.combine comps exts)
  @ [ i32 1; I.Return ]

let check_coord st (b : Ssa_buffer.t) (c : Ssa_value.t Expr.Coord.t) =
  let comps = List.map (reg st) (Expr.Coord.to_list c) in
  let exts = extents b in
  let outside =
    List.concat
      (List.mapi
         (fun k (l, e) ->
           [ get l; index_const e; n Wasm_op.I32_ge_u ]
           @ if k = 0 then [] else [ n Wasm_op.I32_or ])
         (List.combine comps exts))
  in
  outside @ [ I.If (None, coord_failure st b comps, []) ]

(* ---- loads, stores, locals ------------------------------------------------------ *)

let quant_tables st (b : Ssa_buffer.t) =
  Hashtbl.find st.tables (b.Ssa_buffer.id :> int)

(* One cell decoded to the working value a [Decode] names. [at] is the element
   offset instructions; [coord] the access's coordinates, for a per-channel
   decode. *)
let decode st (b : Ssa_buffer.t) (d : Ssa_op.Decode.t) ~address
    ~(access : Ssa_access.t) =
  let load l = address @ [ I.Load (l, arg (Wasm.Load.natural_align l) 0) ] in
  match d with
  | Ssa_op.Decode.Bf16_to_f64 ->
      load Wasm.Load.I32_load16_u
      @ [
          i32 16;
          n Wasm_op.I32_shl;
          n Wasm_op.F32_reinterpret_i32;
          n Wasm_op.F64_promote_f32;
        ]
  | Ssa_op.Decode.Bool_to_f64 ->
      load Wasm.Load.I32_load8_u
      @ [ i32 0; n Wasm_op.I32_ne; n Wasm_op.F64_convert_i32_u ]
  | Ssa_op.Decode.F16_to_f64 ->
      load Wasm.Load.I32_load16_u @ [ call st R.Callee.F16_to_float ]
  | Ssa_op.Decode.F32_to_f64 ->
      load Wasm.Load.F32_load @ [ n Wasm_op.F64_promote_f32 ]
  | Ssa_op.Decode.F64_to_f64 -> load Wasm.Load.F64_load
  | Ssa_op.Decode.I32_to_f64 ->
      load Wasm.Load.I32_load @ [ n Wasm_op.F64_convert_i32_s ]
  | Ssa_op.Decode.I64 -> load Wasm.Load.I64_load
  | Ssa_op.Decode.I64_to_f64 ->
      load Wasm.Load.I64_load @ [ n Wasm_op.F64_convert_i64_s ]
  | Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I8_dequant -> (
      let l =
        if d = Ssa_op.Decode.I8_dequant then Wasm.Load.I32_load8_s
        else Wasm.Load.I32_load16_s
      in
      match Ssa_format.quant b.Ssa_buffer.format with
      | None ->
          invalid_arg "Ssa_wasm: a dequantizing load of an unquantized buffer"
      | Some (Ssa_format.Per_tensor { scale; zero_point }) ->
          [ f64 scale ]
          @ load l
          @ [
              n Wasm_op.F64_convert_i32_s;
              f64 (float_of_int zero_point);
              n Wasm_op.F64_sub;
              n Wasm_op.F64_mul;
            ]
      | Some (Ssa_format.Per_channel _) ->
          let c =
            match access with
            | Ssa_access.Coord c -> c.Expr.Coord.c
            | Ssa_access.Flat _ ->
                invalid_arg "Ssa_wasm: a flat per-channel load"
          in
          let scales, zeros = quant_tables st b in
          let table base =
            [ i32 base ]
            @ read st c
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

let encode_store (e : Ssa_op.Encode.t) =
  match e with
  | Ssa_op.Encode.Bool_nonzero ->
      [ f64 0.; n Wasm_op.F64_ne; I.Store (Wasm.Store.I32_store8, arg 0 0) ]
  | Ssa_op.Encode.F32_round ->
      [ n Wasm_op.F32_demote_f64; I.Store (Wasm.Store.F32_store, arg 2 0) ]
  | Ssa_op.Encode.I64 -> [ I.Store (Wasm.Store.I64_store, arg 3 0) ]

let local_address st local at =
  [ get (reg st local) ]
  @ read st at
  @ [ i32 3; n Wasm_op.I32_shl; n Wasm_op.I32_add ]

(* ---- vectors -------------------------------------------------------------------- *)

let vn op = I.Numeric op

(* The element distance between consecutive lanes of a vector access. *)
let lane_delta (b : Ssa_buffer.t) (steps : int64 Expr.Coord.t) =
  let exts = Array.of_list (extents b) in
  let steps = Array.of_list (Expr.Coord.to_list steps) in
  let stride = Array.make 6 1L in
  for i = 4 downto 0 do
    stride.(i) <- Int64.mul stride.(i + 1) exts.(i + 1)
  done;
  let delta = ref 0L in
  Array.iteri
    (fun i s -> delta := Int64.add !delta (Int64.mul s stride.(i)))
    steps;
  !delta

let splat_op = function
  | Ssa_type.F32 -> Wasm_op.F32x4_splat
  | _ -> Wasm_op.F64x2_splat

let replace_lane = function
  | Ssa_type.F32 -> Wasm.Simd_lane.F32x4_replace
  | _ -> Wasm.Simd_lane.F64x2_replace

let extract_lane = function
  | Ssa_type.F32 -> Wasm.Simd_lane.F32x4_extract
  | _ -> Wasm.Simd_lane.F64x2_extract

(* Registers built from scalar lane values (each an instruction list): splat the
   first lane of a register, replace the rest. *)
let gather elem (lanes : I.t list list) =
  let per = lanes_per_register elem in
  let rec registers = function
    | [] -> []
    | l ->
        let reg = List.filteri (fun i _ -> i < per) l in
        let rest = List.filteri (fun i _ -> i >= per) l in
        reg :: registers rest
  in
  List.map
    (function
      | first :: rest ->
          first
          @ [ vn (splat_op elem) ]
          @ List.concat
              (List.mapi
                 (fun k l -> l @ [ I.Simd_lane (replace_lane elem, k + 1) ])
                 rest)
      | [] -> assert false)
    (registers lanes)

let vbin elem (op : Expr.Value.binary_op) =
  match (elem, op) with
  | Ssa_type.F32, Expr.Value.Add -> Wasm_op.F32x4_add
  | Ssa_type.F32, Expr.Value.Div -> Wasm_op.F32x4_div
  | Ssa_type.F32, Expr.Value.Mul -> Wasm_op.F32x4_mul
  | Ssa_type.F32, Expr.Value.Sub -> Wasm_op.F32x4_sub
  | _, Expr.Value.Add -> Wasm_op.F64x2_add
  | _, Expr.Value.Div -> Wasm_op.F64x2_div
  | _, Expr.Value.Mul -> Wasm_op.F64x2_mul
  | _, Expr.Value.Sub -> Wasm_op.F64x2_sub

(* The lanes a mask would hold for vectors of this element. *)
let shape_of_elem = function Ssa_type.F32 -> Narrow | _ -> Wide

(* A mask re-held in the other shape, lane by lane. *)
let reshape_mask st (m : Ssa_value.t) ~(target : mask_shape) =
  let src = regs st m in
  let have = mask_shape st m in
  if have = target then Array.to_list (Array.map (fun r -> [ get r ]) src)
  else
    let lanes = lanes_of m in
    let lane_value k =
      match have with
      | Wide ->
          [
            get src.(k / 2); I.Simd_lane (Wasm.Simd_lane.I64x2_extract, k mod 2);
          ]
      | Narrow ->
          [
            get src.(k / 4); I.Simd_lane (Wasm.Simd_lane.I32x4_extract, k mod 4);
          ]
    in
    match target with
    | Narrow ->
        List.init (lanes / 4) (fun j ->
            let l k = lane_value ((4 * j) + k) @ [ n Wasm_op.I32_wrap_i64 ] in
            l 0
            @ [ vn Wasm_op.I32x4_splat ]
            @ List.concat
                (List.init 3 (fun k ->
                     l (k + 1)
                     @ [ I.Simd_lane (Wasm.Simd_lane.I32x4_replace, k + 1) ])))
    | Wide ->
        List.init (lanes / 2) (fun j ->
            let l k =
              lane_value ((2 * j) + k) @ [ n Wasm_op.I64_extend_i32_s ]
            in
            l 0
            @ [ vn Wasm_op.I64x2_splat ]
            @ l 1
            @ [ I.Simd_lane (Wasm.Simd_lane.I64x2_replace, 1) ])
