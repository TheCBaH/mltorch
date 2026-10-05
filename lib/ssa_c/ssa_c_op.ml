open Ssa_ir
module R = Loop_ir.Loop_c_runtime
module F = Loop_ir.Loop_js_failure
module LF = Loop_ir.Loop_failure
open Ssa_c_ctx

let is_f32 (v : Ssa_value.t) =
  Ssa_type.equal v.Ssa_value.ty (Ssa_type.Scalar Ssa_type.F32)

let kind_num k = string_of_int (R.kind_index k)
let fail kind fields = fail_record kind fields

let sym : Expr.Value.binary_op -> string = function
  | Expr.Value.Add -> "+"
  | Expr.Value.Div -> "/"
  | Expr.Value.Mul -> "*"
  | Expr.Value.Sub -> "-"

(* A binary32 transcendental is the binary64 function on the widened argument,
   rounded once; a square root and a truncation are exact in [float]. *)
let unary cx ~f32 (op : Expr.Value.unary_op) a =
  if f32 then
    let wide f = "(float)" ^ f ^ "((double)" ^ a ^ ")" in
    match op with
    | Expr.Value.Cos -> wide "cos"
    | Expr.Value.Erf -> call cx R.Name.Erf_f32 [ a ]
    | Expr.Value.Exp -> wide "exp"
    | Expr.Value.Log -> wide "log"
    | Expr.Value.Sin -> wide "sin"
    | Expr.Value.Sqrt -> "sqrtf(" ^ a ^ ")"
    | Expr.Value.Trunc -> "truncf(" ^ a ^ ")"
  else
    match op with
    | Expr.Value.Cos -> "cos(" ^ a ^ ")"
    | Expr.Value.Erf -> call cx R.Name.Erf [ a ]
    | Expr.Value.Exp -> "exp(" ^ a ^ ")"
    | Expr.Value.Log -> "log(" ^ a ^ ")"
    | Expr.Value.Sin -> "sin(" ^ a ^ ")"
    | Expr.Value.Sqrt -> "sqrt(" ^ a ^ ")"
    | Expr.Value.Trunc -> "trunc(" ^ a ^ ")"

let const (c : Ssa_const.t) =
  match c with
  | Ssa_const.F32 x -> Loop_ir.Loop_numerics.f32_literal x
  | Ssa_const.F64 x -> float_lit x
  | Ssa_const.I64 n | Ssa_const.Index n -> i64_lit n
  | Ssa_const.Pred b -> if b then "1" else "0"

(* ---- addresses ---------------------------------------------------------------- *)

let extents (b : Ssa_buffer.t) = Expr.Coord.to_list b.Ssa_buffer.extents

(* The row-major element offset of a coordinate, folded left to right. *)
let coord_offset cx (b : Ssa_buffer.t) (c : Ssa_value.t Expr.Coord.t) =
  match (Expr.Coord.to_list c, extents b) with
  | first :: rest, _ :: exts ->
      List.fold_left2
        (fun acc comp e ->
          Printf.sprintf "(%s * %s + %s)" acc (i64_lit e) (name cx comp))
        (name cx first) rest exts
  | _ -> invalid_arg "Ssa_c: a coordinate has six components"

let access_offset cx b = function
  | Ssa_access.Coord c -> coord_offset cx b c
  | Ssa_access.Flat o -> name cx o

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

let decode_cell cx (d : Ssa_op.Decode.t) cell =
  match d with
  | Ssa_op.Decode.Bf16_to_f64 -> call cx R.Name.Bf16_to_float [ cell ]
  | Ssa_op.Decode.Bool_to_f64 -> "(" ^ cell ^ " != 0 ? 1.0 : 0.0)"
  | Ssa_op.Decode.F16_to_f64 -> call cx R.Name.F16_to_float [ cell ]
  | Ssa_op.Decode.F32_to_f64 | Ssa_op.Decode.I32_to_f64
  | Ssa_op.Decode.I64_to_f64 ->
      "(double)" ^ cell
  | Ssa_op.Decode.F64_to_f64 | Ssa_op.Decode.I64 -> cell
  | Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I8_dequant ->
      invalid_arg "Ssa_c: a dequantizing load reached the emitter"

let encode_value (e : Ssa_op.Encode.t) v =
  match e with
  | Ssa_op.Encode.Bool_nonzero -> "(" ^ v ^ " != 0.0 ? 1 : 0)"
  | Ssa_op.Encode.F32_round -> "(float)" ^ v
  | Ssa_op.Encode.I64 -> v

(* ---- the checks an access owes ------------------------------------------------ *)

let check_coord cx depth (b : Ssa_buffer.t) (c : Ssa_value.t Expr.Coord.t) =
  let comps = List.map (name cx) (Expr.Coord.to_list c) in
  let exts = extents b in
  let outside =
    List.map2
      (fun comp e -> Printf.sprintf "%s < 0 || %s >= %s" comp comp (i64_lit e))
      comps exts
  in
  line cx depth "if (%s) {" (String.concat " || " outside);
  line cx (depth + 1) "const int64_t ext[6] = {%s};"
    (String.concat ", " (List.map i64_lit exts));
  line cx (depth + 1) "const int64_t co[6] = {%s};" (String.concat ", " comps);
  line cx (depth + 1) "return %s;"
    (call cx R.Name.Coord_failure
       [ "err"; int_lit (b.Ssa_buffer.id :> int); "ext"; "co" ]);
  line cx depth "}"

(* ---- vectors ------------------------------------------------------------------ *)

let vec_parts (ty : Ssa_type.t) =
  match ty with
  | Ssa_type.Vec (Ssa_type.F32, l) -> (`F32, l)
  | Ssa_type.Vec (Ssa_type.F64, l) -> (`F64, l)
  | Ssa_type.Mask l -> (`Mask, l)
  | Ssa_type.Effect | Ssa_type.Local | Ssa_type.Scalar _ | Ssa_type.Vec _ ->
      invalid_arg "Ssa_c: a vector type"

let vtype cx ty =
  let elem, l = vec_parts ty in
  vector_type cx ~elem l

let lanes_of ty = Ssa_type.Lanes.to_int (snd (vec_parts ty))
let elem_f32 ty = fst (vec_parts ty) = `F32

(* A helper applying a scalar expression to every lane. *)
let lane_helper cx ~prefix ~ty ~arity ~body =
  let t = vtype cx ty in
  let n = lanes_of ty in
  let fname = Printf.sprintf "%s_%s" prefix t in
  let params =
    String.concat ", "
      (List.init arity (fun i -> Printf.sprintf "%s %c" t (Char.chr (97 + i))))
  in
  let args =
    body (List.init arity (fun i -> Printf.sprintf "%c[k]" (Char.chr (97 + i))))
  in
  vector_helper cx fname
    (Printf.sprintf
       "static inline %s %s(%s) { %s r; for (int k = 0; k < %d; k++) r[k] = \
        %s; return r; }"
       t fname params t n args);
  fname

let splat_helper cx ty =
  let t = vtype cx ty in
  let n = lanes_of ty in
  let fname = "ssa_splat_" ^ t in
  let elem = match fst (vec_parts ty) with `F32 -> "float" | _ -> "double" in
  vector_helper cx fname
    (Printf.sprintf
       "static inline %s %s(%s x) { %s r; for (int k = 0; k < %d; k++) r[k] = \
        x; return r; }"
       t fname elem t n);
  fname

let mask_splat_helper cx ty =
  let t = vtype cx ty in
  let n = lanes_of ty in
  let fname = "ssa_msplat_" ^ t in
  vector_helper cx fname
    (Printf.sprintf
       "static inline %s %s(int p) { %s r; for (int k = 0; k < %d; k++) r[k] = \
        p ? -1 : 0; return r; }"
       t fname t n);
  fname

(* A mask as the lanes of the vector type it selects between. *)
let select_helper cx ~mask_ty ~ty =
  let t = vtype cx ty in
  let mt = vtype cx mask_ty in
  let fname = "ssa_vsel_" ^ t in
  (match fst (vec_parts ty) with
  | `F64 ->
      vector_helper cx fname
        (Printf.sprintf
           "static inline %s %s(%s m, %s a, %s b) { return (%s)(((%s)a & m) | \
            ((%s)b & ~m)); }"
           t fname mt t t t mt mt)
  | `F32 ->
      let n = lanes_of ty in
      let narrow = vector_type cx ~elem:`Mask32 (Ssa_type.Lanes.of_int n) in
      vector_helper cx fname
        (Printf.sprintf
           "static inline %s %s(%s m, %s a, %s b) { %s w = \
            __builtin_convertvector(m, %s); return (%s)(((%s)a & w) | ((%s)b & \
            ~w)); }"
           t fname mt t t narrow narrow t narrow narrow)
  | `Mask ->
      vector_helper cx fname
        (Printf.sprintf
           "static inline %s %s(%s m, %s a, %s b) { return (a & m) | (b & ~m); \
            }"
           t fname mt t t));
  fname

let operand_ty (v : Ssa_value.t) = v.Ssa_value.ty

(* The result of a lane-wise operation, as an expression. *)
let lanewise cx (inner : Ssa_op.t) ~(result : Ssa_value.t) =
  let n = name cx in
  let rty = result.Ssa_value.ty in
  match inner with
  | Ssa_op.Convert ((Ssa_op.Convert.F32_to_f64 | Ssa_op.Convert.F64_to_f32), a)
    ->
      Printf.sprintf "__builtin_convertvector(%s, %s)" (n a) (vtype cx rty)
  | Ssa_op.Float_binary (op, a, b) ->
      Printf.sprintf "(%s %s %s)" (n a) (sym op) (n b)
  | Ssa_op.Float_compare (c, a, b) ->
      let cmp =
        Printf.sprintf "(%s %s %s)" (n a)
          (match c with Ssa_op.Compare.Eq -> "==" | Ssa_op.Compare.Lt -> "<")
          (n b)
      in
      if elem_f32 (operand_ty a) then
        Printf.sprintf "__builtin_convertvector(%s, %s)" cmp (vtype cx rty)
      else Printf.sprintf "(%s)%s" (vtype cx rty) cmp
  | Ssa_op.Float_max (a, b) ->
      let f32 = elem_f32 rty in
      let fn =
        if f32 then call cx R.Name.Float_max_f32 []
        else call cx R.Name.Float_max []
      in
      ignore fn;
      let name_of = if f32 then R.Name.Float_max_f32 else R.Name.Float_max in
      use cx name_of;
      let h =
        lane_helper cx ~prefix:"ssa_vmax" ~ty:rty ~arity:2 ~body:(function
          | [ x; y ] -> R.Name.to_string name_of ^ "(" ^ x ^ ", " ^ y ^ ")"
          | _ -> assert false)
      in
      Printf.sprintf "%s(%s, %s)" h (n a) (n b)
  | Ssa_op.Float_fma (a, b, c) ->
      let f32 = elem_f32 rty in
      let h =
        lane_helper cx ~prefix:"ssa_vfma" ~ty:rty ~arity:3 ~body:(function
          | [ x; y; z ] ->
              (if f32 then "fmaf(" else "fma(") ^ x ^ ", " ^ y ^ ", " ^ z ^ ")"
          | _ -> assert false)
      in
      Printf.sprintf "%s(%s, %s, %s)" h (n a) (n b) (n c)
  | Ssa_op.Float_unary (op, a) ->
      let f32 = elem_f32 rty in
      let tag =
        match op with
        | Expr.Value.Cos -> "cos"
        | Expr.Value.Erf -> "erf"
        | Expr.Value.Exp -> "exp"
        | Expr.Value.Log -> "log"
        | Expr.Value.Sin -> "sin"
        | Expr.Value.Sqrt -> "sqrt"
        | Expr.Value.Trunc -> "trunc"
      in
      let h =
        lane_helper cx ~prefix:("ssa_v" ^ tag) ~ty:rty ~arity:1 ~body:(function
          | [ x ] -> unary cx ~f32 op x
          | _ -> assert false)
      in
      Printf.sprintf "%s(%s)" h (n a)
  | Ssa_op.Pool_better (best, value) ->
      let vt = operand_ty value in
      let mt = vtype cx rty in
      let fname = "ssa_vpool_" ^ vtype cx vt in
      let count = lanes_of vt in
      vector_helper cx fname
        (Printf.sprintf
           "static inline %s %s(%s best, %s value) { %s r; for (int k = 0; k < \
            %d; k++) r[k] = (value[k] > best[k] || value[k] != value[k]) ? -1 \
            : 0; return r; }"
           mt fname (vtype cx vt) (vtype cx vt) mt count);
      Printf.sprintf "%s(%s, %s)" fname (n best) (n value)
  | Ssa_op.Pred_not a -> Printf.sprintf "(~%s)" (n a)
  | Ssa_op.Pred_or (a, b) -> Printf.sprintf "(%s | %s)" (n a) (n b)
  | Ssa_op.Select (p, a, b) ->
      let h = select_helper cx ~mask_ty:(operand_ty p) ~ty:rty in
      Printf.sprintf "%s(%s, %s, %s)" h (n p) (n a) (n b)
  | _ -> invalid_arg "Ssa_c: an operation with no lane-wise form"

(* ---- one instruction ---------------------------------------------------------- *)

let first_result cx (i : Ssa_instr.t) =
  match i.Ssa_instr.results with
  | r :: _ -> (r, define cx r)
  | [] -> invalid_arg "Ssa_c: an instruction without a result"

let assign cx depth res expr = line cx depth "%s = %s;" res expr

let instr cx depth (i : Ssa_instr.t) =
  let n = name cx in
  let op = i.Ssa_instr.op in
  (* an operation with no value result defines nothing *)
  let no_result () = () in
  match op with
  | Ssa_op.Mark _ | Ssa_op.Mark_lanes _ -> no_result ()
  | Ssa_op.Check_access { buffer; at } -> (
      let b = find_buffer cx buffer in
      match at with
      | Ssa_access.Coord c -> check_coord cx depth b c
      | Ssa_access.Flat _ -> ())
  | Ssa_op.Check_gather { raw; extent } ->
      line cx depth "if (%s < %s || %s >= %s) %s" (n raw)
        (i64_lit (Int64.neg extent))
        (n raw) (i64_lit extent)
        (fail F.Kind.Gather_index_out_of_range
           [ n raw; Int64.to_string extent ])
  | Ssa_op.Check_local { var; at; extent } ->
      let s =
        site cx
          (LF.Local_out_of_range
             {
               local = var;
               index = Loop_ir.Loop_index.Const 0;
               extent = Int64.to_int extent;
             })
      in
      line cx depth "if (%s < 0 || %s >= %s) %s" (n at) (n at) (i64_lit extent)
        (fail F.Kind.Unbound_local [ string_of_int s ])
  | Ssa_op.Check_scan { var; row; lane; row_extent; lane_extent } ->
      let record which s extent =
        fail F.Kind.Scan_projection
          [
            which;
            (if Option.is_some var then "1" else "0");
            n row;
            n lane;
            Int64.to_string extent;
            string_of_int s;
          ]
      in
      let row_site =
        site cx
          (LF.Scan_row_out_of_range
             {
               local = var;
               row = Loop_ir.Loop_index.Const 0;
               lane = Loop_ir.Loop_index.Const 0;
               extent = Int64.to_int row_extent;
             })
      in
      let lane_site =
        site cx
          (LF.Scan_lane_out_of_range
             {
               local = var;
               row = Loop_ir.Loop_index.Const 0;
               lane = Loop_ir.Loop_index.Const 0;
               extent = Int64.to_int lane_extent;
             })
      in
      line cx depth "if (%s < 0 || %s >= %s) %s" (n row) (n row)
        (i64_lit row_extent)
        (record "1" row_site row_extent);
      line cx depth "if (%s < 0 || %s >= %s) %s" (n lane) (n lane)
        (i64_lit lane_extent)
        (record "0" lane_site lane_extent)
  | Ssa_op.Const c ->
      let _, r = first_result cx i in
      assign cx depth r (const c)
  | Ssa_op.Convert (c, a) ->
      let _, r = first_result cx i in
      assign cx depth r
        (match c with
        | Ssa_op.Convert.F32_to_f64 | Ssa_op.Convert.I64_to_f64
        | Ssa_op.Convert.Index_to_f64 ->
            "(double)" ^ n a
        | Ssa_op.Convert.F64_to_f32 | Ssa_op.Convert.I64_to_f32 ->
            "(float)" ^ n a
        | Ssa_op.Convert.Index_to_i64 -> n a)
  | Ssa_op.Float_binary (o, a, b) ->
      let _, r = first_result cx i in
      assign cx depth r (Printf.sprintf "(%s %s %s)" (n a) (sym o) (n b))
  | Ssa_op.Float_compare (c, a, b) ->
      let _, r = first_result cx i in
      assign cx depth r
        (Printf.sprintf "(%s %s %s)" (n a)
           (match c with Ssa_op.Compare.Eq -> "==" | Ssa_op.Compare.Lt -> "<")
           (n b))
  | Ssa_op.Float_fma (a, b, c) ->
      let res, r = first_result cx i in
      assign cx depth r
        (Printf.sprintf "%s(%s, %s, %s)"
           (if is_f32 res then "fmaf" else "fma")
           (n a) (n b) (n c))
  | Ssa_op.Float_max (a, b) ->
      let res, r = first_result cx i in
      assign cx depth r
        (call cx
           (if is_f32 res then R.Name.Float_max_f32 else R.Name.Float_max)
           [ n a; n b ])
  | Ssa_op.Float_to_i64 a ->
      let _, r = first_result cx i in
      line cx depth
        "if (!(%s >= -9223372036854775808.0 && %s < 9223372036854775808.0)) \
         return %s;"
        (n a) (n a)
        (call cx R.Name.I64_from_float_failure [ "err"; n a ]);
      assign cx depth r (call cx R.Name.I64_from_float [ n a ])
  | Ssa_op.Float_unary (o, a) ->
      let res, r = first_result cx i in
      assign cx depth r (unary cx ~f32:(is_f32 res) o (n a))
  | Ssa_op.I64_arith (o, a, b) ->
      let _, r = first_result cx i in
      let s =
        match o with
        | Ssa_op.I64_op.Add -> "+"
        | Ssa_op.I64_op.Mul -> "*"
        | Ssa_op.I64_op.Sub -> "-"
      in
      assign cx depth r
        (Printf.sprintf "((int64_t)((uint64_t)%s %s (uint64_t)%s))" (n a) s
           (n b))
  | Ssa_op.I64_compare (c, a, b) | Ssa_op.Index_compare (c, a, b) ->
      let _, r = first_result cx i in
      assign cx depth r
        (Printf.sprintf "(%s %s %s)" (n a)
           (match c with Ssa_op.Compare.Eq -> "==" | Ssa_op.Compare.Lt -> "<")
           (n b))
  | Ssa_op.I64_div (a, b) ->
      let _, r = first_result cx i in
      line cx depth "if (%s == 0) %s" (n b)
        (fail F.Kind.I64_division_by_zero []);
      line cx depth "if (%s == %s && %s == -1) %s" (n a) (i64_lit Int64.min_int)
        (n b)
        (fail F.Kind.I64_division_overflow []);
      assign cx depth r (call cx R.Name.I64_div [ n a; n b ])
  | Ssa_op.Index_add (a, b) ->
      let _, r = first_result cx i in
      assign cx depth r (Printf.sprintf "%s + %s" (n a) (n b));
      line cx depth "if %s %s" (outside_int32 r)
        (fail F.Kind.Index_overflow [ "0"; n a; n b ])
  | Ssa_op.Index_add_in_domain (a, b) ->
      let _, r = first_result cx i in
      assign cx depth r (Printf.sprintf "%s + %s" (n a) (n b))
  | Ssa_op.Index_ceil_div (k, a) ->
      let _, r = first_result cx i in
      assign cx depth r ("-" ^ call cx R.Name.Floor_div [ "-" ^ n a; i64_lit k ])
  | Ssa_op.Index_clamp_low a ->
      let _, r = first_result cx i in
      assign cx depth r (call cx R.Name.Idx_clamp_low [ n a ])
  | Ssa_op.Index_floor_div (k, a) ->
      let _, r = first_result cx i in
      assign cx depth r (call cx R.Name.Floor_div [ n a; i64_lit k ])
  | Ssa_op.Index_max (a, b) ->
      let _, r = first_result cx i in
      assign cx depth r (call cx R.Name.Idx_max [ n a; n b ])
  | Ssa_op.Index_min (a, b) ->
      let _, r = first_result cx i in
      assign cx depth r (call cx R.Name.Idx_min [ n a; n b ])
  | Ssa_op.Index_of_i64 a ->
      let _, r = first_result cx i in
      assign cx depth r (n a)
  | Ssa_op.Index_scale (k, a) ->
      let _, r = first_result cx i in
      assign cx depth r (Printf.sprintf "%s * %s" (i64_lit k) (n a));
      line cx depth "if %s %s" (outside_int32 r)
        (fail F.Kind.Index_overflow [ "1"; i64_lit k; n a ])
  | Ssa_op.Index_scale_in_domain (k, a) ->
      let _, r = first_result cx i in
      assign cx depth r (Printf.sprintf "%s * %s" (i64_lit k) (n a))
  | Ssa_op.Lanewise inner ->
      let res, r = first_result cx i in
      assign cx depth r (lanewise cx inner ~result:res)
  | Ssa_op.Load { buffer; at; decode }
  | Ssa_op.Load_in_bounds { buffer; at; decode } ->
      let b = find_buffer cx buffer in
      (match (op, at) with
      | Ssa_op.Load _, Ssa_access.Coord c -> check_coord cx depth b c
      | _ -> ());
      let _, r = first_result cx i in
      let cell =
        Printf.sprintf "%s[%s]" (buffer_name cx buffer) (access_offset cx b at)
      in
      assign cx depth r (decode_cell cx decode cell)
  | Ssa_op.Local_alloc { slots; var } ->
      let res, r = first_result cx i in
      Hashtbl.replace cx.locals (res.Ssa_value.id :> int) (slots, var);
      let off = cx.local_doubles in
      cx.local_doubles <- Int64.add off slots;
      line cx depth "%s = local + %Ld;" r off;
      line cx depth "memset(%s, 0, %Ld * sizeof(double));" r slots
  | Ssa_op.Local_read { local; at } ->
      let _, r = first_result cx i in
      (* outside the object fails as unbound when it names a variable; with no
         variable it is a defect of the program, never a row *)
      (match Hashtbl.find_opt cx.locals (local.Ssa_value.id :> int) with
      | Some (slots, Some var) ->
          let s =
            site cx
              (LF.Local_out_of_range
                 {
                   local = var;
                   index = Loop_ir.Loop_index.Const 0;
                   extent = Int64.to_int slots;
                 })
          in
          line cx depth "if (%s < 0 || %s >= %s) %s" (n at) (n at)
            (i64_lit slots)
            (fail F.Kind.Unbound_local [ string_of_int s ])
      | Some (_, None) | None -> ());
      assign cx depth r (Printf.sprintf "%s[%s]" (n local) (n at))
  | Ssa_op.Local_write { local; at; value } ->
      line cx depth "%s[%s] = %s;" (n local) (n at) (n value)
  | Ssa_op.Meter_charge ->
      invalid_arg "Ssa_c.instr: meters are emitted by the kernel"
  | Ssa_op.Meter_release _ | Ssa_op.Meter_reserve _ | Ssa_op.Meter_reset ->
      invalid_arg "Ssa_c.instr: meters are emitted by the kernel"
  | Ssa_op.Pool_better (best, value) ->
      let _, r = first_result cx i in
      assign cx depth r
        (Printf.sprintf "(%s > %s || %s != %s)" (n value) (n best) (n value)
           (n value))
  | Ssa_op.Pred_not a ->
      let _, r = first_result cx i in
      assign cx depth r (Printf.sprintf "(!%s)" (n a))
  | Ssa_op.Pred_or (a, b) ->
      let _, r = first_result cx i in
      assign cx depth r (Printf.sprintf "(%s || %s)" (n a) (n b))
  | Ssa_op.Select (p, a, b) ->
      let _, r = first_result cx i in
      assign cx depth r (Printf.sprintf "(%s ? %s : %s)" (n p) (n a) (n b))
  | Ssa_op.Store { buffer; at; encode; value } ->
      let b = find_buffer cx buffer in
      line cx depth "%s[%s] = %s;" (buffer_name cx buffer)
        (access_offset cx b at)
        (encode_value encode (n value))
  | Ssa_op.Vec_extract { lane; vector } ->
      let res, r = first_result cx i in
      let k = Ssa_type.Lane.to_int lane in
      if Ssa_type.equal res.Ssa_value.ty (Ssa_type.Scalar Ssa_type.Pred) then
        assign cx depth r (Printf.sprintf "(%s[%d] != 0)" (n vector) k)
      else assign cx depth r (Printf.sprintf "%s[%d]" (n vector) k)
  | Ssa_op.Vec_insert { lane; vector; element } -> (
      let res, r = first_result cx i in
      let k = Ssa_type.Lane.to_int lane in
      assign cx depth r (n vector);
      match res.Ssa_value.ty with
      | Ssa_type.Mask _ -> line cx depth "%s[%d] = %s ? -1 : 0;" r k (n element)
      | _ -> line cx depth "%s[%d] = %s;" r k (n element))
  | Ssa_op.Vec_iota { base; step; lanes } ->
      let res, r = first_result cx i in
      let ty = if elem_f32 res.Ssa_value.ty then "float" else "double" in
      for k = 0 to Ssa_type.Lanes.to_int lanes - 1 do
        line cx depth "%s[%d] = (%s)(%s + (int64_t)%d * %s);" r k ty (n base) k
          (i64_lit step)
      done
  | Ssa_op.Vec_splat { element; lanes = _ } ->
      let res, r = first_result cx i in
      assign cx depth r
        (match res.Ssa_value.ty with
        | Ssa_type.Mask _ ->
            Printf.sprintf "%s(%s)"
              (mask_splat_helper cx res.Ssa_value.ty)
              (n element)
        | _ ->
            Printf.sprintf "%s(%s)"
              (splat_helper cx res.Ssa_value.ty)
              (n element))
  | Ssa_op.Vec_load { buffer; at; steps; decode; lanes } ->
      let b = find_buffer cx buffer in
      let res, r = first_result cx i in
      let count = Ssa_type.Lanes.to_int lanes in
      let delta = lane_delta b steps in
      let o = temp cx in
      line cx depth "{";
      line cx (depth + 1) "const int64_t %s = %s;" o (coord_offset cx b at);
      let cell k =
        Printf.sprintf "%s[%s + %Ld]" (buffer_name cx buffer) o
          (Int64.mul (Int64.of_int k) delta)
      in
      (match (decode, Int64.equal delta 1L) with
      | Ssa_op.Decode.F32_to_f64, true ->
          let st = vector_type cx ~elem:`F32 lanes in
          line cx (depth + 1) "%s s; memcpy(&s, &%s[%s], sizeof s);" st
            (buffer_name cx buffer) o;
          line cx (depth + 1) "%s = __builtin_convertvector(s, %s);" r
            (vtype cx res.Ssa_value.ty)
      | Ssa_op.Decode.F64_to_f64, true ->
          line cx (depth + 1) "memcpy(&%s, &%s[%s], sizeof %s);" r
            (buffer_name cx buffer) o r
      | _ ->
          for k = 0 to count - 1 do
            line cx (depth + 1) "%s[%d] = %s;" r k
              (decode_cell cx decode (cell k))
          done);
      line cx depth "}"
  | Ssa_op.Vec_store { buffer; at; steps; encode; value; lanes } ->
      let b = find_buffer cx buffer in
      let count = Ssa_type.Lanes.to_int lanes in
      let delta = lane_delta b steps in
      let o = temp cx in
      line cx depth "{";
      line cx (depth + 1) "const int64_t %s = %s;" o (coord_offset cx b at);
      (match (encode, Int64.equal delta 1L) with
      | Ssa_op.Encode.F32_round, true ->
          let st = vector_type cx ~elem:`F32 lanes in
          line cx (depth + 1) "%s s = __builtin_convertvector(%s, %s);" st
            (n value) st;
          line cx (depth + 1) "memcpy(&%s[%s], &s, sizeof s);"
            (buffer_name cx buffer) o
      | _ ->
          for k = 0 to count - 1 do
            line cx (depth + 1) "%s[%s + %Ld] = %s;" (buffer_name cx buffer) o
              (Int64.mul (Int64.of_int k) delta)
              (encode_value encode (Printf.sprintf "%s[%d]" (n value) k))
          done);
      line cx depth "}"
