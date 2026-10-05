open Ssa_ir
module A = Js_ast
module R = Loop_ir.Loop_js_runtime
module F = Loop_ir.Loop_js_failure
module LF = Loop_ir.Loop_failure

type error =
  [ `Unsupported_format of Ssa_id.Buffer.t * string
  | `Unsupported_operation of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Unsupported_format (b, f) ->
      Fmt.pf ppf "%a: format %s has no JavaScript implementation"
        Ssa_id.Buffer.pp b f
  | `Unsupported_operation o -> Fmt.pf ppf "%s has no JavaScript form" o

exception Refused of error

let function_name = Loop_ir.Loop_js.function_name
let id = Js_ident.v

(* ---- expressions: the AST is untyped, the SSA types keep it honest ------------- *)

let num x = A.Number x
let lit i = A.Number (float_of_int i)
let idx_lit (n : int64) = A.Number (Int64.to_float n)
let big_lit n = A.Bigint n
let bin op a b = A.Binary (op, a, b)
let member o name = A.Member (o, id name)
let math name args = A.Call (member (A.Global A.Global.Math) name, args)
let big_of e = A.Call (A.Global A.Global.Big_int, [ e ])
let number_of e = A.Call (A.Global A.Global.Number, [ e ])

let as_int64 e =
  A.Call (member (A.Global A.Global.Big_int) "asIntN", [ num 64.; e ])

let runtime n args = A.Call (A.Var (id (R.Name.to_string n)), args)
let record = F.record
let string s = A.String s
let bool b = A.Bool b
let return_ e = A.Stmt.Return (Some e)

let outside_int32 v =
  bin A.Or (bin A.Lt v (num (-2147483648.))) (bin A.Ge v (num 2147483648.))

(* An index that becomes a float must be [+0]: ceil and floor can make [-0]. *)
let float_of_index i = bin A.Add i (num 0.)

(* ---- state --------------------------------------------------------------------- *)

type t = {
  names : (int, Js_ident.t) Hashtbl.t;
  mutable decls : (Js_ident.t * A.expr) list;
  mutable next_name : int;
  mutable next_temp : int;
  mutable sites : LF.t list;
  mutable site_count : int;
  mutable tables : A.Stmt.t list;
  locals : (int, int64 * Expr.Local_var.t option) Hashtbl.t;
  buffers : Ssa_buffer.t list;
}

let buffer_index cx bid =
  let rec go i = function
    | [] -> invalid_arg "Ssa_js: an undeclared buffer"
    | (b : Ssa_buffer.t) :: rest ->
        if Ssa_id.Buffer.equal b.Ssa_buffer.id bid then i else go (i + 1) rest
  in
  go 0 cx.buffers

let buffer_var cx bid = A.Var (id ("b" ^ string_of_int (buffer_index cx bid)))

let find_buffer cx bid =
  match
    List.find_opt
      (fun (b : Ssa_buffer.t) -> Ssa_id.Buffer.equal b.Ssa_buffer.id bid)
      cx.buffers
  with
  | Some b -> b
  | None -> invalid_arg "Ssa_js: an undeclared buffer"

let typed_array (b : Ssa_buffer.t) =
  match b.Ssa_buffer.format with
  | Ssa_format.Bf16 | Ssa_format.F16 -> "Uint16Array"
  | Ssa_format.Bool -> "Uint8Array"
  | Ssa_format.F32 -> "Float32Array"
  | Ssa_format.F64 -> "Float64Array"
  | Ssa_format.I16 _ -> "Int16Array"
  | Ssa_format.I32 -> "Int32Array"
  | Ssa_format.I64 -> "BigInt64Array"
  | Ssa_format.I8 _ -> "Int8Array"

(* The initial value of a variable of this type. *)
let init_of (ty : Ssa_type.t) =
  match ty with
  | Ssa_type.Effect -> None
  | Ssa_type.Local -> Some A.Null
  | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64 | Ssa_type.Index) ->
      Some (num 0.)
  | Ssa_type.Scalar Ssa_type.I64 -> Some (big_lit 0L)
  | Ssa_type.Scalar Ssa_type.Pred -> Some (bool false)
  | Ssa_type.Scalar Ssa_type.Offset ->
      invalid_arg "Ssa_js: a native byte offset has no kernel form"
  | Ssa_type.Mask _ | Ssa_type.Vec _ ->
      raise (Refused (`Unsupported_operation "a vector or mask"))

let is_erased (v : Ssa_value.t) = Ssa_type.equal v.Ssa_value.ty Ssa_type.Effect

let is_f32 (v : Ssa_value.t) =
  Ssa_type.equal v.Ssa_value.ty (Ssa_type.Scalar Ssa_type.F32)

let define cx (v : Ssa_value.t) =
  match init_of v.Ssa_value.ty with
  | None -> None
  | Some init ->
      let n = id (Printf.sprintf "v%d" cx.next_name) in
      cx.next_name <- cx.next_name + 1;
      Hashtbl.replace cx.names (v.Ssa_value.id :> int) n;
      cx.decls <- (n, init) :: cx.decls;
      Some n

let ident cx (v : Ssa_value.t) =
  match Hashtbl.find_opt cx.names (v.Ssa_value.id :> int) with
  | Some n -> n
  | None -> invalid_arg "Ssa_js: a value used before it is defined"

let var cx v = A.Var (ident cx v)

let temp cx =
  let n = id (Printf.sprintf "t%d" cx.next_temp) in
  cx.next_temp <- cx.next_temp + 1;
  n

let site cx f =
  let k = cx.site_count in
  cx.sites <- f :: cx.sites;
  cx.site_count <- k + 1;
  k

let assign cx (v : Ssa_value.t) e =
  match define cx v with
  | Some n -> [ A.Stmt.Assign (A.Lvar n, A.Eq, e) ]
  | None -> []

let if_ p yes = A.Stmt.If (p, yes, [])

(* ---- addresses and cells ------------------------------------------------------- *)

let extents (b : Ssa_buffer.t) = Expr.Coord.to_list b.Ssa_buffer.extents

let coord_offset cx (b : Ssa_buffer.t) (c : Ssa_value.t Expr.Coord.t) =
  match (Expr.Coord.to_list c, extents b) with
  | first :: rest, _ :: exts ->
      List.fold_left2
        (fun acc comp e -> bin A.Add (bin A.Mul acc (idx_lit e)) (var cx comp))
        (var cx first) rest exts
  | _ -> invalid_arg "Ssa_js: a coordinate has six components"

let access_offset cx b = function
  | Ssa_access.Coord c -> coord_offset cx b c
  | Ssa_access.Flat o -> var cx o

let quant_scales cx bid =
  id ("b" ^ string_of_int (buffer_index cx bid) ^ "_scale")

let quant_zeros cx bid = id ("b" ^ string_of_int (buffer_index cx bid) ^ "_zero")

let decode cx (b : Ssa_buffer.t) (d : Ssa_op.Decode.t) ~(at : Ssa_access.t) cell
    =
  match d with
  | Ssa_op.Decode.Bf16_to_f64 -> runtime R.Name.Bf16_to_float [ cell ]
  | Ssa_op.Decode.Bool_to_f64 ->
      A.Cond (bin A.Ne_strict cell (num 0.), num 1., num 0.)
  | Ssa_op.Decode.F16_to_f64 -> runtime R.Name.F16_to_float [ cell ]
  | Ssa_op.Decode.F32_to_f64 | Ssa_op.Decode.F64_to_f64
  | Ssa_op.Decode.I32_to_f64 ->
      cell
  | Ssa_op.Decode.I64 -> cell
  | Ssa_op.Decode.I64_to_f64 -> number_of cell
  | Ssa_op.Decode.I16_dequant | Ssa_op.Decode.I8_dequant -> (
      match Ssa_format.quant b.Ssa_buffer.format with
      | None ->
          invalid_arg "Ssa_js: a dequantizing load of an unquantized buffer"
      | Some (Ssa_format.Per_tensor { scale; zero_point }) ->
          bin A.Mul (num scale) (bin A.Sub cell (lit zero_point))
      | Some (Ssa_format.Per_channel { scale; zero_point }) ->
          let bid = b.Ssa_buffer.id in
          if
            not
              (List.exists
                 (function
                   | A.Stmt.Const (n, _) ->
                       Js_ident.equal n (quant_scales cx bid)
                   | _ -> false)
                 cx.tables)
          then
            cx.tables <-
              cx.tables
              @ [
                  A.Stmt.Const
                    ( quant_scales cx bid,
                      A.Array (Array.to_list (Array.map num scale)) );
                  A.Stmt.Const
                    ( quant_zeros cx bid,
                      A.Array
                        (Array.to_list
                           (Array.map
                              (fun z -> num (float_of_int z))
                              zero_point)) );
                ];
          let channel =
            match at with
            | Ssa_access.Coord c -> var cx c.Expr.Coord.c
            | Ssa_access.Flat _ -> invalid_arg "Ssa_js: a flat per-channel load"
          in
          bin A.Mul
            (A.Index (A.Var (quant_scales cx bid), channel))
            (bin A.Sub cell (A.Index (A.Var (quant_zeros cx bid), channel))))

(* ---- failures ------------------------------------------------------------------ *)

let check_coord cx (b : Ssa_buffer.t) (c : Ssa_value.t Expr.Coord.t) =
  let comps = List.map (var cx) (Expr.Coord.to_list c) in
  let exts = extents b in
  let outside =
    List.map2
      (fun comp e ->
        bin A.Or (bin A.Lt comp (num 0.)) (bin A.Ge comp (idx_lit e)))
      comps exts
  in
  let cond =
    match outside with
    | [] -> bool false
    | first :: rest -> List.fold_left (fun acc o -> bin A.Or acc o) first rest
  in
  [
    if_ cond
      [
        return_
          (runtime R.Name.Coord_failure
             [
               lit (b.Ssa_buffer.id :> int);
               A.Array (List.map idx_lit exts);
               A.Array comps;
             ]);
      ];
  ]

let meter_failure which limit =
  return_
    (record F.Kind.Scan_meter
       [
         (F.Field.Which, string (F.Meter.to_string which));
         (F.Field.Limit, limit);
       ])

let exact_number n =
  let x = Int64.to_float n in
  if Float.abs x <= 9007199254740992. && Int64.equal (Int64.of_float x) n then
    num x
  else invalid_arg "Ssa_js: a scan limit is not exact in a Number"

let scan_remaining = id "scan_remaining"
let scan_live = id "scan_live"
