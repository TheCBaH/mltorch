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

(* ---- one instruction ----------------------------------------------------------- *)

let refuse what = raise (Refused (`Unsupported_operation what))

let unary cx ~f32 (op : Expr.Value.unary_op) a =
  ignore cx;
  let r =
    match op with
    | Expr.Value.Cos -> math "cos" [ a ]
    | Expr.Value.Erf ->
        if f32 then refuse "a binary32 erf" else runtime R.Name.Erf [ a ]
    | Expr.Value.Exp -> math "exp" [ a ]
    | Expr.Value.Log -> math "log" [ a ]
    | Expr.Value.Sin -> math "sin" [ a ]
    | Expr.Value.Sqrt -> math "sqrt" [ a ]
    | Expr.Value.Trunc -> math "trunc" [ a ]
  in
  if f32 then math "fround" [ r ] else r

let binary_op : Expr.Value.binary_op -> A.binop = function
  | Expr.Value.Add -> A.Add
  | Expr.Value.Div -> A.Div
  | Expr.Value.Mul -> A.Mul
  | Expr.Value.Sub -> A.Sub

let cmp : Ssa_op.Compare.t -> A.binop = function
  | Ssa_op.Compare.Eq -> A.Eq_strict
  | Ssa_op.Compare.Lt -> A.Lt

let first_result (i : Ssa_instr.t) =
  match i.Ssa_instr.results with
  | r :: _ -> r
  | [] -> invalid_arg "Ssa_js: an instruction without a result"

let instr cx ~(limits : Expr.Scan_limits.t) (i : Ssa_instr.t) : A.Stmt.t list =
  let v = var cx in
  let op = i.Ssa_instr.op in
  let set e = assign cx (first_result i) e in
  match op with
  | Ssa_op.Mark _ | Ssa_op.Mark_lanes _ -> []
  | Ssa_op.Check_access { buffer; at } -> (
      let b = find_buffer cx buffer in
      match at with
      | Ssa_access.Coord c -> check_coord cx b c
      | Ssa_access.Flat _ -> [])
  | Ssa_op.Check_gather { raw; extent } ->
      [
        if_
          (bin A.Or
             (bin A.Lt (v raw) (num (-.Int64.to_float extent)))
             (bin A.Ge (v raw) (idx_lit extent)))
          [
            return_
              (record F.Kind.Gather_index_out_of_range
                 [
                   (F.Field.Raw, A.Call (member (v raw) "toString", []));
                   (F.Field.Extent, idx_lit extent);
                 ]);
          ];
      ]
  | Ssa_op.Check_local { var = lv; at; extent } ->
      let s =
        site cx
          (LF.Local_out_of_range
             {
               local = lv;
               index = Loop_ir.Loop_index.Const 0;
               extent = Int64.to_int extent;
             })
      in
      [
        if_
          (bin A.Or
             (bin A.Lt (v at) (num 0.))
             (bin A.Ge (v at) (idx_lit extent)))
          [ return_ (record F.Kind.Unbound_local [ (F.Field.Site, lit s) ]) ];
      ]
  | Ssa_op.Check_scan { var = lv; row; lane; row_extent; lane_extent } ->
      let row_site =
        site cx
          (LF.Scan_row_out_of_range
             {
               local = lv;
               row = Loop_ir.Loop_index.Const 0;
               lane = Loop_ir.Loop_index.Const 0;
               extent = Int64.to_int row_extent;
             })
      in
      let lane_site =
        site cx
          (LF.Scan_lane_out_of_range
             {
               local = lv;
               row = Loop_ir.Loop_index.Const 0;
               lane = Loop_ir.Loop_index.Const 0;
               extent = Int64.to_int lane_extent;
             })
      in
      let rec_ which s extent =
        record F.Kind.Scan_projection
          [
            (F.Field.Which, string (F.Projection.to_string which));
            (F.Field.Cached, bool (Option.is_some lv));
            (F.Field.Row, v row);
            (F.Field.Lane, v lane);
            (F.Field.Extent, idx_lit extent);
            (F.Field.Site, lit s);
          ]
      in
      [
        if_
          (bin A.Or
             (bin A.Lt (v row) (num 0.))
             (bin A.Ge (v row) (idx_lit row_extent)))
          [ return_ (rec_ F.Projection.Row row_site row_extent) ];
        if_
          (bin A.Or
             (bin A.Lt (v lane) (num 0.))
             (bin A.Ge (v lane) (idx_lit lane_extent)))
          [ return_ (rec_ F.Projection.Lane lane_site lane_extent) ];
      ]
  | Ssa_op.Const c ->
      set
        (match c with
        | Ssa_const.F32 x | Ssa_const.F64 x -> num x
        | Ssa_const.I64 n -> big_lit n
        | Ssa_const.Index n -> idx_lit n
        | Ssa_const.Pred b -> bool b)
  | Ssa_op.Convert (c, a) ->
      set
        (match c with
        | Ssa_op.Convert.F32_to_f64 -> v a
        | Ssa_op.Convert.F64_to_f32 -> math "fround" [ v a ]
        | Ssa_op.Convert.I64_to_f64 -> number_of (v a)
        | Ssa_op.Convert.I64_to_f32 -> refuse "an int64 to binary32 conversion"
        | Ssa_op.Convert.Index_to_f64 -> float_of_index (v a)
        | Ssa_op.Convert.Index_to_i64 -> big_of (v a))
  | Ssa_op.Float_binary (o, a, b) ->
      let r = bin (binary_op o) (v a) (v b) in
      set (if is_f32 (first_result i) then math "fround" [ r ] else r)
  | Ssa_op.Float_compare (c, a, b) -> set (bin (cmp c) (v a) (v b))
  | Ssa_op.Float_fma _ -> refuse "a fused multiply-add"
  | Ssa_op.Float_max (a, b) -> set (math "max" [ v a; v b ])
  | Ssa_op.Float_to_i64 a ->
      let x = v a in
      if_
        (A.Unary
           ( A.Not,
             bin A.And
               (bin A.Ge x (num (-9223372036854775808.)))
               (bin A.Lt x (num 9223372036854775808.)) ))
        [ return_ (runtime R.Name.I64_from_float_failure [ x ]) ]
      :: set (big_of (math "trunc" [ x ]))
  | Ssa_op.Float_unary (o, a) ->
      set (unary cx ~f32:(is_f32 (first_result i)) o (v a))
  | Ssa_op.I64_arith (o, a, b) ->
      set
        (as_int64
           (bin
              (match o with
              | Ssa_op.I64_op.Add -> A.Add
              | Ssa_op.I64_op.Mul -> A.Mul
              | Ssa_op.I64_op.Sub -> A.Sub)
              (v a) (v b)))
  | Ssa_op.I64_compare (c, a, b) | Ssa_op.Index_compare (c, a, b) ->
      set (bin (cmp c) (v a) (v b))
  | Ssa_op.I64_div (a, b) ->
      if_
        (bin A.Eq_strict (v b) (big_lit 0L))
        [ return_ (record F.Kind.I64_division_by_zero []) ]
      :: if_
           (bin A.And
              (bin A.Eq_strict (v a) (big_lit Int64.min_int))
              (bin A.Eq_strict (v b) (big_lit (-1L))))
           [ return_ (record F.Kind.I64_division_overflow []) ]
      :: set (bin A.Div (v a) (v b))
  | Ssa_op.Index_add (a, b) ->
      let r = first_result i in
      let defined = set (bin A.Add (v a) (v b)) in
      defined
      @ [
          if_
            (outside_int32 (var cx r))
            [
              return_
                (record F.Kind.Index_overflow
                   [
                     ( F.Field.Op,
                       string (F.Overflow_op.to_string F.Overflow_op.Add) );
                     (F.Field.Lhs, v a);
                     (F.Field.Rhs, v b);
                   ]);
            ];
        ]
  | Ssa_op.Index_add_in_domain (a, b) -> set (bin A.Add (v a) (v b))
  | Ssa_op.Index_ceil_div (k, a) ->
      set (math "ceil" [ bin A.Div (v a) (idx_lit k) ])
  | Ssa_op.Index_clamp_low a -> set (math "max" [ num 0.; v a ])
  | Ssa_op.Index_floor_div (k, a) ->
      set (math "floor" [ bin A.Div (v a) (idx_lit k) ])
  | Ssa_op.Index_max (a, b) -> set (math "max" [ v a; v b ])
  | Ssa_op.Index_min (a, b) -> set (math "min" [ v a; v b ])
  | Ssa_op.Index_of_i64 a -> set (number_of (v a))
  | Ssa_op.Index_scale (k, a) ->
      let r = first_result i in
      let defined = set (bin A.Mul (idx_lit k) (v a)) in
      defined
      @ [
          if_
            (outside_int32 (var cx r))
            [
              return_
                (record F.Kind.Index_overflow
                   [
                     ( F.Field.Op,
                       string (F.Overflow_op.to_string F.Overflow_op.Mul) );
                     (F.Field.Lhs, idx_lit k);
                     (F.Field.Rhs, v a);
                   ]);
            ];
        ]
  | Ssa_op.Index_scale_in_domain (k, a) -> set (bin A.Mul (idx_lit k) (v a))
  | Ssa_op.Lanewise _ -> refuse "a lane-wise operation"
  | Ssa_op.Load { buffer; at; decode = d }
  | Ssa_op.Load_in_bounds { buffer; at; decode = d } ->
      let b = find_buffer cx buffer in
      let check =
        match (op, at) with
        | Ssa_op.Load _, Ssa_access.Coord c -> check_coord cx b c
        | _ -> []
      in
      let cell = A.Index (buffer_var cx buffer, access_offset cx b at) in
      check @ set (decode cx b d ~at cell)
  | Ssa_op.Local_alloc { slots; var = lv } ->
      let r = first_result i in
      let stmts =
        set (A.New (A.Global A.Global.Float64_array, [ idx_lit slots ]))
      in
      Hashtbl.replace cx.locals (r.Ssa_value.id :> int) (slots, lv);
      stmts
  | Ssa_op.Local_read { local; at } ->
      let check =
        match Hashtbl.find_opt cx.locals (local.Ssa_value.id :> int) with
        | Some (slots, Some lv) ->
            let s =
              site cx
                (LF.Local_out_of_range
                   {
                     local = lv;
                     index = Loop_ir.Loop_index.Const 0;
                     extent = Int64.to_int slots;
                   })
            in
            [
              if_
                (bin A.Or
                   (bin A.Lt (v at) (num 0.))
                   (bin A.Ge (v at) (idx_lit slots)))
                [
                  return_
                    (record F.Kind.Unbound_local [ (F.Field.Site, lit s) ]);
                ];
            ]
        | Some (_, None) | None -> []
      in
      check @ set (A.Index (v local, v at))
  | Ssa_op.Local_write { local; at; value } ->
      [ A.Stmt.Assign (A.Lindex (v local, v at), A.Eq, v value) ]
  | Ssa_op.Meter_charge ->
      [
        if_
          (bin A.Le (A.Var scan_remaining) (num 0.))
          [
            meter_failure F.Meter.Updates_exhausted
              (exact_number (Expr.Scan_limits.max_updates limits));
          ];
        A.Stmt.Assign (A.Lvar scan_remaining, A.Minus_eq, num 1.);
      ]
  | Ssa_op.Meter_release width ->
      [
        A.Stmt.Assign
          (A.Lvar scan_live, A.Minus_eq, num (2. *. Int64.to_float width));
      ]
  | Ssa_op.Meter_reserve width ->
      let live = 2. *. Int64.to_float width in
      let max_state = Expr.Scan_limits.max_state limits in
      [
        if_
          (bin A.Gt (bin A.Add (A.Var scan_live) (num live)) (lit max_state))
          [ meter_failure F.Meter.State_over_limit (lit max_state) ];
        A.Stmt.Assign (A.Lvar scan_live, A.Plus_eq, num live);
      ]
  | Ssa_op.Meter_reset ->
      [
        A.Stmt.Assign
          ( A.Lvar scan_remaining,
            A.Eq,
            exact_number (Expr.Scan_limits.max_updates limits) );
        A.Stmt.Assign (A.Lvar scan_live, A.Eq, num 0.);
      ]
  | Ssa_op.Pool_better (best, value) ->
      set (runtime R.Name.Pool_better [ v best; v value ])
  | Ssa_op.Pred_not a -> set (A.Unary (A.Not, v a))
  | Ssa_op.Pred_or (a, b) -> set (bin A.Or (v a) (v b))
  | Ssa_op.Select (p, a, b) -> set (A.Cond (v p, v a, v b))
  | Ssa_op.Store { buffer; at; encode; value } ->
      let b = find_buffer cx buffer in
      let e =
        match encode with
        | Ssa_op.Encode.Bool_nonzero ->
            A.Cond (bin A.Ne_strict (v value) (num 0.), num 1., num 0.)
        | Ssa_op.Encode.F32_round | Ssa_op.Encode.I64 -> v value
      in
      [
        A.Stmt.Assign
          (A.Lindex (buffer_var cx buffer, access_offset cx b at), A.Eq, e);
      ]
  | Ssa_op.Vec_extract _ | Ssa_op.Vec_insert _ | Ssa_op.Vec_iota _
  | Ssa_op.Vec_load _ | Ssa_op.Vec_splat _ | Ssa_op.Vec_store _ ->
      refuse "a vector operation"

(* ---- control flow -------------------------------------------------------------- *)

let rec region cx ~limits (r : Ssa_region.t) =
  List.concat_map (stmt cx ~limits) r.Ssa_region.body

and stmt cx ~limits (s : Ssa_region.t Ssa_stmt.t) : A.Stmt.t list =
  match s with
  | Ssa_stmt.Instr i -> instr cx ~limits i
  | Ssa_stmt.For { lo; hi; step; inits; results; body } ->
      let iv, carried =
        match body.Ssa_region.params with
        | iv :: carried -> (iv, carried)
        | [] -> invalid_arg "Ssa_js: a loop without an induction value"
      in
      ignore (define cx iv);
      List.iter (fun p -> ignore (define cx p)) carried;
      let trips = temp cx and k = temp cx in
      let span = bin A.Sub (var cx hi) (var cx lo) in
      let count =
        if Int64.equal step 1L then math "max" [ num 0.; span ]
        else
          math "max" [ num 0.; math "ceil" [ bin A.Div span (idx_lit step) ] ]
      in
      let init_carried =
        List.concat
          (List.map2
             (fun (p : Ssa_value.t) init ->
               if is_erased p then []
               else [ A.Stmt.Assign (A.Lvar (ident cx p), A.Eq, var cx init) ])
             carried inits)
      in
      let set_iv =
        A.Stmt.Assign
          ( A.Lvar (ident cx iv),
            A.Eq,
            bin A.Add (var cx lo) (bin A.Mul (A.Var k) (idx_lit step)) )
      in
      let body_stmts = region cx ~limits body in
      let transfer =
        let moves =
          List.filter
            (fun ((p : Ssa_value.t), (y : Ssa_value.t)) ->
              (not (is_erased p)) && not (Ssa_value.equal p y))
            (List.combine carried body.Ssa_region.yields)
        in
        let temps = List.map (fun (p, y) -> (p, temp cx, y)) moves in
        List.map (fun (_, t, y) -> A.Stmt.Const (t, var cx y)) temps
        @ List.map
            (fun ((p : Ssa_value.t), t, _) ->
              A.Stmt.Assign (A.Lvar (ident cx p), A.Eq, A.Var t))
            temps
      in
      let results_stmts =
        List.concat
          (List.map2
             (fun (r : Ssa_value.t) (p : Ssa_value.t) ->
               if is_erased r then [] else assign cx r (var cx p))
             results carried)
      in
      (A.Stmt.Const (trips, count) :: init_carried)
      @ [
          A.Stmt.For
            {
              var = k;
              init = num 0.;
              test = bin A.Lt (A.Var k) (A.Var trips);
              body = (set_iv :: body_stmts) @ transfer;
            };
        ]
      @ results_stmts
  | Ssa_stmt.If { cond; results; then_; else_ } ->
      List.iter (fun r -> ignore (define cx r)) results;
      let arm (r : Ssa_region.t) =
        (* the arm's statements define what its yields name: build them first *)
        let stmts = region cx ~limits r in
        stmts
        @ List.concat
            (List.map2
               (fun (res : Ssa_value.t) y ->
                 if is_erased res then []
                 else [ A.Stmt.Assign (A.Lvar (ident cx res), A.Eq, var cx y) ])
               results r.Ssa_region.yields)
      in
      let yes = arm then_ in
      let no = arm else_ in
      [ A.Stmt.If (var cx cond, yes, no) ]
  | Ssa_stmt.Ordered_sum { lo; hi; seed; token = _; results; body } ->
      let iv =
        match body.Ssa_region.params with
        | iv :: _ -> iv
        | [] -> invalid_arg "Ssa_js: a sum without an induction value"
      in
      ignore (define cx iv);
      let trips = temp cx and k = temp cx and acc = temp cx in
      let span = bin A.Sub (var cx hi) (var cx lo) in
      let term = List.hd body.Ssa_region.yields in
      let body_stmts = region cx ~limits body in
      let sum = bin A.Add (A.Var acc) (var cx term) in
      let sum = if is_f32 seed then math "fround" [ sum ] else sum in
      let results_stmts =
        match results with s :: _ -> assign cx s (A.Var acc) | [] -> []
      in
      [
        A.Stmt.Const (trips, math "max" [ num 0.; span ]);
        A.Stmt.Let (acc, var cx seed);
        A.Stmt.For
          {
            var = k;
            init = num 0.;
            test = bin A.Lt (A.Var k) (A.Var trips);
            body =
              A.Stmt.Assign
                (A.Lvar (ident cx iv), A.Eq, bin A.Add (var cx lo) (A.Var k))
              :: body_stmts
              @ [ A.Stmt.Assign (A.Lvar acc, A.Eq, sum) ];
          };
      ]
      @ results_stmts

let rec uses_meter (r : Ssa_region.t) =
  List.exists
    (function
      | Ssa_stmt.Instr i -> (
          match i.Ssa_instr.op with
          | Ssa_op.Meter_charge | Ssa_op.Meter_release _
          | Ssa_op.Meter_reserve _ | Ssa_op.Meter_reset ->
              true
          | _ -> false)
      | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } ->
          uses_meter body
      | Ssa_stmt.If { then_; else_; _ } -> uses_meter then_ || uses_meter else_)
    r.Ssa_region.body

let arguments (p : Ssa_program.t) =
  let seen = ref Ssa_id.Buffer.Set.empty in
  let note bid = seen := Ssa_id.Buffer.Set.add bid !seen in
  let rec region (r : Ssa_region.t) = List.iter stmt r.Ssa_region.body
  and stmt = function
    | Ssa_stmt.Instr i -> (
        match i.Ssa_instr.op with
        | Ssa_op.Check_access { buffer; _ }
        | Ssa_op.Load { buffer; _ }
        | Ssa_op.Load_in_bounds { buffer; _ }
        | Ssa_op.Store { buffer; _ }
        | Ssa_op.Vec_load { buffer; _ }
        | Ssa_op.Vec_store { buffer; _ } ->
            note buffer
        | _ -> ())
    | Ssa_stmt.For { body; _ } | Ssa_stmt.Ordered_sum { body; _ } -> region body
    | Ssa_stmt.If { then_; else_; _ } ->
        region then_;
        region else_
  in
  region p.Ssa_program.entry;
  List.filter
    (fun (b : Ssa_buffer.t) -> Ssa_id.Buffer.Set.mem b.Ssa_buffer.id !seen)
    p.Ssa_program.buffers

(* The helpers the entry function needs: start from its free names, add every
   helper that defines one of them, and repeat with the added helpers' own free
   names. *)
let prelude entry =
  let needed_by (h : R.Helper.t) needed =
    List.exists (fun d -> Js_ident.Set.mem d needed) h.R.Helper.defines
  in
  let rec fix chosen needed =
    let added =
      List.filter
        (fun h -> (not (List.memq h chosen)) && needed_by h needed)
        R.helpers
    in
    if added = [] then chosen
    else
      let needed =
        List.fold_left
          (fun acc (h : R.Helper.t) ->
            Js_ident.Set.union acc (Js_check.free h.R.Helper.body))
          needed added
      in
      fix (chosen @ added) needed
  in
  let chosen = fix [] (Js_check.free [ A.Stmt.Function entry ]) in
  List.concat_map
    (fun h -> if List.memq h chosen then h.R.Helper.body else [])
    R.helpers

let program (p : Ssa_program.t) =
  (match Err.payload (Ssa_verify.check p) with
  | Ok () -> ()
  | Error e ->
      invalid_arg
        (Fmt.str "Ssa_js.program: the program does not verify: %a"
           Ssa_verify.pp_error e));
  let buffers = arguments p in
  let cx =
    {
      names = Hashtbl.create 64;
      decls = [];
      next_name = 0;
      next_temp = 0;
      sites = [];
      site_count = 0;
      tables = [];
      locals = Hashtbl.create 8;
      buffers;
    }
  in
  match
    let limits = p.Ssa_program.scan_limits in
    let entry = p.Ssa_program.entry in
    List.iter (fun v -> ignore (define cx v)) entry.Ssa_region.params;
    (region cx ~limits entry, limits)
  with
  | exception Refused e -> Error e
  | body, limits ->
      let params = List.mapi (fun i _ -> id ("b" ^ string_of_int i)) buffers in
      let decls =
        List.rev_map (fun (n, init) -> A.Stmt.Let (n, init)) cx.decls
      in
      let meter =
        if uses_meter p.Ssa_program.entry then
          [
            A.Stmt.Let
              ( scan_remaining,
                exact_number (Expr.Scan_limits.max_updates limits) );
            A.Stmt.Let (scan_live, num 0.);
          ]
        else []
      in
      let entry_fn =
        {
          A.Func.name = id function_name;
          params;
          body =
            decls @ cx.tables @ meter @ body @ [ A.Stmt.Return (Some A.Null) ];
        }
      in
      let program =
        { A.Program.prelude = prelude entry_fn; entry = entry_fn }
      in
      (match Js_check.closed program with
      | Ok () -> ()
      | Error faults ->
          invalid_arg
            (Fmt.str "Ssa_js.program: the program is not closed: %a"
               Fmt.(list ~sep:(any "; ") Js_check.Fault.pp)
               faults));
      Ok (program, Array.of_list (List.rev cx.sites))
