open Js_build

module Name = struct
  type t =
    | Bf16_to_float
    | Coord_failure
    | Erf
    | F16_to_float
    | Float_max
    | I64_from_float_failure
    | Pool_better

  let all =
    [
      Bf16_to_float;
      Coord_failure;
      Erf;
      F16_to_float;
      Float_max;
      I64_from_float_failure;
      Pool_better;
    ]

  let to_string = function
    | Bf16_to_float -> "bf16_to_float"
    | Coord_failure -> "coord_failure"
    | Erf -> "erf"
    | F16_to_float -> "f16_to_float"
    | Float_max -> "float_max"
    | I64_from_float_failure -> "i64_from_float_failure"
    | Pool_better -> "pool_better"
end

module Failure = Loop_js_failure

module Helper = struct
  type t = {
    name : Name.t;
    defines : Js_ident.t list;
    body : Js_ast.Stmt.t list;
  }
end

let id = Js_ident.v
let name n = id (Name.to_string n)
let num_var s = Num.var (id s)

let fn n params body =
  Js_ast.Stmt.Function
    { Js_ast.Func.name = name n; params = List.map id params; body }

(* [Expr_bridge.bound_in_range]: the FIRST axis, in N T D H W C order, whose
   component lies outside the buffer's shape. The record's kind and fields are
   the ones the interpreter's [`Coord_out_of_range] row carries. *)
let coord_failure_body =
  let coord = Arr.idx (id "coord") and extents = Arr.idx (id "extents") in
  let axis = Idx.var (id "axis") in
  [
    fn Name.Coord_failure
      [ "buffer"; "extents"; "coord" ]
      [
        Stmt.for_ (id "axis") ~lo:(Idx.const 0) ~hi:(Idx.const 6)
          [
            Stmt.if_
              (Pred.or_
                 (Idx.lt (load coord axis) (Idx.const 0))
                 (Idx.ge (load coord axis) (load extents axis)))
              [
                Stmt.return_
                  (Failure.record Failure.Kind.Coord_out_of_range
                     [
                       (Failure.Field.Buffer, expr (Idx.var (id "buffer")));
                       (Failure.Field.Axis, expr axis);
                       (Failure.Field.Index, expr (load coord axis));
                       (Failure.Field.Coord, expr coord);
                     ]);
              ]
              [];
          ];
        Stmt.return_ (Failure.record Failure.Kind.Defect []);
      ];
  ]

(* [Max_op.apply Float_max] is [Float.max]: NaN propagates and [+0.] is above
   [-0.]. [Math.max] has exactly that contract, so the helper is a name for it
   rather than a second definition of the rule. *)
let float_max_body =
  [
    fn Name.Float_max [ "a"; "b" ]
      [ Stmt.return_ (expr (Num.max (num_var "a") (num_var "b"))) ];
  ]

(* [Max_op.pool_better]: the candidate wins on strict greater-than OR on NaN, so
   an ordinary tie keeps the incumbent and every NaN re-triggers (the last wins). *)
let pool_better_body =
  let best = num_var "best" and value = num_var "value" in
  [
    fn Name.Pool_better [ "best"; "value" ]
      [
        Stmt.return_ (expr (Pred.or_ (Num.gt value best) (Num.ne value value)));
      ];
  ]

(* [Value.erf_approx], operation for operation: the Abramowitz-Stegun
   polynomial, not libm's erf. Only its inner [exp] is a transcendental, and so
   the only place it may differ from OCaml. No FMA is formed by JavaScript, while
   [ocamlopt] may contract a multiply-add here: a native-vs-JS difference
   confined to this function is triaged against the bytecode build first. *)
let erf_body =
  let x = num_var "x" and p = num_var "p" and t = num_var "t" in
  let a1 = num_var "a1" and a2 = num_var "a2" and a3 = num_var "a3" in
  let a4 = num_var "a4" and a5 = num_var "a5" in
  let sign = num_var "sign" and ax = num_var "ax" and poly = num_var "poly" in
  let one = Num.const 1. in
  let ( + ) = Num.add and ( * ) = Num.mul in
  [
    fn Name.Erf [ "x" ]
      [
        Stmt.const_num (id "p") (Num.const 0.3275911);
        Stmt.const_num (id "a1") (Num.const 0.254829592);
        Stmt.const_num (id "a2") (Num.const (-0.284496736));
        Stmt.const_num (id "a3") (Num.const 1.421413741);
        Stmt.const_num (id "a4") (Num.const (-1.453152027));
        Stmt.const_num (id "a5") (Num.const 1.061405429);
        Stmt.const_num (id "sign")
          (select (Num.lt x (Num.const 0.)) (Num.const (-1.)) one);
        Stmt.const_num (id "ax") (Num.abs x);
        Stmt.const_num (id "t") (Num.div one (one + (p * ax)));
        Stmt.const_num (id "poly")
          (t * (a1 + (t * (a2 + (t * (a3 + (t * (a4 + (t * a5)))))))));
        Stmt.return_
          (expr (sign * Num.sub one (poly * Num.exp (Num.mul (Num.neg ax) ax))));
      ];
  ]

(* [Half.Bf16.to_float]: bfloat16 is the high half of a binary32 pattern. The
   shared scratch pair reinterprets the 32 bits without allocating; [bits << 16]
   is negative when the sign bit is set, and a [Uint32Array] store wraps it back
   to the pattern. *)
let bf16_body =
  let bits = Arr.uint32 (id "bf16_bits") and view = Arr.num (id "bf16_view") in
  [
    Stmt.const_arr (id "bf16_bits") (Arr.new_uint32 1);
    Stmt.const_arr (id "bf16_view") (Arr.view_float32 bits);
    fn Name.Bf16_to_float [ "bits" ]
      [
        store bits (Idx.const 0) (Bits.shl16 (Bits.var (id "bits")));
        Stmt.return_ (expr (load view (Idx.const 0)));
      ];
  ]

(* [Half.Half.to_float]: IEEE binary16, transcribed branch for branch. [ldexp] is
   an exact power-of-two scaling, so it is a multiplication by an exactly
   representable power of two. *)
let f16_body =
  let bits s = Bits.var (id s) in
  let h = bits "h" and exp = bits "exp" and mant = bits "mant" in
  let m = num_var "m" in
  let pow2 e = Num.pow (Num.const 2.) e in
  [
    fn Name.F16_to_float [ "h" ]
      [
        Stmt.const_bits (id "sign") (Bits.and_ (Bits.shr h 15) (Bits.const 1));
        Stmt.const_bits (id "exp") (Bits.and_ (Bits.shr h 10) (Bits.const 0x1f));
        Stmt.const_bits (id "mant") (Bits.and_ h (Bits.const 0x3ff));
        Stmt.let_num (id "m") (Num.const 0.);
        Stmt.if_
          (Bits.eq exp (Bits.const 0))
          [
            Stmt.assign_num (id "m")
              (Num.mul (Num.of_bits mant) (pow2 (Num.const (-24.))));
          ]
          [
            Stmt.if_
              (Bits.eq exp (Bits.const 0x1f))
              [
                Stmt.assign_num (id "m")
                  (select
                     (Bits.eq mant (Bits.const 0))
                     (Num.const Float.infinity) (Num.const Float.nan));
              ]
              [
                Stmt.assign_num (id "m")
                  (Num.mul
                     (Num.of_bits (Bits.or_ mant (Bits.const 0x400)))
                     (pow2 (Num.sub (Num.of_bits exp) (Num.const 25.))));
              ];
          ];
        Stmt.return_
          (expr (select (Bits.eq (bits "sign") (Bits.const 1)) (Num.neg m) m));
      ];
  ]

(* The failure [Value.i64_of_float] reports for a value outside the int64 range:
   NaN, an infinity, or a finite value beyond [-2^63, 2^63). *)
let i64_from_float_failure_body =
  let x = num_var "x" in
  [
    fn Name.I64_from_float_failure [ "x" ]
      [
        Stmt.if_ (Num.is_nan x)
          [ Stmt.return_ (Failure.record Failure.Kind.I64_from_float_nan []) ]
          [];
        Stmt.if_
          (Pred.not_ (Num.is_finite x))
          [
            Stmt.return_
              (Failure.record Failure.Kind.I64_from_float_infinite []);
          ]
          [];
        Stmt.return_
          (Failure.record Failure.Kind.I64_from_float_out_of_range
             [ (Failure.Field.Value, expr x) ]);
      ];
  ]

let helper n =
  let one body = { Helper.name = n; defines = [ name n ]; body } in
  match n with
  | Name.Bf16_to_float ->
      {
        Helper.name = n;
        defines = [ id "bf16_bits"; id "bf16_view"; name n ];
        body = bf16_body;
      }
  | Name.Coord_failure -> one coord_failure_body
  | Name.Erf -> one erf_body
  | Name.F16_to_float -> one f16_body
  | Name.Float_max -> one float_max_body
  | Name.I64_from_float_failure -> one i64_from_float_failure_body
  | Name.Pool_better -> one pool_better_body

let helpers = List.map helper Name.all

let source =
  String.concat "\n"
    (List.map (fun (h : Helper.t) -> Js_print.stmts h.Helper.body) helpers)

let call n args = Unsafe.call (name n) args
let bf16_to_float b = call Name.Bf16_to_float [ expr b ]

let coord_failure ~buffer ~extents ~coord =
  expr
    (call Name.Coord_failure
       [
         expr buffer;
         Js_ast.Array (List.map (fun e -> expr e) extents);
         Js_ast.Array (List.map (fun c -> expr c) coord);
       ])

let erf x = call Name.Erf [ expr x ]
let f16_to_float b = call Name.F16_to_float [ expr b ]
let float_max a b = call Name.Float_max [ expr a; expr b ]

let i64_from_float_failure x =
  expr (call Name.I64_from_float_failure [ expr x ])

let pool_better best value = call Name.Pool_better [ expr best; expr value ]
