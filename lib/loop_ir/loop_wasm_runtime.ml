module I = Wasm.Instr

module Callee = struct
  type t =
    | Ceil_div
    | Cos
    | Erf
    | Exp
    | F16_to_float
    | Fail_set
    | Floor_div
    | I64_div
    | Log
    | Sin

  let all =
    [
      Ceil_div;
      Cos;
      Erf;
      Exp;
      F16_to_float;
      Fail_set;
      Floor_div;
      I64_div;
      Log;
      Sin;
    ]

  let index c =
    let rec go i = function
      | [] -> invalid_arg "Loop_wasm_runtime.Callee.index"
      | c' :: rest -> if c = c' then i else go (i + 1) rest
    in
    go 0 all

  let of_index i = List.nth all i

  (* The host's [Math] function a callee is bound to, for the four that are
     imports; every other callee is a function defined in the module. *)
  let import = function
    | Cos -> Some "cos"
    | Exp -> Some "exp"
    | Log -> Some "log"
    | Sin -> Some "sin"
    | Ceil_div | Erf | F16_to_float | Fail_set | Floor_div | I64_div -> None

  let deps = function
    | Erf -> [ Exp ]
    | Ceil_div | Cos | Exp | F16_to_float | Fail_set | Floor_div | I64_div | Log
    | Sin ->
        []
end

let import_module = "math"
let call c = I.Call (Callee.index c)
let f64 x = I.F64_const (Int64.bits_of_float x)
let i32 n = I.I32_const (Int32.of_int n)
let n op = I.Numeric op
let get i = I.Local_get i
let set i = I.Local_set i
let ( ++ ) a b = a @ b
let ft params results = { Wasm.Func_type.params; results }
let wt_f64 = Wasm_type.F64
let wt_i32 = Wasm_type.I32
let wt_i64 = Wasm_type.I64
let arg3 offset = { Wasm.Mem_arg.align = 3; offset }
let arg2 offset = { Wasm.Mem_arg.align = 2; offset }

(* [fail_set kind]: the record's [kind], [invocation = -1] and every [v] slot
   zero, as the C helper of the same name does. *)
let fail_set =
  {
    Wasm.Func.type_ = ft [ wt_i32 ] [];
    locals = [];
    body =
      [
        i32 Loop_wasm_failure.kind_offset;
        get 0;
        I.Store (Wasm.Store.I32_store, arg2 Loop_wasm_failure.kind_offset);
        i32 0;
        i32 (-1);
        I.Store (Wasm.Store.I32_store, arg2 Loop_wasm_failure.invocation_offset);
      ]
      @ List.concat_map
          (fun k ->
            [
              i32 0;
              I.I64_const 0L;
              I.Store
                (Wasm.Store.I64_store, arg3 (Loop_wasm_failure.slot_offset k));
            ])
          (List.init Loop_wasm_failure.error_words Fun.id);
  }

(* Division by a positive constant, rounding toward negative infinity
   ([floor_div]) or positive infinity ([ceil_div]). Both are exact over the
   whole [int32] range, which negating a numerator is not. *)
let rounded_div ~floor =
  let q = 2 in
  {
    Wasm.Func.type_ = ft [ wt_i32; wt_i32 ] [ wt_i32 ];
    locals = [ wt_i32 ];
    body =
      [ get 0; get 1; n Wasm_op.I32_div_s; set q ]
      @ [ get q; i32 (if floor then -1 else 1); n Wasm_op.I32_add; get q ]
      @ [ get 0; get 1; n Wasm_op.I32_rem_s; i32 0 ]
      @ [ n (if floor then Wasm_op.I32_lt_s else Wasm_op.I32_gt_s); I.Select ];
  }

(* Total, like the C helper: a [Fail_if] precedes every division, so neither
   special case is reachable, and this one can never trap. *)
let i64_div =
  {
    Wasm.Func.type_ = ft [ wt_i64; wt_i64 ] [ wt_i64 ];
    locals = [];
    body =
      [
        get 1;
        n Wasm_op.I64_eqz;
        I.If
          ( Some wt_i64,
            [ I.I64_const 0L ],
            [
              get 1;
              I.I64_const (-1L);
              n Wasm_op.I64_eq;
              I.If
                ( Some wt_i64,
                  [ I.I64_const 0L; get 0; n Wasm_op.I64_sub ],
                  [ get 0; get 1; n Wasm_op.I64_div_s ] );
            ] );
      ];
  }

(* [Half.Half.to_float], branch for branch. The scale [2^(exp - 25)] is built
   from its exponent bits, so it is exact. *)
let f16_to_float =
  let sign = 1 and exp = 2 and mant = 3 and m = 4 in
  let field shift mask =
    [ get 0; i32 shift; n Wasm_op.I32_shr_u; i32 mask; n Wasm_op.I32_and ]
  in
  {
    Wasm.Func.type_ = ft [ wt_i32 ] [ wt_f64 ];
    locals = [ wt_i32; wt_i32; wt_i32; wt_f64 ];
    body =
      field 15 1
      @ [ set sign ]
      @ field 10 0x1f
      @ [ set exp ]
      @ field 0 0x3ff
      @ [ set mant ]
      @ [
          get exp;
          n Wasm_op.I32_eqz;
          I.If
            ( None,
              [
                get mant;
                n Wasm_op.F64_convert_i32_u;
                f64 (Float.ldexp 1. (-24));
                n Wasm_op.F64_mul;
                set m;
              ],
              [
                get exp;
                i32 0x1f;
                n Wasm_op.I32_eq;
                I.If
                  ( None,
                    [
                      get mant;
                      n Wasm_op.I32_eqz;
                      I.If
                        (Some wt_f64, [ f64 Float.infinity ], [ f64 Float.nan ]);
                      set m;
                    ],
                    [
                      get mant;
                      i32 0x400;
                      n Wasm_op.I32_or;
                      n Wasm_op.F64_convert_i32_u;
                      get exp;
                      i32 998;
                      n Wasm_op.I32_add;
                      n Wasm_op.I64_extend_i32_u;
                      I.I64_const 52L;
                      n Wasm_op.I64_shl;
                      n Wasm_op.F64_reinterpret_i64;
                      n Wasm_op.F64_mul;
                      set m;
                    ] );
              ] );
          get sign;
          I.If (Some wt_f64, [ get m; n Wasm_op.F64_neg ], [ get m ]);
        ];
  }

(* [Value.erf_approx], operation for operation: the Abramowitz-Stegun
   polynomial, whose only transcendental is its inner [exp]. *)
let erf =
  let ax = 1 and t = 2 and poly = 3 in
  let ( * ) a b = a ++ b ++ [ n Wasm_op.F64_mul ] in
  let ( + ) a b = a ++ b ++ [ n Wasm_op.F64_add ] in
  let c x = [ f64 x ] in
  let tv = [ get t ] in
  let horner =
    tv
    * (c 0.254829592
      + tv
        * (c (-0.284496736)
          + tv
            * (c 1.421413741 + (tv * (c (-1.453152027) + (tv * c 1.061405429))))
          ))
  in
  {
    Wasm.Func.type_ = ft [ wt_f64 ] [ wt_f64 ];
    locals = [ wt_f64; wt_f64; wt_f64 ];
    body =
      [ get 0; n Wasm_op.F64_abs; set ax ]
      @ c 1.
      @ (c 1. + (c 0.3275911 * [ get ax ]))
      @ [ n Wasm_op.F64_div; set t ]
      @ horner
      @ [ set poly ]
      @ [ get 0; f64 0.; n Wasm_op.F64_lt ]
      @ [ I.If (Some wt_f64, [ f64 (-1.) ], [ f64 1. ]) ]
      @ c 1.
      @ [ get poly ]
      @ [
          get ax; n Wasm_op.F64_neg; get ax; n Wasm_op.F64_mul; call Callee.Exp;
        ]
      @ [ n Wasm_op.F64_mul; n Wasm_op.F64_sub; n Wasm_op.F64_mul ];
  }

let body : Callee.t -> Wasm.Func.t option = function
  | Callee.Ceil_div -> Some (rounded_div ~floor:false)
  | Callee.Erf -> Some erf
  | Callee.F16_to_float -> Some f16_to_float
  | Callee.Fail_set -> Some fail_set
  | Callee.Floor_div -> Some (rounded_div ~floor:true)
  | Callee.I64_div -> Some i64_div
  | Callee.Cos | Callee.Exp | Callee.Log | Callee.Sin -> None

let signature : Callee.t -> Wasm.Func_type.t = function
  | Callee.Ceil_div | Callee.Floor_div -> ft [ wt_i32; wt_i32 ] [ wt_i32 ]
  | Callee.Cos | Callee.Erf | Callee.Exp | Callee.Log | Callee.Sin ->
      ft [ wt_f64 ] [ wt_f64 ]
  | Callee.F16_to_float -> ft [ wt_i32 ] [ wt_f64 ]
  | Callee.Fail_set -> ft [ wt_i32 ] []
  | Callee.I64_div -> ft [ wt_i64; wt_i64 ] [ wt_i64 ]
