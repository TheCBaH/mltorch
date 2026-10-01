open Wasm

let hex s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

let leb f n =
  let b = Buffer.create 8 in
  f b n;
  hex (Buffer.contents b)

let%expect_test "LEB128 at the boundaries" =
  let u = leb Wasm_encode.uleb128 and s = leb Wasm_encode.sleb128 in
  List.iter
    (fun n -> Fmt.pr "u %Ld = %s@." n (u n))
    [ 0L; 127L; 128L; 624485L; 0x7FFFFFFFL; 0xFFFFFFFFL; -1L ];
  List.iter
    (fun n -> Fmt.pr "s %Ld = %s@." n (s n))
    [
      0L;
      63L;
      64L;
      -1L;
      -64L;
      -65L;
      -2147483648L;
      2147483647L;
      Int64.max_int;
      Int64.min_int;
    ];
  [%expect
    {|
    u 0 = 00
    u 127 = 7f
    u 128 = 8001
    u 624485 = e58e26
    u 2147483647 = ffffffff07
    u 4294967295 = ffffffff0f
    u -1 = ffffffffffffffffff01
    s 0 = 00
    s 63 = 3f
    s 64 = c000
    s -1 = 7f
    s -64 = 40
    s -65 = bf7f
    s -2147483648 = 8080808078
    s 2147483647 = ffffffff07
    s 9223372036854775807 = ffffffffffffffffff00
    s -9223372036854775808 = 8080808080808080807f
    |}]

let fn ?(locals = []) params results body =
  { Func.type_ = { Func_type.params; results }; locals; body }

let encode m =
  match Err.payload (Wasm_encode.module_ m) with
  | Ok s -> hex s
  | Error e -> Fmt.str "error: %a" Wasm_check.pp_error e

let add_module =
  {
    Module.empty with
    funcs =
      [
        fn
          [ Wasm_type.I32; Wasm_type.I32 ]
          [ Wasm_type.I32 ]
          [
            Instr.Local_get 0; Instr.Local_get 1; Instr.Numeric Wasm_op.I32_add;
          ];
      ];
    exports = [ { Export.name = "add"; kind = Export.Func 0 } ];
  }

let%expect_test "the empty module and a one-function module" =
  Fmt.pr "%s@.%s@." (encode Module.empty) (encode add_module);
  [%expect
    {|
    0061736d01000000
    0061736d0100000001070160027f7f017f030201000707010361646400000a09010700200020016a0b
    |}]

let one ?(params = []) ?(results = []) ?(locals = []) ?(memory = true)
    ?(globals = []) body =
  {
    Module.empty with
    funcs = [ fn ~locals params results body ];
    memory =
      (if memory then Some { Memory.min_pages = 1; max_pages = None } else None);
    globals;
  }

let refuse name m = Fmt.pr "%-22s %s@." name (encode m)
let i32 n = Instr.I32_const n
let i64 n = Instr.I64_const n
let f64 x = Instr.F64_const (Int64.bits_of_float x)
let op o = Instr.Numeric o

let%expect_test "every validator refusal" =
  let open Wasm_type in
  refuse "underflow" (one [ op Wasm_op.I32_add ]);
  refuse "operand mismatch"
    (one ~results:[ I32 ] [ i64 1L; i32 1l; op Wasm_op.I32_add ]);
  refuse "result missing" (one ~results:[ I32 ] []);
  refuse "extra value" (one [ i32 1l ]);
  refuse "unknown local" (one [ Instr.Local_get 3; Instr.Drop ]);
  refuse "unknown func" (one [ Instr.Call 4 ]);
  refuse "unknown global" (one [ Instr.Global_get 0; Instr.Drop ]);
  refuse "bad label" (one [ Instr.Br 1 ]);
  refuse "bad label in block" (one [ Instr.Block (None, [ Instr.Br 2 ]) ]);
  refuse "immutable global"
    (one
       ~globals:[ { Global.type_ = I32; mutable_ = false; init = i32 0l } ]
       [ i32 1l; Instr.Global_set 0 ]);
  refuse "no memory"
    (one ~memory:false
       [
         i32 0l;
         Instr.Load (Load.I32_load, { Mem_arg.align = 2; offset = 0 });
         Instr.Drop;
       ]);
  refuse "alignment"
    (one
       [
         i32 0l;
         Instr.Load (Load.I32_load, { Mem_arg.align = 3; offset = 0 });
         Instr.Drop;
       ]);
  refuse "offset"
    (one
       [
         i32 0l;
         Instr.Load (Load.I32_load8_u, { Mem_arg.align = 0; offset = -1 });
         Instr.Drop;
       ]);
  refuse "store type"
    (one
       [
         i32 0l;
         i32 0l;
         Instr.Store (Store.F64_store, { Mem_arg.align = 3; offset = 0 });
       ]);
  refuse "if arm result"
    (one ~results:[ I32 ] [ i32 1l; Instr.If (Some I32, [ i32 1l ], []) ]);
  refuse "select mismatch"
    (one ~results:[ I32 ] [ i32 1l; i64 2L; i32 0l; Instr.Select ]);
  refuse "br_if carries"
    (one ~results:[ I32 ]
       [ Instr.Block (Some I32, [ i32 1l; Instr.Br_if 0; i32 0l ]) ]);
  refuse "global init type"
    (one ~globals:[ { Global.type_ = I32; mutable_ = true; init = i64 0L } ] []);
  refuse "global init not const"
    (one
       ~globals:[ { Global.type_ = I32; mutable_ = true; init = Instr.Drop } ]
       []);
  refuse "duplicate export"
    {
      (one []) with
      exports =
        [
          { Export.name = "f"; kind = Export.Func 0 };
          { Export.name = "f"; kind = Export.Func 0 };
        ];
    };
  refuse "export target"
    { (one []) with exports = [ { Export.name = "f"; kind = Export.Func 9 } ] };
  refuse "data past memory"
    { (one []) with data = [ { Data.offset = 65535; bytes = "ab" } ] };
  refuse "limits"
    { (one []) with memory = Some { Memory.min_pages = 2; max_pages = Some 1 } };
  [%expect
    {|
    underflow              error: invalid module (function 0): operand stack underflow
    operand mismatch       error: invalid module (function 0): expected i32, found i64
    result missing         error: invalid module (function 0): operand stack underflow
    extra value            error: invalid module (function 0): block ends with 1 values, expected 0
    unknown local          error: invalid module (function 0): unknown local 3
    unknown func           error: invalid module (function 0): unknown function 4
    unknown global         error: invalid module (function 0): unknown global 0
    bad label              error: invalid module (function 0): branch depth 1 names no enclosing block
    bad label in block     error: invalid module (function 0): branch depth 2 names no enclosing block
    immutable global       error: invalid module (function 0): global 0 is immutable
    no memory              error: invalid module (function 0): the module has no memory
    alignment              error: invalid module (function 0): alignment 2^3 exceeds the natural 2^2
    offset                 error: invalid module (function 0): offset -1 is outside [0, 2^31)
    store type             error: invalid module (function 0): expected f64, found i32
    if arm result          error: invalid module (function 0): operand stack underflow
    select mismatch        error: invalid module (function 0): select operands i32 and i64 differ
    br_if carries          error: invalid module (function 0): operand stack underflow
    global init type       error: invalid module: expected i32, found i64
    global init not const  error: invalid module: initializer is not a constant
    duplicate export       error: invalid module: duplicate export "f"
    export target          error: invalid module: export of unknown index 9
    data past memory       error: invalid module: data segment [65535, +2) exceeds the initial memory
    limits                 error: invalid module: memory limits min=2 max=1 |}]
