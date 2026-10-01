open Loop_ir
open Loop_fixtures

(* The emitted module's bytes are pure data, so these expect blocks also run
   under js_of_ocaml ((modes best js) in this directory's dune): the digest and
   size of each module must be identical natively and under a 32-bit [int]. *)

let lowered p =
  match Err.payload (Loop_wasm.lower p) with
  | Ok l -> l
  | Error e -> Fmt.failwith "%a" Loop_wasm.pp_error e

let bytes p =
  match Err.payload (Loop_wasm.encode (lowered p)) with
  | Ok s -> s
  | Error e -> Fmt.failwith "%a" Wasm_check.pp_error e

let summary p =
  let s = bytes p in
  Fmt.str "%d bytes, md5 %s" (String.length s) (Digest.to_hex (Digest.string s))

let%expect_test "a kernel is one exported function over linear memory" =
  Fmt.pr "%a@." Wasm_wat.pp (lowered Loop_programs.doubling).Loop_wasm.module_;
  Fmt.pr "heap_base %d@." (lowered Loop_programs.doubling).Loop_wasm.heap_base;
  [%expect
    {|
      (module
        (memory 1)
        (func 0 (param i32 i32) (result i32)
          (local i32)
          i32.const 0
          local.set 2
          block
            loop
              local.get 2
              i32.const 4
              i32.ge_s
              br_if 1
              local.get 1
              local.get 2
              i32.const 2
              i32.shl
              i32.add
              local.get 0
              local.get 2
              i32.const 2
              i32.shl
              i32.add
              f32.load offset=0 align=4
              f64.promote_f32
              f64.const 0x4000000000000000 (0x1p+1)
              f64.mul
              f32.demote_f64
              f32.store offset=0 align=4
              local.get 2
              i32.const 1
              i32.add
              local.set 2
              br 0
            end
          end
          i32.const 0
        )
        (export "memory" (memory 0))
        (export "loop_kernel" (func 0))
      )
      heap_base 112
    |}]

let%expect_test "lowering twice is byte-identical, whatever the allocation ids"
    =
  let build ~var ~temp:t =
    let i = Loop_index.Var (v var) in
    program
      ~buffers:[ buffer 1 (shape_w 2) f32 Loop_buffer.Output ]
      [
        Loop_stmt.For
          {
            var = v var;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 2;
            body =
              [
                Loop_stmt.Assign
                  (Loop_carrier.Float, temp t, Loop_expr.Value_of_index i);
                Loop_stmt.Store
                  {
                    buffer = buffer 1 (shape_w 2) f32 Loop_buffer.Output;
                    coord = at_w i;
                    value =
                      Loop_stored.F32
                        (Loop_expr.Temp (Loop_carrier.Float, temp t));
                  };
              ];
          };
      ]
  in
  let a = bytes (build ~var:0 ~temp:0) in
  Fmt.pr "%b@." (String.equal a (bytes (build ~var:0 ~temp:0)));
  Fmt.pr "%b@." (String.equal a (bytes (build ~var:31 ~temp:77)));
  [%expect {|
    true
    true |}]

let i0 = Loop_index.Var (v 0)

let failing ?(buffers = []) pred failure =
  program ~buffers
    [
      Loop_stmt.For
        {
          var = v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 1;
          body = [ Loop_stmt.Fail_if (pred, failure) ];
        };
    ]

let always = Loop_bool.Index_lt (Loop_index.Const 0, Loop_index.Const 1)
let local = Expr.Builder.run Expr.Builder.fresh_local

let%expect_test "one module per failure constructor, stable across backends" =
  let b = buffer 1 (shape_w 4) f32 Loop_buffer.Input in
  let show name p = Fmt.pr "%-22s %s@." name (summary p) in
  show "gather"
    (failing always
       (Loop_failure.Gather_out_of_range
          { raw = Loop_expr.I64_const (-5L); extent = 3 }));
  show "i64 division by zero" (failing always Loop_failure.I64_division_by_zero);
  show "i64 division overflow"
    (failing always Loop_failure.I64_division_overflow);
  show "i64 from float"
    (failing always
       (Loop_failure.I64_from_float { value = Loop_expr.Const 1. }));
  let overflowing =
    Loop_index.Add (Loop_index.Scale (3, i0), Loop_index.Const 5)
  in
  show "index overflow"
    (failing (Loop_bool.Index_overflows overflowing)
       (Loop_failure.Index_overflow { index = overflowing }));
  show "load out of range"
    (failing ~buffers:[ b ] always
       (Loop_failure.Load_out_of_range { buffer = b; coord = at_w i0 }));
  show "local out of range"
    (failing always
       (Loop_failure.Local_out_of_range { local; index = i0; extent = 2 }));
  show "scan lane"
    (failing always
       (Loop_failure.Scan_lane_out_of_range
          {
            local = Some local;
            row = i0;
            lane = Loop_index.Const 1;
            extent = 2;
          }));
  show "scan row"
    (failing always
       (Loop_failure.Scan_row_out_of_range
          { local = None; row = i0; lane = Loop_index.Const 1; extent = 2 }));
  [%expect
    {|
    gather                 239 bytes, md5 945224237e3e5b9784faab3de1e11dbd
    i64 division by zero   225 bytes, md5 f7ac7a8926349791f9b1b72736b52ff0
    i64 division overflow  225 bytes, md5 eff58f60ebe5a25363bbade721ffb18c
    i64 from float         286 bytes, md5 14377ef2d12de351ea60d304a1132d0f
    index overflow         323 bytes, md5 c746ed6348e88a5fd1c0736dfa237d02
    load out of range      454 bytes, md5 06bd3517c0ec58bddcb069ee73911af8
    local out of range     232 bytes, md5 ca51c525dfbf665e47157df478aebc25
    scan lane              269 bytes, md5 76b614fc29d6d3912038ae8cea4a5bae
    scan row               269 bytes, md5 53ec7262a4a5b6dccd26bfe08f45a490 |}]

let%expect_test "constants outside the 32-bit index domain are refused" =
  let p =
    program
      ~buffers:[ buffer 1 (shape_w 2) f32 Loop_buffer.Output ]
      [
        Loop_stmt.Assign_index
          (temp 0, Loop_index.Floor_div_pos (Loop_index.Const 1, 0));
      ]
  in
  (match Err.payload (Loop_wasm.lower p) with
  | Ok _ -> Fmt.pr "accepted@."
  | Error e -> Fmt.pr "%a@." Loop_wasm.pp_error e);
  [%expect {| index constant 0 is not a valid 32-bit index operand |}]
