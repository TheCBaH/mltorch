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
        (func 0 (param i32 i32 i32) (result i32)
          (local i32)
          i32.const 0
          local.set 3
          block
            loop
              local.get 3
              i32.const 4
              i32.ge_s
              br_if 1
              local.get 2
              local.get 3
              i32.const 2
              i32.shl
              i32.add
              local.get 1
              local.get 3
              i32.const 2
              i32.shl
              i32.add
              f32.load offset=0 align=4
              f64.promote_f32
              f64.const 0x4000000000000000 (0x1p+1)
              f64.mul
              f32.demote_f64
              f32.store offset=0 align=4
              local.get 3
              i32.const 1
              i32.add
              local.set 3
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
    gather                 240 bytes, md5 ca919d65659454900a20d770a3d35544
    i64 division by zero   226 bytes, md5 3cc15f0b0d4baf84f693b0205ba9bf31
    i64 division overflow  226 bytes, md5 0cfe428c45a57d8f3f8bdb4120a4fe12
    i64 from float         287 bytes, md5 fc19fc1bc1aab1a16512ab0823d5a60d
    index overflow         324 bytes, md5 07eb66d3555d4f4822fd496402a29d23
    load out of range      455 bytes, md5 def5557e43d6d068d2f1e692d6384b7f
    local out of range     233 bytes, md5 d6cca77caa1f0cfe818e0224d48e4124
    scan lane              270 bytes, md5 350c621696159eab69280a38ed1a1877
    scan row               270 bytes, md5 e9acb33106434c4cbc9c58848a8f00af |}]

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

(* ---- whole models: bytes and layouts, native and 32-bit [int] -------------- *)

let model_summary g =
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error
      (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)
  in
  let w =
    Err.or_raise ~pp_error:Loop_bundle_wasm.pp_error (Loop_bundle_wasm.build b)
  in
  let st = w.Loop_bundle_wasm.stats in
  Fmt.str "%d invocations, %d kernels, %d bytes, md5 %s, memory %d bytes"
    st.Loop_bundle_wasm.invocations st.Loop_bundle_wasm.distinct_kernels
    st.Loop_bundle_wasm.module_bytes
    (Digest.to_hex w.Loop_bundle_wasm.identity)
    w.Loop_bundle_wasm.placement.Loop_bundle_wasm.Placement.total

let%expect_test "whole-model modules are identical natively and under jsoo" =
  List.iter
    (fun (name, g) -> Fmt.pr "%-12s %s@." name (model_summary g))
    [
      ("chain", Native_test.Graph_fixtures.chain ());
      ("residual", Native_test.Graph_fixtures.residual ());
      ("layer_norm", Native_test.Graph_fixtures.sink_permute_layer_norm ());
      ("sdpa", Native_test.Graph_fixtures.sink_permute_sdpa ());
    ];
  [%expect
    {|
    chain        3 invocations, 3 kernels, 935 bytes, md5 97246f834f7160adb3dc8dec061ab634, memory 1408 bytes
    residual     3 invocations, 2 kernels, 453 bytes, md5 c756474f6c8cd53a32161975024daa32, memory 640 bytes
    layer_norm   2 invocations, 2 kernels, 899 bytes, md5 84cb2209bd3c90ac5457073f3b2ccd0f, memory 1024 bytes
    sdpa         2 invocations, 2 kernels, 1186 bytes, md5 7cbce0f175154662a31f39a8959cf9ef, memory 2048 bytes |}]

let%expect_test "an unsupported storage configuration is refused, not degraded"
    =
  let g = Native_test.Graph_fixtures.chain () in
  (match
     Err.payload
       (Loop_bundle.build
          ~config:
            {
              Loop_bundle_wasm.default_config with
              Storage_script.Config.inputs = Storage_script.Ownership.Copied;
            }
          g)
   with
  | Error e -> Fmt.pr "bundle: %a@." Loop_bundle.pp_error e
  | Ok b -> (
      match Err.payload (Loop_bundle_wasm.build b) with
      | Error e -> Fmt.pr "refused: %a@." Loop_bundle_wasm.pp_error e
      | Ok _ -> Fmt.pr "accepted@."));
  [%expect
    {| refused: storage config layout=separate constants=borrowed inputs=copied: the whole-model backends admit only separate layout with borrowed constants and inputs |}]

(* js_of_ocaml's stack is far shallower than a native one: a kernel body of tens
   of thousands of statements must lower without recursing once per element. *)
let%expect_test
    "a very long kernel body lowers and encodes under a shallow stack" =
  let b = buffer 1 (shape_w 2) f32 Loop_buffer.Output in
  let store k =
    Loop_stmt.Store
      {
        buffer = b;
        coord = at_w (Loop_index.Const (k land 1));
        value = Loop_stored.F32 (Loop_expr.Const (float_of_int k));
      }
  in
  let p = program ~buffers:[ b ] (List.init 60_000 store) in
  Fmt.pr "%s@." (summary p);
  [%expect {| 1260081 bytes, md5 9d5366ae7edb55f062d6a5c66094841c |}]
