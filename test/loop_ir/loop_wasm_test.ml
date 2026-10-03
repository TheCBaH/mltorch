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
    gather                 401 bytes, md5 e50e49e668546dacf847b20652b435cd
    i64 division by zero   387 bytes, md5 752fb0e5e1623beb4844970770dc8630
    i64 division overflow  387 bytes, md5 4e6466a408711d6b0e0738e239e38c0f
    i64 from float         448 bytes, md5 66dc896b7b224a5e2084d52a1812c84d
    index overflow         485 bytes, md5 06f1f38931c0b99bd30c5e209e7ba20b
    load out of range      616 bytes, md5 086dd9c05e0107330ad001b5acb23daa
    local out of range     394 bytes, md5 5205f21b493b315cd29433996e03c12f
    scan lane              431 bytes, md5 98831e2533a05848bd6d7f846de4dba1
    scan row               431 bytes, md5 8eb48d7c80478547d80b5f809a7d1185 |}]

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
    chain        3 invocations, 3 kernels, 1108 bytes, md5 667fe40e9c184d345d40b327fffe9a54, memory 1408 bytes
    residual     3 invocations, 2 kernels, 626 bytes, md5 51e1a375a140fca7361504e60a36ac1d, memory 640 bytes
    layer_norm   2 invocations, 2 kernels, 1072 bytes, md5 542288b5f71f06882795734961cd2d45, memory 1024 bytes
    sdpa         2 invocations, 2 kernels, 1368 bytes, md5 60281ccfd7da69ce62fe1954cdb78dbb, memory 2048 bytes |}]

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
  [%expect {| 1260233 bytes, md5 1ffff941c40d5a4432da5e51e6320ead |}]

(* ---- the manifest: what a host should know before compiling ---------------- *)

let manifest_of (m : Wasm.Module.t) =
  match
    List.find_opt
      (fun (c : Wasm.Custom.t) -> c.Wasm.Custom.name = "manifest")
      m.Wasm.Module.customs
  with
  | Some c -> c.Wasm.Custom.payload
  | None -> "(no manifest)"

let%expect_test "the manifest names the ABI, features, imports and helpers" =
  let show name m = Fmt.pr "-- %s@.%s" name (manifest_of m) in
  show "doubling" (lowered Loop_programs.doubling).Loop_wasm.module_;
  let overflowing =
    Loop_index.Add (Loop_index.Scale (3, i0), Loop_index.Const 5)
  in
  show "index overflow check"
    (lowered
       (failing (Loop_bool.Index_overflows overflowing)
          (Loop_failure.Index_overflow { index = overflowing })))
      .Loop_wasm.module_;
  show "float to i64"
    (lowered
       (failing always
          (Loop_failure.I64_from_float { value = Loop_expr.Const 1. })))
      .Loop_wasm.module_;
  let g = Native_test.Graph_fixtures.sink_permute_layer_norm () in
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error
      (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)
  in
  show "layer_norm model"
    (Err.or_raise ~pp_error:Loop_bundle_wasm.pp_error (Loop_bundle_wasm.build b))
      .Loop_bundle_wasm.module_;
  [%expect
    {|
    -- doubling
    loop-wasm/1
    features:
    imports:
    helpers:
    numerics: working=f64 f32=round-and-widen fma=none reassociation=none i64=modular index=i32-checked
    -- index overflow check
    loop-wasm/1
    features:
    imports:
    helpers: fail_set
    numerics: working=f64 f32=round-and-widen fma=none reassociation=none i64=modular index=i32-checked
    -- float to i64
    loop-wasm/1
    features:
    imports:
    helpers: fail_set
    numerics: working=f64 f32=round-and-widen fma=none reassociation=none i64=modular index=i32-checked
    -- layer_norm model
    loop-wasm/1
    features: bulk-memory
    imports:
    helpers: fill_f32
    numerics: working=f64 f32=round-and-widen fma=none reassociation=none i64=modular index=i32-checked |}]

(* ---- memory admission: every region is bounded before a pointer is made ----- *)

let%expect_test "a model whose memory exceeds the 2 GiB policy is refused" =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:600_000_000 in
  let g =
    Graph_builder.build ~name:"too_big"
      ~outputs:(fun o -> [ o ])
      Graph_builder.(
        let* x = input ~shape () in
        relu x)
    |> Err.or_raise ~pp_error:Graph_builder.pp_error
  in
  (match
     Err.payload (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)
   with
  | Error e -> Fmt.pr "bundle: %a@." Loop_bundle.pp_error e
  | Ok b -> (
      match Err.payload (Loop_bundle_wasm.build b) with
      | Error e -> Fmt.pr "refused: %a@." Loop_bundle_wasm.pp_error e
      | Ok w ->
          Fmt.pr "accepted with %d bytes of memory@."
            w.Loop_bundle_wasm.placement.Loop_bundle_wasm.Placement.total));
  [%expect {| refused: 2400000000 bytes exceed the 2 GiB memory policy |}]
