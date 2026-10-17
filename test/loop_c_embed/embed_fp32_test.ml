(* The binary32 half of the embedding tests: what a loader for an F-only
   target (no binary64 arithmetic) must pass. *)

open Graph_ir
open Loop_ir
open Embed_common

(* Under [Fp32_ordered] the loaded unit is binary32 throughout and equals the
   Loop interpreter at binary32, bit for bit; the binary64 reference differs
   somewhere, or the oracle would check nothing. *)
let%expect_test "Fp32_ordered: bitwise the binary32 oracle" =
  let g, b, constants = setup () in
  let p =
    Err.or_raise ~pp_error:Loop_c_embed.pp_error
      (E.prepare ~numerics:Loop_numerics.Fp32_ordered b)
  in
  let ctx = context p ~constants in
  List.iter
    (fun salt ->
      let oracle =
        Err.or_raise ~pp_error:Loop_bundle_oracle.pp_error
          (Loop_bundle_oracle.run b ~constants:(Fun.const None)
             ~inputs:(fun id ->
               match constants id with
               | Some t -> Some t
               | None -> map_of (inputs b ~salt) id))
      in
      let reference = reference g b ~constants ~salt in
      match E.run ctx ~bind:(map_of (inputs b ~salt)) |> Err.payload with
      | Error e -> Fmt.pr "run failed: %a@." Loop_c_embed.pp_error e
      | Ok outs ->
          List.iteri
            (fun k t ->
              let id = List.nth g.Graph.outputs k in
              Fmt.pr
                "salt %d: equals the oracle %b, equals the f64 reference %b@."
                salt
                (bits t = bits (List.nth oracle k))
                (bits t = bits (Tensor_id.Map.find id reference)))
            outs)
    [ 0; 2; 5 ];
  E.close_context ctx;
  E.close p;
  [%expect
    {|
    salt 0: equals the oracle true, equals the f64 reference false
    salt 2: equals the oracle true, equals the f64 reference false
    salt 5: equals the oracle true, equals the f64 reference false |}]

(* The failure record of a binary32 unit decodes as the binary64 one does. *)
let%expect_test "Fp32_ordered: a failing invocation is typed" =
  let _, b0, constants = setup () in
  let failing =
    C_host_fixtures.poisoned_bundle b0 ~at:1 Loop_failure.I64_division_by_zero
  in
  let p =
    Err.or_raise ~pp_error:Loop_c_embed.pp_error
      (E.prepare ~numerics:Loop_numerics.Fp32_ordered failing)
  in
  let ctx = context p ~constants in
  (match Err.payload (E.run ctx ~bind:(map_of (inputs b0 ~salt:0))) with
  | Error (`Inference_failed (i, e)) ->
      Fmt.pr "invocation %d, %a@." i Loop_interp.pp_error e
  | Error e -> Fmt.pr "%a@." Loop_c_embed.pp_error e
  | Ok _ -> print_endline "no failure");
  E.close_context ctx;
  E.close p;
  [%expect {| invocation 1, I64 division by zero |}]

(* The Region-authored nodes (scratch cells, a scan, the meter) under
   [Fp32_ordered], bitwise the binary32 oracle: through gcc here and through the
   embedded CompCert in test/loop_c_embed_compcert (this file is copied there,
   and into the riscv32 suite, whose branch fixup refuses sdpa's kernel: out of
   range, so sdpa is covered by the Rivet and gcc suites instead). *)
let%expect_test "Fp32_ordered: Region nodes equal the oracle" =
  let module F = Native_test.Graph_fixtures in
  List.iter
    (fun (name, g) ->
      let g = g () in
      let b =
        Err.or_raise ~pp_error:Loop_bundle.pp_error
          (Loop_bundle.build ~config:Loop_bundle_c.default_config g)
      in
      let constants =
        map_of
          (List.map
             (fun id -> (id, tensor_of ~salt:3 (sig_of_edge b id)))
             b.Loop_bundle.constants)
      in
      let p =
        Err.or_raise ~pp_error:Loop_c_embed.pp_error
          (E.prepare ~numerics:Loop_numerics.Fp32_ordered b)
      in
      let ctx = context p ~constants in
      let inputs = inputs b ~salt:1 in
      let oracle =
        Err.or_raise ~pp_error:Loop_bundle_oracle.pp_error
          (Loop_bundle_oracle.run b ~constants ~inputs:(fun id ->
               List.assoc_opt id inputs))
      in
      (match E.run ctx ~bind:(map_of inputs) |> Err.payload with
      | Error e -> Fmt.pr "%s: run failed: %a@." name Loop_c_embed.pp_error e
      | Ok outs ->
          Fmt.pr "%s: equals the oracle %b@." name
            (List.for_all2 (fun t o -> bits t = bits o) outs oracle));
      E.close_context ctx;
      E.close p)
    [
      ( "softmax",
        fun () ->
          F.build "softmax"
            Graph_builder.(
              let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
              softmax { Reduce.Softmax.axis = Axis.C } x) );
      ( "layer_norm",
        fun () ->
          F.build "layer_norm"
            Graph_builder.(
              let* x = input ~shape:(F.s 1 1 1 2 4 5) () in
              layer_norm
                { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
                ~x ()) );
    ];
  [%expect
    {|
    softmax: equals the oracle true
    layer_norm: equals the oracle true |}]
