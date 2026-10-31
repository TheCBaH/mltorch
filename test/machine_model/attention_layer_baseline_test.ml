(* The baseline the granted attention layer is measured against: the layer
   decomposed into existing nodes (the hand-written fixture), through the
   generic, AArch64 and x86-64 Machine IR routes, every node on its own
   generated path. Under the binary64 pipeline it is the reference; under the
   binary32 plans it is the Loop oracle at binary32. Its own file for the
   inline-test time ceiling. *)

open Model_test
module M = Machine_model.Mir_model
module LF = Native_test.Attention_layer_fixtures
module L = Attention_layer.Attention_layer

let fp32_contracted =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_contracted;
      target = Ssa_ir.Ssa_target.scalar;
    }

let pre = LF.with_norm L.Norm.Kind.Layer L.Placement.Pre (LF.all_flags LF.small)

let routes =
  [
    ("generic", None);
    ("aarch64", Some (M.Route.Aarch64 M.Stage.Scanned));
    ("x86_64", Some (M.Route.X86_64 M.Stage.Scanned));
  ]

let%expect_test "the decomposed layer through every route" =
  List.iter
    (fun (fname, c) ->
      List.iter
        (fun (pname, pipeline, oracle) ->
          List.iter
            (fun (rname, route) ->
              check ?route ~pipeline ~oracle ~same:same_portable ~calls:1
                ~input:(fun ~salt:_ _ sg -> values ~salt:0 sg)
                (Fmt.str "%s, %s, %s" fname pname rname)
                (fun () -> LF.decomposed c))
            (if pname = "exact" then [ List.hd routes ] else routes))
        [
          ("exact", Ssa_backends.Pipeline.Exact, Reference_f64);
          ("fp32_ordered", fp32_ordered, Fp32);
          ("fp32_contracted", fp32_contracted, Fp32_contracted);
        ])
    [ ("no norm", LF.all_flags LF.small); ("pre layer", pre) ];
  [%expect
    {|
    no norm, exact, generic (15 invocations): bitwise
    no norm, fp32_ordered, generic (15 invocations): bitwise
    no norm, fp32_ordered, aarch64 (15 invocations): bitwise
    no norm, fp32_ordered, x86_64 (15 invocations): bitwise
    no norm, fp32_contracted, generic (15 invocations): bitwise
    no norm, fp32_contracted, aarch64 (15 invocations): bitwise
    no norm, fp32_contracted, x86_64 (15 invocations): bitwise
    pre layer, exact, generic (16 invocations): bitwise
    pre layer, fp32_ordered, generic (16 invocations): bitwise
    pre layer, fp32_ordered, aarch64 (16 invocations): bitwise
    pre layer, fp32_ordered, x86_64 (16 invocations): bitwise
    pre layer, fp32_contracted, generic (16 invocations): bitwise
    pre layer, fp32_contracted, aarch64 (16 invocations): bitwise
    pre layer, fp32_contracted, x86_64 (16 invocations): bitwise |}]
