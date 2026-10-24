(* The matrix-product fixtures ({!Model_test.contraction_fixtures}): the
   adversarial inputs that separate the binary32 policies, and the contracted
   policy against its oracle. Every pipeline and route is in
   [contraction_routes_test.ml]. *)

open Graph_ir
open Model_test
module M = Machine_model.Mir_model

(* The adversarial inputs round: binary32 against the binary64 reference
   differs, and most outputs stay finite. *)
let%expect_test "matrix products: adversarial inputs separate the policies" =
  List.iter
    (fun (name, g) ->
      let g' = g () in
      let b = bundle g' in
      let constants =
        List.map
          (fun id -> (id, adversarial ~salt:3 (sig_of g' id)))
          b.Loop_ir.Loop_bundle.constants
      in
      let inputs =
        List.mapi
          (fun i id -> (id, adversarial ~salt:i (sig_of g' id)))
          b.Loop_ir.Loop_bundle.inputs
      in
      let out = List.hd g'.Graph.outputs in
      let f64 =
        Tensor_id.Map.find out (expected Reference_f64 g' b ~constants ~inputs)
      in
      let f32 =
        Tensor_id.Map.find out (expected Fp32 g' b ~constants ~inputs)
      in
      let finite =
        List.length
          (List.filter
             (fun x -> Float.is_finite (Int64.float_of_bits x))
             (bits f32))
      in
      Fmt.pr "%s: %d of %d finite; binary32 oracle %s the binary64 reference@."
        name finite
        (List.length (bits f32))
        (if bits f32 = bits f64 then "equals" else "differs from"))
    contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64: 195 of 288 finite; binary32 oracle differs from the binary64 reference
    linear 9x32 k64 bias: 195 of 288 finite; binary32 oracle differs from the binary64 reference
    linear 7x29 k63 bias: 154 of 203 finite; binary32 oracle differs from the binary64 reference
    bmm 1x9x32 k64: 196 of 288 finite; binary32 oracle differs from the binary64 reference
    bmm 3x7x29 k63: 456 of 609 finite; binary32 oracle differs from the binary64 reference |}]

let%expect_test "matrix products: multiply-adds" =
  List.iter
    (fun (name, g) -> Fmt.pr "%s: %d@." name (contraction_macs g))
    contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64: 18432
    linear 9x32 k64 bias: 18432
    linear 7x29 k63 bias: 12789
    bmm 1x9x32 k64: 18432
    bmm 3x7x29 k63: 38367 |}]

(* The contracted oracle is the reference's binary32 sequential dot: the Loop
   interpreter with each single-product sum fused, against [Direct] under
   [Binary32_sequential], two independent evaluators. With the bias added,
   the reference adds it in binary64 and rounds once on store, as the oracle's
   binary32 add does. The unfused oracle differs on adversarial inputs. *)
let%expect_test "matrix products: the contracted oracle is the sequential dot" =
  List.iter
    (fun (name, g) ->
      let g' = g () in
      let b = bundle g' in
      List.iter
        (fun (iname, input) ->
          let constants =
            List.map
              (fun id -> (id, input ~salt:3 (sig_of g' id)))
              b.Loop_ir.Loop_bundle.constants
          in
          let inputs =
            List.mapi
              (fun i id -> (id, input ~salt:i (sig_of g' id)))
              b.Loop_ir.Loop_bundle.inputs
          in
          let out = List.hd g'.Graph.outputs in
          let get o =
            Tensor_id.Map.find out (expected o g' b ~constants ~inputs)
          in
          let direct =
            Direct.with_dot_accumulation Direct.Binary32_sequential (fun () ->
                get Reference_f64)
          in
          let contracted = get Fp32_contracted and ordered = get Fp32 in
          Fmt.pr "%s, %s: contracted %s the sequential reference; ordered %s@."
            name iname
            (if same_portable contracted direct then "equals"
             else "DIFFERS from")
            (if same_portable ordered direct then "equals it" else "differs"))
        [
          ("values", fun ~salt sg -> values ~salt sg);
          ("adversarial", fun ~salt sg -> adversarial ~salt sg);
        ])
    contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64, values: contracted equals the sequential reference; ordered equals it
    linear 9x32 k64, adversarial: contracted equals the sequential reference; ordered differs
    linear 9x32 k64 bias, values: contracted equals the sequential reference; ordered equals it
    linear 9x32 k64 bias, adversarial: contracted equals the sequential reference; ordered differs
    linear 7x29 k63 bias, values: contracted equals the sequential reference; ordered equals it
    linear 7x29 k63 bias, adversarial: contracted equals the sequential reference; ordered differs
    bmm 1x9x32 k64, values: contracted equals the sequential reference; ordered equals it
    bmm 1x9x32 k64, adversarial: contracted equals the sequential reference; ordered differs
    bmm 3x7x29 k63, values: contracted equals the sequential reference; ordered equals it
    bmm 3x7x29 k63, adversarial: contracted equals the sequential reference; ordered differs |}]

(* [Fp32_contracted]: each single-product sum's accumulation fused, on the
   generic interpreter and both targets' scanned programs, scalar and
   vectorized, against the contracted oracle. *)
let%expect_test "matrix products: the contracted policy is its oracle" =
  let contracted target =
    Ssa_backends.Pipeline.Planned
      { numerics = Ssa_ir.Ssa_numerics.Fp32_contracted; target }
  in
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (pname, pipeline) ->
          List.iter
            (fun (rname, route) ->
              check ?route ~pipeline ~oracle:Fp32_contracted ~same:same_portable
                ~input:adversarial_input
                ~constant:(fun sg -> adversarial ~salt:3 sg)
                (Fmt.str "%s, %s, %s" name pname rname)
                g)
            [
              ("generic", None);
              ("aarch64", Some (M.Route.Aarch64 M.Stage.Scanned));
              ("x86_64", Some (M.Route.X86_64 M.Stage.Scanned));
            ])
        [
          ("scalar", contracted Ssa_ir.Ssa_target.scalar);
          ("neon128", contracted Ssa_ir.Ssa_target.neon128);
        ])
    (contraction_fixtures @ [ List.nth vector_fixtures 3 ]);
  [%expect
    {|
    linear 9x32 k64, scalar, generic (1 invocations): bitwise, bitwise
    linear 9x32 k64, scalar, aarch64 (1 invocations): bitwise, bitwise
    linear 9x32 k64, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 9x32 k64, neon128, generic (1 invocations): bitwise, bitwise
    linear 9x32 k64, neon128, aarch64 (1 invocations): bitwise, bitwise
    linear 9x32 k64, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 9x32 k64 bias, scalar, generic (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, scalar, aarch64 (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 9x32 k64 bias, neon128, generic (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, neon128, aarch64 (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 7x29 k63 bias, scalar, generic (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, scalar, aarch64 (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 7x29 k63 bias, neon128, generic (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, neon128, aarch64 (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    bmm 1x9x32 k64, scalar, generic (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, scalar, aarch64 (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    bmm 1x9x32 k64, neon128, generic (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, neon128, aarch64 (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    bmm 3x7x29 k63, scalar, generic (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, scalar, aarch64 (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    bmm 3x7x29 k63, neon128, generic (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, neon128, aarch64 (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    conv, batch norm, relu to C=32, scalar, generic (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, scalar, aarch64 (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    conv, batch norm, relu to C=32, neon128, generic (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, neon128, aarch64 (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include |}]
