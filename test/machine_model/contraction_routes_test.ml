(* The matrix-product fixtures ({!Model_test.contraction_fixtures}) through
   every pipeline and route. Its own file so that the partition stays under the
   inline-test time ceiling. *)

open Model_test
module M = Machine_model.Mir_model

let%expect_test "matrix products: every pipeline and route is bitwise" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (pname, pipeline, oracle) ->
          List.iter
            (fun (rname, route) ->
              List.iter
                (fun (iname, input, constant) ->
                  check ?route ~pipeline ~oracle ~input ~constant
                    (Fmt.str "%s, %s, %s, %s" name pname rname iname)
                    g)
                [
                  ("values", (fun ~salt _ -> values ~salt), values ~salt:3);
                  ( "adversarial",
                    adversarial_input,
                    fun sg -> adversarial ~salt:3 sg );
                ])
            [
              ("generic", None);
              ("aarch64", Some (M.Route.Aarch64 M.Stage.Scanned));
              ("x86_64", Some (M.Route.X86_64 M.Stage.Scanned));
            ])
        [
          ("exact", Ssa_backends.Pipeline.Exact, Reference_f64);
          ("fp32", fp32_ordered, Fp32);
          ("fp32 neon128", fp32_vector, Fp32);
        ])
    contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64, exact, generic, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, exact, generic, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64, exact, aarch64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, exact, aarch64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64, exact, x86_64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, exact, x86_64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32, generic, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32, generic, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32, aarch64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32, aarch64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32, x86_64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32, x86_64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32 neon128, generic, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32 neon128, generic, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32 neon128, aarch64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32 neon128, aarch64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32 neon128, x86_64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, fp32 neon128, x86_64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, exact, generic, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, exact, generic, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, exact, aarch64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, exact, aarch64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, exact, x86_64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, exact, x86_64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32, generic, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32, generic, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32, aarch64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32, aarch64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32, x86_64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32, x86_64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32 neon128, generic, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32 neon128, generic, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32 neon128, aarch64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32 neon128, aarch64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32 neon128, x86_64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, fp32 neon128, x86_64, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, exact, generic, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, exact, generic, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, exact, aarch64, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, exact, aarch64, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, exact, x86_64, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, exact, x86_64, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32, generic, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32, generic, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32, aarch64, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32, aarch64, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32, x86_64, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32, x86_64, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32 neon128, generic, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32 neon128, generic, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32 neon128, aarch64, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32 neon128, aarch64, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32 neon128, x86_64, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, fp32 neon128, x86_64, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, exact, generic, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, exact, generic, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, exact, aarch64, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, exact, aarch64, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, exact, x86_64, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, exact, x86_64, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32, generic, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32, generic, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32, aarch64, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32, aarch64, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32, x86_64, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32, x86_64, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32 neon128, generic, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32 neon128, generic, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32 neon128, aarch64, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32 neon128, aarch64, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32 neon128, x86_64, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, fp32 neon128, x86_64, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, exact, generic, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, exact, generic, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, exact, aarch64, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, exact, aarch64, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, exact, x86_64, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, exact, x86_64, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32, generic, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32, generic, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32, aarch64, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32, aarch64, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32, x86_64, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32, x86_64, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32 neon128, generic, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32 neon128, generic, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32 neon128, aarch64, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32 neon128, aarch64, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32 neon128, x86_64, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, fp32 neon128, x86_64, adversarial (1 invocations): bitwise, bitwise |}]
