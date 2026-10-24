(* The forms each binary32 bundle's published RV32 artifacts emit, summed over
   its invocations: scalar and Zve32f plans under [Fp32_ordered]. A selection
   change moves these counts before any simulator runs. *)

module M = Machine_model.Mir_model
module T = Machine_model_test.Model_test
module R = Machine_rivet_riscv32_mir.Rv32_mir_route
module C = Machine_rivet_riscv32_mir.Rv32_mir_census

let planned target =
  Ssa_backends.Pipeline.Planned
    { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target }

let census pipeline g =
  let b = T.bundle (g ()) in
  match M.prepare ~pipeline b with
  | Error _ -> Error "refused"
  | Ok m ->
      List.fold_left2
        (fun acc inv g ->
          Result.bind acc (fun (acc : C.Census.t) ->
              Result.bind
                (R.publish ~allocation:R.Allocation.Scanned ~sites:(M.sites inv)
                   g)
                (fun published ->
                  match Err.payload (C.of_artifact published.R.artifact) with
                  | Error r ->
                      Error
                        (Fmt.str "%a"
                           Machine_rivet_riscv32_mir.Rv32_mir_form.Refusal.pp r)
                  | Ok (c : C.Census.t) ->
                      let vector =
                        List.fold_left
                          (fun v (k, n) ->
                            let n' =
                              Option.value ~default:0 (List.assoc_opt k v)
                            in
                            (k, n + n') :: List.remove_assoc k v)
                          acc.vector c.vector
                        |> List.sort (fun (a, _) (b, _) -> String.compare a b)
                      in
                      Ok
                        {
                          C.Census.instructions =
                            acc.instructions + c.instructions;
                          vector;
                          configurations = acc.configurations + c.configurations;
                          spills = acc.spills + c.spills;
                          reloads = acc.reloads + c.reloads;
                          remats = acc.remats + c.remats;
                          frame_bytes = Int64.add acc.frame_bytes c.frame_bytes;
                          helper_calls = acc.helper_calls + c.helper_calls;
                        })))
        (Ok
           {
             C.Census.instructions = 0;
             vector = [];
             configurations = 0;
             spills = 0;
             reloads = 0;
             remats = 0;
             frame_bytes = 0L;
             helper_calls = 0;
           })
        b.Loop_ir.Loop_bundle.invocations (M.generic m)

let show fixtures =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (tname, target) ->
          match census (planned target) g with
          | Error e -> Fmt.pr "%s, %s: %s@." name tname e
          | Ok c -> Fmt.pr "@[<v 2>%s, %s:@,%a@]@." name tname C.Census.pp c)
        [
          ("scalar", Ssa_ir.Ssa_target.scalar);
          ("zve32f128", Ssa_ir.Ssa_target.zve32f128);
        ])
    fixtures

let%expect_test "forms of the vector fixtures" =
  show T.vector_fixtures;
  [%expect
    {|
    relu then add over C=32, scalar:
      instructions 145, configurations 0, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector none
    relu then add over C=32, zve32f128:
      instructions 229, configurations 3, spills 0, reloads 0, remats 0, frame 128 bytes, helper calls 0
      vector vfadd.vv 4, vfmv.v.f 4, vle32.v 16, vmerge.vim 4, vmerge.vvm 4, vmflt.vv 4, vmsne.vi 4, vmv.v.i 4, vse32.v 8
    softmax over H, W=32, scalar:
      instructions 840, configurations 0, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector none
    softmax over H, W=32, zve32f128:
      instructions 840, configurations 0, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector none
    layer_norm over H, W=32, scalar:
      instructions 602, configurations 0, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector none
    layer_norm over H, W=32, zve32f128:
      instructions 602, configurations 0, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector none
    conv, batch norm, relu to C=32, scalar:
      instructions 336, configurations 0, spills 1, reloads 1, remats 5, frame 144 bytes, helper calls 0
      vector none
    conv, batch norm, relu to C=32, zve32f128:
      instructions 877, configurations 27, spills 11, reloads 11, remats 6, frame 208 bytes, helper calls 0
      vector vfadd.vv 56, vfdiv.vv 4, vfmul.vv 20, vfmv.v.f 19, vfsqrt.v 4, vfsub.vv 4, vle32.v 40, vlse32.v 12, vmerge.vim 4, vmerge.vvm 4, vmflt.vv 4, vmsne.vi 4, vmv.v.i 4, vmv.v.v 82, vse32.v 20 |}]

let%expect_test "forms of the matrix products" =
  show T.contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64, scalar:
      instructions 95, configurations 0, spills 0, reloads 0, remats 0, frame 32 bytes, helper calls 0
      vector none
    linear 9x32 k64, zve32f128:
      instructions 263, configurations 7, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector vfadd.vv 24, vfmul.vv 12, vfmv.v.f 7, vlse32.v 12, vmv.v.v 12, vse32.v 12
    linear 9x32 k64 bias, scalar:
      instructions 118, configurations 0, spills 0, reloads 0, remats 0, frame 48 bytes, helper calls 0
      vector none
    linear 9x32 k64 bias, zve32f128:
      instructions 288, configurations 7, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector vfadd.vv 24, vfmul.vv 12, vfmv.v.f 7, vle32.v 12, vlse32.v 12, vmv.v.v 12, vse32.v 12
    linear 7x29 k63 bias, scalar:
      instructions 165, configurations 0, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector none
    linear 7x29 k63 bias, zve32f128:
      instructions 253, configurations 4, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector vfadd.vv 8, vfmul.vv 4, vfmv.v.f 5, vle32.v 4, vlse32.v 4, vmv.v.v 4, vse32.v 4
    bmm 1x9x32 k64, scalar:
      instructions 92, configurations 0, spills 0, reloads 0, remats 0, frame 32 bytes, helper calls 0
      vector none
    bmm 1x9x32 k64, zve32f128:
      instructions 240, configurations 7, spills 0, reloads 0, remats 0, frame 48 bytes, helper calls 0
      vector vfadd.vv 12, vfmul.vv 12, vfmv.v.f 7, vle32.v 12, vmv.v.v 12, vse32.v 12
    bmm 3x7x29 k63, scalar:
      instructions 169, configurations 0, spills 0, reloads 0, remats 0, frame 64 bytes, helper calls 0
      vector none
    bmm 3x7x29 k63, zve32f128:
      instructions 246, configurations 4, spills 0, reloads 0, remats 0, frame 80 bytes, helper calls 0
      vector vfadd.vv 4, vfmul.vv 4, vfmv.v.f 5, vle32.v 4, vmv.v.v 4, vse32.v 4 |}]

(* The fused vector multiply-add under each binary32 policy, on the Zve32f
   plans of the matrix products: the ordered policy never fuses; the
   contracted one fuses each single-product sum's accumulation. *)
let%expect_test "vfmacc by policy" =
  List.iter
    (fun (name, g) ->
      let count numerics =
        match
          census
            (Ssa_backends.Pipeline.Planned
               { numerics; target = Ssa_ir.Ssa_target.zve32f128 })
            g
        with
        | Error e -> e
        | Ok c ->
            string_of_int
              (Option.value ~default:0
                 (List.assoc_opt "vfmacc.vv" c.C.Census.vector))
      in
      Fmt.pr "%s: fp32_ordered %s, fp32_contracted %s@." name
        (count Ssa_ir.Ssa_numerics.Fp32_ordered)
        (count Ssa_ir.Ssa_numerics.Fp32_contracted))
    T.contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64: fp32_ordered 0, fp32_contracted 12
    linear 9x32 k64 bias: fp32_ordered 0, fp32_contracted 12
    linear 7x29 k63 bias: fp32_ordered 0, fp32_contracted 4
    bmm 1x9x32 k64: fp32_ordered 0, fp32_contracted 12
    bmm 3x7x29 k63: fp32_ordered 0, fp32_contracted 4 |}]
