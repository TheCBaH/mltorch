open Loop_ir
module E = Loop_c_embed.Make (Loop_c_embed_compcert.Compcert_loader)
module T = Loop_c_embed_compcert.Compcert_embed_target
module F = Native_test.Graph_fixtures

(* What CompCert makes of a bundle's C: the model's own assembly and the
   [__compcert_i64_*] helpers it links, each through the AArch64 census. *)

let asm_of ~numerics g =
  let b =
    Err.or_raise ~pp_error:Loop_bundle.pp_error
      (Loop_bundle.build ~config:Loop_bundle_c.default_config (g ()))
  in
  let p =
    Err.or_raise ~pp_error:Loop_c_embed.pp_error (E.prepare ~numerics b)
  in
  let source = E.source p in
  E.close p;
  match T.compile_to_asm ~name:"model.c" source with
  | Ok asm -> asm
  | Error e -> Fmt.failwith "%a" Compcert_embed.pp_error e

let census text =
  match Err.payload (Asm_census.of_text Asm_census.Isa.Aarch64 text) with
  | Ok c ->
      Fmt.str "%a%a" Asm_census.pp c
        Fmt.(list ~sep:nop (fun ppf (m, n) -> pf ppf "; %s x%d" m n))
        (Asm_census.mnemonics c)
  | Error e -> Fmt.str "error: %a" Asm_census.pp_error e

let softmax () =
  F.build "softmax"
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
      softmax { Reduce.Softmax.axis = Axis.C } x)

let%expect_test "baseline census of CompCert's assembly" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun numerics ->
          Fmt.pr "%s %s: %s@." name
            (Loop_numerics.name numerics)
            (census (asm_of ~numerics g)))
        Loop_numerics.[ Reference_f64; Fp32_ordered ])
    [ ("chain", F.chain); ("softmax", softmax) ];
  [%expect
    {|
    chain reference_f64: arithmetic 12, conversion 12, memory_move 7; fadd x6; fcmp x1; fcsel x1; fcvt x12; fdiv x1; fmov x5; fmul x3; fsub x1; ldr x1
    chain fp32_ordered: arithmetic 0, conversion 0, memory_move 1; fcsel x1
    softmax reference_f64: arithmetic 13, conversion 4, memory_move 13; fadd x1; fcmp x8; fcsel x2; fcvt x4; fdiv x2; fmov x2; fsub x2; ldr x6; str x3
    softmax fp32_ordered: arithmetic 4, conversion 5, memory_move 5; fcmp x3; fcsel x2; fcvt x5; fdiv x1; fmov x1; ldr x1; str x1 |}]
