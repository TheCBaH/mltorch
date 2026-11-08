open Ssa_ir
module B = Ssa_builder
module F = Ssa_fixtures
module E = Ssa_expand_contract_test
module P = Ssa_contract_plan
module Engine = Contraction.Select.Engine

let capabilities matrix =
  { Contraction.Select.Capabilities.matrix; vector = true }

let desc ?(format = Contraction.Element.F32) numerics =
  E.desc ~format ~m:2 ~n:3 ~k:4 ~seed:Contraction.Seed.Zero ~numerics ()

let program ?engine d =
  let input_format =
    match d.Contraction.format with
    | Contraction.Element.Bf16 -> Ssa_format.Bf16
    | _ -> Ssa_format.F32
  in
  F.build
    ~buffers:
      [
        F.buffer 0 ~h:1L ~w:8L input_format Ssa_buffer.Input;
        F.buffer 1 ~h:1L ~w:12L input_format Ssa_buffer.Input;
        F.buffer 2 ~h:1L ~w:6L Ssa_format.F32 Ssa_buffer.Output;
      ]
    (fun b -> B.contract ?engine b d ~a:(F.buf 0) ~b:(F.buf 1) ~c:(F.buf 2))

let planned caps p =
  Err.or_raise ~pp_error:P.pp_error (P.program ~capabilities:caps p)

let engine p = match P.choices p with [ (_, e) ] -> e | _ -> assert false

let%test_unit "final SSA chooses an engine without relaxing numerics" =
  List.iter
    (fun (format, numerics, matrix, want) ->
      let p = program (desc ~format numerics) in
      assert (P.choices p = []);
      let q = planned (capabilities matrix) p in
      assert (engine q = want);
      assert (Ssa_pp.to_string p <> Ssa_pp.to_string q);
      assert (Result.is_ok (Err.payload (Ssa_verify.check q)));
      assert (
        Ssa_pp.to_string q = Ssa_pp.to_string (planned (capabilities matrix) q)))
    [
      (Contraction.Element.F32, Contraction.Numerics.Ordered, true, Engine.Zvt);
      (Contraction.Element.F32, Contraction.Numerics.Ordered, false, Engine.Rvv);
      ( Contraction.Element.F32,
        Contraction.Numerics.Fused_sequential,
        true,
        Engine.Rvv );
      (Contraction.Element.Bf16, Contraction.Numerics.Ordered, true, Engine.Rvv);
      (Contraction.Element.Bf16, Contraction.Numerics.Paired, true, Engine.Zvt);
    ]

let%test_unit
    "finalization preserves an explicitly selected legal vector engine" =
  let d = desc Contraction.Numerics.Ordered in
  let p = program ~engine:Engine.Rvv d in
  assert (engine (planned (capabilities true) p) = Engine.Rvv)

let%test_unit "an unsupported recorded engine is refused rather than replaced" =
  let d = desc Contraction.Numerics.Ordered in
  let p = program ~engine:Engine.Zvt d in
  match Err.payload (P.program ~capabilities:(capabilities false) p) with
  | Error (`Unsupported_contraction_engine u) -> assert (u.engine = Engine.Zvt)
  | _ -> assert false

let%test_unit "paired arithmetic cannot silently become a vector contraction" =
  let p =
    program (desc ~format:Contraction.Element.Bf16 Contraction.Numerics.Paired)
  in
  match Err.payload (P.program ~capabilities:(capabilities false) p) with
  | Error (`Unsupported_contraction_engine _) -> ()
  | _ -> assert false

let%test_unit "a matrix engine cannot silently replace a fused sequential dot" =
  let p =
    program ~engine:Engine.Zvt (desc Contraction.Numerics.Fused_sequential)
  in
  match Err.payload (P.program ~capabilities:(capabilities true) p) with
  | Error (`Unsupported_contraction_engine _) -> ()
  | _ -> assert false

let%test_unit "selecting implementations preserves the SSA interpreter's bits" =
  let p = program (desc Contraction.Numerics.Ordered) in
  let a_cells = F.floats (Array.init 8 (fun i -> float_of_int (i - 3) /. 7.)) in
  let b_cells =
    F.floats (Array.init 12 (fun i -> float_of_int (i - 5) /. 3.))
  in
  let run p =
    E.run p ~a_cells ~b_cells ~nc:6 |> Array.map Int32.bits_of_float
  in
  let want = run p in
  List.iter
    (fun matrix -> assert (run (planned (capabilities matrix) p) = want))
    [ false; true ]

let%test_unit "conflicting engines for one helper binding are refused" =
  let d = desc Contraction.Numerics.Ordered in
  let p = program ~engine:Engine.Rvv d in
  let q =
    F.build ~buffers:p.Ssa_program.buffers (fun b ->
        B.contract ~engine:Engine.Rvv b d ~a:(F.buf 0) ~b:(F.buf 1) ~c:(F.buf 2);
        B.contract ~engine:Engine.Zvt b d ~a:(F.buf 0) ~b:(F.buf 1) ~c:(F.buf 2))
  in
  match Err.payload (P.program ~capabilities:(capabilities true) q) with
  | Error (`Contraction_engine_conflict c) -> assert (c.first <> c.second)
  | _ -> assert false
