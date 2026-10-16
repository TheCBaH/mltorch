(* Whole bundles under a planned binary32 vector policy. The oracle is not the
   binary64 reference: it is the same planned program on the selected-stage
   interpreter, which the native code must equal bit for bit; the reference is
   only a tolerance check that the policy computes the same function. *)

module F = Native_test.Graph_fixtures
module T = Machine_model_test.Model_test
module H = Machine_rivet_aarch64.Rivet_a64_host
module Mm = Machine_model.Mir_model
module Rt = Machine_rivet_aarch64.Rivet_a64_route
module Rn = Machine_rivet_aarch64.Rivet_a64_runtime
open Graph_ir

let planned ?(numerics = Ssa_ir.Ssa_numerics.Simd_fp32_ordered) () =
  Ssa_backends.Pipeline.Planned { numerics; target = Ssa_ir.Ssa_target.neon128 }

let bits_of t = T.bits t

(* The largest relative difference between two tensors' cells. *)
let relative_error a b =
  let worst = ref 0. in
  List.iter2
    (fun x y ->
      let x = Int64.float_of_bits x and y = Int64.float_of_bits y in
      let d = Float.abs (x -. y) /. Float.max 1. (Float.abs y) in
      if Float.is_nan d then worst := Float.infinity
      else worst := Float.max !worst d)
    (bits_of a) (bits_of b);
  !worst

let check ?(allocation = Rt.Allocation.Reference)
    ?(runtime = Rn.Dependency_free)
    ?(numerics = Ssa_ir.Ssa_numerics.Simd_fp32_ordered) ?(calls = 2) ?constant
    ?(input = fun ~salt _ -> T.values ~salt) name g =
  let g = g () in
  let b = T.bundle g in
  let constant = Option.value constant ~default:(T.values ~salt:3) in
  let constants =
    List.map
      (fun id -> (id, constant (T.sig_of g id)))
      b.Loop_ir.Loop_bundle.constants
  in
  let pipeline = planned ~numerics () in
  let interp =
    Mm.prepare ~route:(Mm.Route.Aarch64 Mm.Stage.Selected) ~pipeline b
  in
  match (interp, H.prepare ~allocation ~runtime ~pipeline b) with
  | Error rs, _ | _, Error rs ->
      Fmt.pr "%s: refused: %a@." name
        Fmt.(list ~sep:(any "; ") Mm.Refusal.pp)
        rs
  | Ok mi, Ok host -> (
      match
        ( Mm.Context.create mi ~constants:(fun id -> List.assoc_opt id constants),
          H.Context.create host ~constants:(fun id ->
              List.assoc_opt id constants) )
      with
      | Error s, _ | _, Error s -> Fmt.pr "%s: %a@." name Mm.Stop.pp s
      | Ok ci, Ok cn ->
          let verdicts =
            List.init calls (fun call ->
                let inputs =
                  List.mapi
                    (fun i id -> (id, input ~salt:(call + i) i (T.sig_of g id)))
                    b.Loop_ir.Loop_bundle.inputs
                in
                let lookup id = List.assoc_opt id inputs in
                let reference =
                  Err.or_raise ~pp_error:Eval_direct.pp_error
                    (Eval_direct.run g ~constants ~inputs)
                in
                match
                  ( Mm.Context.run ci ~inputs:lookup,
                    H.Context.run cn ~inputs:lookup )
                with
                | Error s, _ | _, Error s -> Fmt.str "%a" Mm.Stop.pp s
                | Ok oi, Ok on ->
                    let same =
                      List.for_all2 (fun a b -> bits_of a = bits_of b) oi on
                    in
                    let err =
                      List.fold_left2
                        (fun w id t ->
                          Float.max w
                            (relative_error t (Tensor_id.Map.find id reference)))
                        0. g.Graph.outputs on
                    in
                    Fmt.str "%s, within 1e-4 of binary64: %b"
                      (if same then "bitwise as interpreted" else "DIFFERS")
                      (err < 1e-4))
          in
          Fmt.pr "%s (%d invocations): %s@." name (H.invocations host)
            (String.concat "; " verdicts))

let%expect_test "conv, batch norm, relu under the planned policy" =
  check ~constant:T.positive ~calls:3 "chain" F.chain;
  check ~constant:T.positive "wide chain" T.wide_chain;
  [%expect
    {|
    chain (3 invocations): bitwise as interpreted, within 1e-4 of binary64: true; bitwise as interpreted, within 1e-4 of binary64: true; bitwise as interpreted, within 1e-4 of binary64: true
    wide chain (3 invocations): bitwise as interpreted, within 1e-4 of binary64: true; bitwise as interpreted, within 1e-4 of binary64: true |}]

(* ---- Fp32_ordered through the table binding ---------------------------------- *)

let fp32_ordered = T.fp32_ordered

let table_check ?(allocation = Rt.Allocation.Reference)
    ?(pipeline = fp32_ordered) ?constant ?(calls = 2)
    ?(input = fun ~salt _ sg -> T.values ~salt sg) name g =
  let g = g () in
  let b = T.bundle g in
  let constant = Option.value constant ~default:T.positive in
  let constants =
    List.map
      (fun id -> (id, constant (T.sig_of g id)))
      b.Loop_ir.Loop_bundle.constants
  in
  match H.prepare ~allocation ~pipeline b with
  | Error rs ->
      Fmt.pr "%s: refused: %a@." name
        Fmt.(list ~sep:(any "; ") Mm.Refusal.pp)
        rs
  | Ok host -> (
      match
        H.Context.create host ~constants:(fun id -> List.assoc_opt id constants)
      with
      | Error s -> Fmt.pr "%s: %a@." name Mm.Stop.pp s
      | Ok cn ->
          let verdicts =
            List.init calls (fun call ->
                let inputs =
                  List.mapi
                    (fun i id -> (id, input ~salt:(call + i) i (T.sig_of g id)))
                    b.Loop_ir.Loop_bundle.inputs
                in
                let oracle = T.expected T.Fp32 g b ~constants ~inputs in
                match
                  H.Context.run cn ~inputs:(fun id -> List.assoc_opt id inputs)
                with
                | Error s -> Fmt.str "%a" Mm.Stop.pp s
                | Ok outs ->
                    if
                      List.for_all2
                        (fun id t ->
                          bits_of t = bits_of (Tensor_id.Map.find id oracle))
                        g.Graph.outputs outs
                    then "bitwise"
                    else "DIFFERS")
          in
          Fmt.pr "%s (%d invocations): %s@." name (H.invocations host)
            (String.concat ", " verdicts))

let%expect_test "Fp32_ordered: the table binding is the binary32 oracle" =
  List.iter
    (fun allocation ->
      let n s = Fmt.str "%s %s" s (Rt.Allocation.name allocation) in
      table_check ~allocation (n "chain") F.chain;
      table_check ~allocation (n "wide chain") T.wide_chain)
    Rt.Allocation.[ Reference; Scanned ];
  [%expect
    {|
    chain reference (3 invocations): bitwise, bitwise
    wide chain reference (3 invocations): bitwise, bitwise
    chain scanned (3 invocations): bitwise, bitwise
    wide chain scanned (3 invocations): bitwise, bitwise |}]

(* ---- Fp32_ordered with vectors -------------------------------------------------

   The policy planned for neon128, so loops vectorize on the exact binary32
   form: the image route against the selected-stage interpreter, the table
   route against the binary32 oracle, both allocators. *)

let%expect_test "Fp32_ordered vectors on this CPU" =
  List.iter
    (fun (name, g) ->
      check ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered ~constant:T.positive
        ("image " ^ name) g;
      List.iter
        (fun allocation ->
          table_check ~allocation ~pipeline:T.fp32_vector
            (Fmt.str "table %s %s" (Rt.Allocation.name allocation) name)
            g)
        Rt.Allocation.[ Reference; Scanned ])
    T.vector_fixtures;
  [%expect
    {|
    image relu then add over C=32 (2 invocations): bitwise as interpreted, within 1e-4 of binary64: true; bitwise as interpreted, within 1e-4 of binary64: true
    table reference relu then add over C=32 (2 invocations): bitwise, bitwise
    table scanned relu then add over C=32 (2 invocations): bitwise, bitwise
    image softmax over H, W=32 (1 invocations): bitwise as interpreted, within 1e-4 of binary64: true; bitwise as interpreted, within 1e-4 of binary64: true
    table reference softmax over H, W=32 (1 invocations): bitwise, bitwise
    table scanned softmax over H, W=32 (1 invocations): bitwise, bitwise
    image layer_norm over H, W=32 (1 invocations): bitwise as interpreted, within 1e-4 of binary64: true; bitwise as interpreted, within 1e-4 of binary64: true
    table reference layer_norm over H, W=32 (1 invocations): bitwise, bitwise
    table scanned layer_norm over H, W=32 (1 invocations): bitwise, bitwise
    image conv, batch norm, relu to C=32 (3 invocations): bitwise as interpreted, within 1e-4 of binary64: true; bitwise as interpreted, within 1e-4 of binary64: true
    table reference conv, batch norm, relu to C=32 (3 invocations): bitwise, bitwise
    table scanned conv, batch norm, relu to C=32 (3 invocations): bitwise, bitwise |}]

(* The binary32 transcendentals as NEON vector sequences, over the F32_math
   sample set, against the binary32 oracle. *)
let%expect_test "binary32 transcendentals as NEON vectors" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun allocation ->
          table_check ~allocation ~pipeline:T.fp32_vector_forced ~calls:1
            ~input:T.sample_input
            (Fmt.str "%s %s" name (Rt.Allocation.name allocation))
            g)
        Rt.Allocation.[ Reference; Scanned ])
    T.transcendental_fixtures;
  [%expect
    {|
    exp reference (1 invocations): bitwise
    exp scanned (1 invocations): bitwise
    log reference (1 invocations): bitwise
    log scanned (1 invocations): bitwise
    sin reference (1 invocations): bitwise
    sin scanned (1 invocations): bitwise
    cos reference (1 invocations): bitwise
    cos scanned (1 invocations): bitwise
    gelu (erf) reference (1 invocations): bitwise
    gelu (erf) scanned (1 invocations): bitwise |}]
