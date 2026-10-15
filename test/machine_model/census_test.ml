module F = Native_test.Graph_fixtures
module M = Machine_model.Mir_model
module Census = Machine_ir.Mir_f64_census

(* The binary64 census of a bundle's generic Machine IR, summed over its
   invocations, under the binary64 reference pipeline and under the policies
   that exist today. The baseline the binary32 policy is measured against. *)

let bundle g =
  Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
    (Loop_ir.Loop_bundle.build g)

let sum cs role = List.fold_left (fun n c -> n + Census.count c role) 0 cs

let helpers cs =
  List.sort_uniq String.compare
    (List.concat_map (fun c -> List.map fst (Census.helpers c)) cs)

let show name pipeline g =
  match M.prepare ~pipeline (bundle (g ())) with
  | Error rs ->
      Fmt.pr "%s: refused: %a@." name Fmt.(list ~sep:(any "; ") M.Refusal.pp) rs
  | Ok m ->
      let cs = List.map Census.of_program (M.generic m) in
      Fmt.pr "%s: %s@." name
        (String.concat ", "
           (List.map
              (fun r -> Fmt.str "%s %d" (Census.Role.name r) (sum cs r))
              Census.Role.all
           @ [ "helpers [" ^ String.concat " " (helpers cs) ^ "]" ]))

let fixtures =
  [
    ("chain", F.chain);
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
  ]

let%expect_test "baseline census of the fixtures" =
  let planned numerics target =
    Ssa_backends.Pipeline.Planned { numerics; target }
  in
  List.iter
    (fun (name, g) ->
      show (name ^ " exact") Ssa_backends.Pipeline.Exact g;
      show
        (name ^ " simd_fp32_ordered neon128")
        (planned Ssa_ir.Ssa_numerics.Simd_fp32_ordered Ssa_ir.Ssa_target.neon128)
        g;
      show
        (name ^ " simd_fp32_ordered scalar")
        (planned Ssa_ir.Ssa_numerics.Simd_fp32_ordered Ssa_ir.Ssa_target.scalar)
        g)
    fixtures;
  [%expect
    {|
    chain exact: arithmetic 13, conversion 18, helper_call 0, memory_move 4, helpers []
    chain simd_fp32_ordered neon128: arithmetic 13, conversion 18, helper_call 0, memory_move 5, helpers []
    chain simd_fp32_ordered scalar: arithmetic 13, conversion 18, helper_call 0, memory_move 5, helpers []
    softmax exact: arithmetic 5, conversion 6, helper_call 2, memory_move 7, helpers [exp]
    softmax simd_fp32_ordered neon128: arithmetic 5, conversion 6, helper_call 2, memory_move 7, helpers [exp]
    softmax simd_fp32_ordered scalar: arithmetic 5, conversion 6, helper_call 2, memory_move 7, helpers [exp]
    layer_norm exact: arithmetic 16, conversion 8, helper_call 0, memory_move 14, helpers []
    layer_norm simd_fp32_ordered neon128: arithmetic 16, conversion 8, helper_call 0, memory_move 14, helpers []
    layer_norm simd_fp32_ordered scalar: arithmetic 16, conversion 8, helper_call 0, memory_move 14, helpers [] |}]
