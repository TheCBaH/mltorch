open Ssa_bridge

(* The sharing and structure gates over generated placed graphs: every kernel is
   run stored (every value materialized) and fused (as far as the planner
   admits); each must agree with the reference through the SSA lowering, and
   the two forms must give the same output bits, as a placement never changes
   a value. *)

let seeds = max 1 (Fuzz.seeds / 20)
let lengths = [ 1; 2; 3; 5; 8 ]

let verdict name plan case =
  match Ssa_check.run plan ~bind:(Placed_fuzz.bind case) with
  | Ssa_check.Agree -> ()
  | v ->
      Fmt.failwith "seed %Ld, %s: %a" case.Placed_fuzz.seed name
        Ssa_check.pp_verdict v

let%expect_test
    "stored and fused placements agree with the reference and each other" =
  let fused = ref 0 in
  List.iter
    (fun length ->
      for s = 0 to seeds - 1 do
        let case = Placed_fuzz.generate ~seed:(Int64.of_int s) ~length in
        let stored = Fusion_plan.default case.Placed_fuzz.kernel in
        let placed, _ = Fusion_plan.plan case.Placed_fuzz.kernel in
        verdict "stored" stored case;
        verdict "fused" placed case;
        let run plan =
          Err.map
            (Tensor_id.Map.filter (fun id _ ->
                 List.exists (Tensor_id.equal id) case.Placed_fuzz.outputs))
            (Kernel_eval.run_plan plan ~bind:(Placed_fuzz.bind case))
        in
        (match
           Ssa_check.compare_kernel ~reference:(run stored) ~actual:(run placed)
         with
        | Ssa_check.Agree -> ()
        | v ->
            Fmt.failwith "seed %Ld, length %d: stored against fused: %a"
              case.Placed_fuzz.seed length Ssa_check.pp_verdict v);
        if not (placed == stored) then incr fused
      done)
    lengths;
  Fmt.pr "graphs: %d, with a fused value: %s@."
    (seeds * List.length lengths)
    (if !fused > 0 then "some" else "none");
  [%expect {| graphs: 500, with a fused value: some |}]

let%expect_test "the comparison fails on graphs that differ" =
  let a = Placed_fuzz.generate ~seed:1L ~length:5
  and b = Placed_fuzz.generate ~seed:2L ~length:5 in
  let run c =
    Err.map
      (Tensor_id.Map.filter (fun id _ ->
           List.exists (Tensor_id.equal id) c.Placed_fuzz.outputs))
      (Kernel_eval.run_plan
         (Fusion_plan.default c.Placed_fuzz.kernel)
         ~bind:(Placed_fuzz.bind c))
  in
  Fmt.pr "%a@." Ssa_check.pp_verdict
    (Ssa_check.compare_kernel ~reference:(run a) ~actual:(run b));
  [%expect {| DISAGREE: ssa produced an extra output t1 |}]
