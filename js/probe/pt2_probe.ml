(* Tier-2 entry point: reading a real .pt2 and running inference on it. Built
   natively here as the golden that js/jsoo's `(modes js)` build of this same
   source is diffed against.

   Separate from native_probe because the two tiers need different inputs: this
   one needs downloaded weights (~12MB for mobilenet_v3_small), so it is gated,
   while native_probe needs only a committed model.json and runs on every push.
   A gated run and an ungated golden can never diff clean against each other. *)

let () =
  match Sys.argv with
  | [| _; pt2; input |] -> Probe_pt2.run ~pt2 ~input
  | _ ->
      prerr_endline "usage: pt2_probe <model.pt2> <input.pt>";
      exit 2
