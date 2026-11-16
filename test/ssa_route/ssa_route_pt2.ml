(* The route gate over one real model's bundle: the canonical graph, the
   archive's own constants, deterministic inputs. Single process and no
   timing, host or hash in the output, so the printed table is the committed
   golden. Exit 1 when any comparison disagrees or any invocation is refused.

   Usage: ssa_route_pt2 MODEL.pt2 *)

module G = Graph_ir
module Gate = Ssa_route_test.Route_gate

let () =
  let b, constants = Model_load.load Sys.argv.(1) in
  let tally =
    Gate.compare_bundle b
      ~constants:(G.Tensor_id.Map.bindings constants)
      ~inputs:(Gate.default_inputs b)
  in
  Fmt.pr "%s: %a" (Filename.basename Sys.argv.(1)) Gate.Tally.pp tally;
  let sites, agreeing = Gate.emitter_agreement b in
  Fmt.pr "  failure_sites=%d same emit verdict: %s@." sites
    (String.concat ", " (List.map (fun (e, n) -> Fmt.str "%s %d" e n) agreeing));
  let clean =
    Gate.Tally.clean tally
    && tally.Gate.Tally.converted = tally.Gate.Tally.invocations
  in
  Fmt.pr "gate: %s@." (if clean then "pass" else "fail");
  if not clean then exit 1
