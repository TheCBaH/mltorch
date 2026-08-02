(* Full entry point: every ungated section. Built natively as the golden that
   js/jsoo's `(modes js)` build of this same source is diffed against.

   Melange cannot reach sections 3, 4 and 5 (Bigarray, Jsont_bytesrw), which is
   why there is a second entry point, subset_probe.ml, rather than a flag here: a
   two-section run and a five-section golden can never diff clean.

   The tier-2 work -- opening a real .pt2 and running inference on it -- lives in
   a THIRD entry point, pt2_probe.ml, because it needs downloaded weights that
   this one must not require. *)

let () =
  let model_json =
    match Sys.argv with
    | [| _; path |] -> path
    | _ ->
        prerr_endline "usage: native_probe <model.json>";
        exit 2
  in
  Probe_walk_core.run ();
  Probe_core.run ();
  Probe_tensor_json.run ();
  Probe_native.run ();
  Probe_model_json.run model_json
