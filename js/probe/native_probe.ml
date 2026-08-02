(* Full entry point: every section. Built natively as the golden that
   js/jsoo's `(modes js)` build of this same source is diffed against.

   Melange cannot reach sections 3 and 4 (Bigarray, Jsont_bytesrw), which is why
   there is a second entry point, subset_probe.ml, rather than a flag here: a
   two-section run and a four-section golden can never diff clean. *)

let () =
  Probe_walk_core.run ();
  Probe_core.run ();
  Probe_tensor_json.run ();
  Probe_native.run ()
