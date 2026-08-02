(* Subset entry point: exactly the sections melange can reach today. Built
   natively as the golden for js/melange's emit of this same source, so that
   comparison is between two runs of one program rather than between a slice of
   one golden and the whole of another. *)

let () =
  Probe_walk_core.run ();
  Probe_core.run ()
