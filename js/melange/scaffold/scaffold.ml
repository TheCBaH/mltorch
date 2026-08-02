(* The fallback target's entry point: it does nothing except exist, so that
   depending on it forces the three vendored/shimmed libraries to compile and
   link through dune's ordinary melange machinery.

   That is the point -- @melange-scaffold has to keep working even if the jsont
   patch stops applying and the real probe cannot be built, and it must do so
   without naming .cmj paths under .objs/melange, which are dune's internal
   layout rather than a supported target. *)

let () = print_endline "melange scaffold ok"
