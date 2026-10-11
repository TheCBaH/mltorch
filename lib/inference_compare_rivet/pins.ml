(* The components every Rivet route is made of, with the revisions this build
   pins. *)

let rivet =
  {
    Compiled_identity.name = "Rivet assembler, image planner and native loader";
    pin = "vendored/rivet@4bd01a7447241f794a55124cc6f67d63b603f8e9";
  }

let schedule =
  {
    Compiled_identity.name = "Loop IR bundle schedule and storage plan";
    pin = "this repository";
  }

let selection =
  {
    Compiled_identity.name =
      "Machine IR selection, allocation and publication (AArch64)";
    pin = "this repository";
  }
