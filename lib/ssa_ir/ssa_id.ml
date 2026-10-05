(* The id spaces of one program. Each is a fresh type, so a value id cannot be
   passed where a region, buffer or revision is wanted. A buffer id is the
   tensor id it was lowered from, so a failure row names the same source a
   reference executor does. *)

module Buffer =
  Core.Tagged_int.Make
    (struct
      let prefix = "b"
    end)
    ()

module Region =
  Core.Tagged_int.Make
    (struct
      let prefix = "r"
    end)
    ()

module Revision =
  Core.Tagged_int.Make
    (struct
      let prefix = "rev"
    end)
    ()

module Value =
  Core.Tagged_int.Make
    (struct
      let prefix = "v"
    end)
    ()
