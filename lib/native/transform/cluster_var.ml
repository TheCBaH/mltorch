(* See cluster_var.mli. *)

include
  Core.Tagged_int.Make
    (struct
      let prefix = "v"
    end)
    ()
