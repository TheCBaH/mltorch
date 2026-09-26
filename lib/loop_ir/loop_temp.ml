(* A scalar temporary, assigned and never aliased. *)
include
  Core.Tagged_int.Make
    (struct
      let prefix = "t"
    end)
    ()
