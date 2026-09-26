(* A loop counter: always an index. *)
include
  Core.Tagged_int.Make
    (struct
      let prefix = "i"
    end)
    ()
