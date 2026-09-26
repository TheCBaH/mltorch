(* Per-key local storage: one flat [Float64] array. *)
include
  Core.Tagged_int.Make
    (struct
      let prefix = "a"
    end)
    ()
