(* The description builders report a failure as a message. At the points that
   describe a body already admitted, a message is a broken invariant of the
   body's names, so it leaves as one [Err.Exn.E] naming where it arose. *)

let or_raise ~what r =
  Err.or_raise
    ~pp_error:(fun ppf m -> Fmt.pf ppf "%s: %s" what m)
    (Err.import Fun.id r)
