(* A tensor's identity, and [Next], the counter that hands the next one out.
   [of_int] is the builder's allocation; see [Core.Tagged_int]. *)
include Core.Tagged_int.S

val jsont : t Jsont.t
