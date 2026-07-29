(* See brand.mli. The representation is [unit]: two brands are distinguished by
   the rigid type variables their existentials introduce, never at runtime, so
   there is nothing to store and no counter to keep. *)

type 'v t = unit
type packed = Pack : 'v t -> packed

let fresh () = Pack ()
