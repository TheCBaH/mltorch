(* The scalar dimensional roles live in [Core.Dim], below the expression
   language, so [Expr] and this library share them; see there. What stays here
   is the wire codec, since [core] does not depend on [Jsont]. *)

include Core.Dim

let checked_jsont ~kind ~min ~(make : int -> 'r t) : 'r t Jsont.t =
  Jsont.map ~kind
    ~dec:(fun n ->
      if n < min then
        Jsont.Error.msgf Jsont.Meta.none "%s: must be >= %d, got %d" kind min n
      else make n)
    ~enc:to_int Jsont.int

let extent_jsont : extent t Jsont.t =
  checked_jsont ~kind:"extent" ~min:1 ~make:extent

let index_jsont : index t Jsont.t =
  checked_jsont ~kind:"index" ~min:0 ~make:index

let fence_jsont : fence t Jsont.t =
  checked_jsont ~kind:"fence" ~min:0 ~make:fence
