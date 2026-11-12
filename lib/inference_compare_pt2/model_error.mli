val or_raise : what:string -> ('a, string) result -> 'a
(** The [Ok] value, or raises [Err.Exn.E] whose message is [what] followed by
    the builder's message. *)
