(* Reducer identities must be fresh during construction AND rewriting. The
     supply is threaded, not global: [run] starts a fresh identity namespace,
     and [run_from] CONTINUES an existing one, for when several computations
     have to share a namespace rather than each restarting at ordinal 0. Two
     computations run from [initial] deliberately reuse ordinals -- identity is
     meaningful only within one expression, and composing across that boundary
     is what [Rewrite.freshen] is for. *)
type state = int
type 'a t = state -> 'a * state

exception Supply_exhausted
(** Raised by [fresh_reduce] when the ordinal would wrap. A wrapped ordinal
    would re-mint a live identity and break scope, alpha-equivalence, comparison
    and hashing at once, so this is checked rather than assumed -- a
    trusted-capacity boundary, named so it is visible. *)

let initial : state = 0
let return x s = (x, s)

let bind m f s =
  let x, s = m s in
  f x s

let map f m s =
  let x, s = m s in
  (f x, s)

let run_from s m = m s
let run m = fst (m initial)
let fresh s = if s = Stdlib.max_int then raise Supply_exhausted else (s, s + 1)
let fresh_reduce = fresh
let fresh_local = fresh

module Syntax = struct
  let ( let* ) = bind
  let ( let+ ) m f = map f m
end

(* Allocates the variable, hands its index expression to the body, and threads
     the updated supply through. Going through here rather than building a
     [Reduction.t] by hand is what makes scope correct by construction: the
     callback cannot name a variable that is not this reduction's, and the
     variable cannot escape into a sibling. *)
let reduction ~kind ~lo ~hi body s =
  let v, s = fresh_reduce s in
  let b, s = body (Index.reduce v) s in
  (Value.reduce { Reduction.kind; var = v; lo; hi; body = b }, s)
