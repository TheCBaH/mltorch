(* Generic monad building blocks shared across the project. *)

module State : sig
  (* The state-thread monad: a computation that reads and writes a state of type
     ['s] and yields a value of type ['a]. The representation is transparent so
     call sites can specialise the state type with a plain type alias, no
     boxing or functor required. *)
  type ('s, 'a) t = 's -> 'a * 's

  val return : 'a -> ('s, 'a) t
  val bind : ('s, 'a) t -> ('a -> ('s, 'b) t) -> ('s, 'b) t
  val ( let* ) : ('s, 'a) t -> ('a -> ('s, 'b) t) -> ('s, 'b) t
  val map : ('s, 'a) t -> ('a -> 'b) -> ('s, 'b) t
  val ( let+ ) : ('s, 'a) t -> ('a -> 'b) -> ('s, 'b) t

  (* Read the current state without modifying it. *)
  val get : ('s, 's) t
end
