(* Phantom role markers for index expressions. [Position.t] is known >= 0 and is
   what a [Load] coordinate accepts; [Delta.t] is a signed affine value.
   Uninhabited: they exist only to be a type parameter, never a runtime value.

   [Dim.index] and [Dim.delta] (in this library) are manifest aliases of these,
   and [Expr.Role] re-exports them,
   rather than converting at a boundary. They have to: [Symbolic]'s associated
   type is ['role index], so the two role vocabularies must be the SAME types
   for the [SEMANTICS] instantiation to typecheck at all. *)

module Position : sig
  type t
end

module Delta : sig
  type t
end
