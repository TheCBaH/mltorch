(* The one minter of graph-version tags.

   A tagged id is only as good as the impossibility of forging its tag, and
   [lib/native/dune] is [(wrapped false)], so there are no library-private
   modules to hide a constructor in — every entry point here is public API. The
   protection therefore has to come from the shape of the signature rather than
   from visibility.

   [fresh] returns a PACKED brand, so a caller cannot pick which version they
   are minting: unpacking [Pack b] binds [b] at a rigid type variable that
   unifies with nothing else. That is the same device [Rewrite.origin] uses for
   its [origin], and it is why [Universe.create] can take a brand as an ordinary
   argument without becoming a forgery: to build a universe at some EXISTING
   snapshot's version you would need that snapshot's brand, and nothing hands it
   out. Minting your own and building universes under it is harmless, because
   its tag corresponds to no other value in the program.

   The brand carries no information — distinctness is entirely a type-level
   property of the existential — so [Universe.create] takes one and discards it.
   The parameter exists to force possession, not to be read. *)

type 'v t
type packed = Pack : 'v t -> packed

val fresh : unit -> packed
