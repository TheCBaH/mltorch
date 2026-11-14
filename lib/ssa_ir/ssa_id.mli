(** The id spaces of one program. Each is a fresh type, so a value id cannot be
    passed where a region, buffer or revision is wanted. Each space that a pass
    keeps per-id state for also has a hash table keyed by the id. *)

module Block : Core.Tagged_int.S

module Buffer : sig
  include Core.Tagged_int.S
  module Tbl : Hashtbl.S with type key = t
end

module Region : sig
  include Core.Tagged_int.S
  module Tbl : Hashtbl.S with type key = t
end

module Revision : Core.Tagged_int.S

module Stmt : Core.Tagged_int.S
(** A statement's position in its region's body. *)

module Value : sig
  include Core.Tagged_int.S
  module Tbl : Hashtbl.S with type key = t
end
