(* The id spaces of one program. Each is a fresh type, so a value id cannot be
   passed where a region, buffer or revision is wanted. A buffer id is the
   tensor id it was lowered from, so a failure row names the same source a
   reference executor does. *)

(* A hash table keyed by an id, so a pass keeps its per-id state without
   unwrapping the id to an [int]. *)
module Tbl (I : sig
  type t

  val equal : t -> t -> bool
  val to_int : t -> int
end) =
Hashtbl.Make (struct
  type t = I.t

  let equal = I.equal
  let hash v = Hashtbl.hash (I.to_int v)
end)

module Block =
  Core.Tagged_int.Make
    (struct
      let prefix = "bb"
    end)
    ()

module Buffer = struct
  include
    Core.Tagged_int.Make
      (struct
        let prefix = "b"
      end)
      ()

  module Tbl = Tbl (struct
    type nonrec t = t

    let equal = equal
    let to_int = to_int
  end)
end

module Stmt =
  Core.Tagged_int.Make
    (struct
      let prefix = "s"
    end)
    ()

module Region = struct
  include
    Core.Tagged_int.Make
      (struct
        let prefix = "r"
      end)
      ()

  module Tbl = Tbl (struct
    type nonrec t = t

    let equal = equal
    let to_int = to_int
  end)
end

module Revision =
  Core.Tagged_int.Make
    (struct
      let prefix = "rev"
    end)
    ()

module Value = struct
  include
    Core.Tagged_int.Make
      (struct
        let prefix = "v"
      end)
      ()

  module Tbl = Tbl (struct
    type nonrec t = t

    let equal = equal
    let to_int = to_int
  end)
end
