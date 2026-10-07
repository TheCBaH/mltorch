(* A memory object: a byte range with an alignment and the source of its bytes.
   Views name windows of it; two views of one region may overlap, and distinct
   view or tensor ids are no evidence that their bytes are disjoint. *)

type init =
  | Bound  (** supplied by the caller at each invocation *)
  | Constant of string
      (** the program's own read-only bytes, exactly [size] of them *)
  | Uninitialized
      (** program-owned storage that starts every invocation with no defined
          byte: runtime state, scratch *)

type t = { id : Mir_id.Region.t; size : int64; align : int64; init : init }

let init_name = function
  | Bound -> "bound"
  | Constant _ -> "constant"
  | Uninitialized -> "uninit"
