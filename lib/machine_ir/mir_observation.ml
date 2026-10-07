(* What one execution route observed, normalized for comparison: its status,
   the defined cells of each public output, and the logical events. Primitive
   access and instruction counts are diagnostics of a route and never appear
   here. *)

(* A language failure as a route reports it: static identity, payload (typed
   exact bits), invocation and, when the route stored a record, its raw site. *)
module Row = struct
  type t = {
    failure : Mir_failure.t;
    payload : Mir_const.t list;
    invocation : int32 option;
    site : Mir_id.Site.t option;
  }
end

(* A compiler invariant or execution defect: invalid IR, an undefined or
   out-of-domain operation, a bad address, uninitialized bytes. *)
module Defect = struct
  type t =
    | Bad_access  (** outside a region, misaligned, or not permitted *)
    | Domain  (** a partial operation outside its defined domain *)
    | Invalid_program
    | Sentinel_site
    | Uninitialized  (** a read of a byte or flag never defined *)

  let name = function
    | Bad_access -> "bad_access"
    | Domain -> "domain"
    | Invalid_program -> "invalid_program"
    | Sentinel_site -> "sentinel_site"
    | Uninitialized -> "uninitialized"
end

module Status = struct
  type t =
    | Defect of Defect.t
    | Failure of Row.t
    | Fuel_exhausted  (** test resources, never a scan-budget failure *)
    | Success
    | Unsupported of string  (** a capability the route does not have *)
end

(* One output's cells, in element order; [None] is a cell no route defined. *)
module Output = struct
  type t = { source : Expr.Source.t; cells : Mir_const.t option array }
end

type t = {
  status : Status.t;
  outputs : Output.t list;
  events : (Mir_event.t * int64) list;  (** every event, zero counts kept *)
}
