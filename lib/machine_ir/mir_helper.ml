(* A helper's versioned typed descriptor: signature, memory effects and the
   failures it may raise. The interpreter binds a helper id to a deterministic
   model; an unknown helper is an admission refusal, never a host call. *)

module Effect = struct
  type t = Pure | Reads | Reads_writes

  let name = function
    | Pure -> "pure"
    | Reads -> "reads"
    | Reads_writes -> "reads_writes"
end

type t = {
  id : Mir_id.Helper.t;
  name : string;
  version : int;
  params : Mir_type.t list;
  results : Mir_type.t list;
  effects : Effect.t;
  failures : Mir_failure.t list;
      (** the failures it may raise: a call to a fallible helper propagates its
          record unchanged *)
}
