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

(* The revision of a helper's descriptor: a model binds to a helper by name
   and this, so a changed descriptor is a different helper. *)
module Version =
  Core.Tagged_int.Make
    (struct
      let prefix = "version"
    end)
    ()

type t = {
  id : Mir_id.Helper.t;
  name : string;
  version : Version.t;
  params : Mir_type.t list;
  results : Mir_type.t list;
  effects : Effect.t;
  failures : Mir_failure.t list;
      (** the failures it may raise: a call to a fallible helper propagates its
          record unchanged *)
}
