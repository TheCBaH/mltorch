(* A deterministic model of a helper: what the interpreter runs for a call.
   It sees the arguments and the synthetic memory and either returns results
   of the descriptor's types or raises one of its declared failures with a
   payload. No host function pointer is ever called for a helper. *)

open Machine_ir

type result =
  | Fails of Mir_failure.t * Mir_const.t list
  | Returns of Mir_datum.t list

type t = {
  name : string;
  version : int;
  run : Mir_memory.t -> Mir_datum.t list -> result;
}

let find models (h : Mir_helper.t) =
  List.find_opt
    (fun m ->
      String.equal m.name h.Mir_helper.name && m.version = h.Mir_helper.version)
    models
