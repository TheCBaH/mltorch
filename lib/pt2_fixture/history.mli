(** What a static-history decode artifact covers.

    The producer exports a decode graph per history length: its K/V inputs have
    a fixed history axis, and a published case is one step at that length. A run
    or replay of it establishes nothing about the lengths in between, so these
    pure checks let a caller refuse a generation step it has no accepted
    component for -- and refuse to chain a prefill to a decode whose shapes do
    not meet -- rather than pad, truncate or reinterpret. *)

type t = {
  attention_length : int;
  history : int;  (** The one history length the graph was exported at. *)
  maximum_input_history : int;
  state_inputs : string list;  (** [past_*], in call order. *)
  state_outputs : string list;  (** [present_*], in tuple order. *)
}

type fault =
  | Capacity of { requested : int; maximum : int }
  | Chain_mismatch of { what : string; prefill : string; decode : string }
  | Uncovered of { requested : int; covered : int }

val pp_fault : Format.formatter -> fault -> unit

val of_contract_string : string -> (t option, string) result
(** [Some] only for a contract whose [variant] is a static history with a
    [state] block; [None] for every other contract (prefill, forward, dynamic).
    [Error] is a malformed JSON document. *)

val scope : t -> string
(** The sentence a report carries about what the artifact does not establish. *)

val history_of_shape : int64 list -> int option
(** The history axis of a [batch; heads; history; head_dim] K/V shape. *)

val check_feed : t -> requested:int -> (unit, fault) result
(** A feed of [requested] steps of history is covered only when it equals the
    artifact's own history; one beyond the capacity is reported as such. *)

val chain :
  prefill:(string * int64 list) list ->
  decode:t ->
  decode_inputs:(string * int64 list) list ->
  (unit, fault) result
(** Whether a prefill's [present_*] outputs (name, shape) can feed a decode
    artifact's [past_*] inputs: the same names in the same order, the same
    batch, heads and head dimension, and a prefill length equal to the decode's
    history. *)

val chain_tensors :
  prefill:Contract.Tensor_spec.t list ->
  decode:t ->
  decode_inputs:Contract.Tensor_spec.t list ->
  (unit, fault) result
(** [chain] plus exact dtype compatibility for every state binding. *)
