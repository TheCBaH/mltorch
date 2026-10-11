(** Ordinary-graph runners over the synthetic dialect: exact wrapping int64
    elementwise arithmetic on a {!Graph_desc.t} whose operators are named [add],
    [sub], [mul], [neg], [id], [dup] (one input, two results), [addc:K],
    [mulc:K] and [assert_nonneg] (a metadata assertion with no results).
    Anything else is refused at preparation.

    Each runner is configurable with the faults the coordinator must survive: a
    per-node additive deviation, a corrupted import, a refused operator, a
    failing node, output-storage reuse across runs (an unexported earlier result
    changes under the caller) and a simulated asynchronous completion. Counters
    expose every handle so lifecycle tests can prove cleanup. *)

open Inference_compare_schema

type config = {
  name : string;
  offsets : (string * int64) list;
      (** Node local id -> added to every element of each of its results. *)
  corrupt_imports : (int * int64) list;
      (** Zero-based import ordinal -> added to every element on that import, so
          the handle read back differs from the value sent. *)
  refuse_ops : string list;  (** Operator names refused at preparation. *)
  fail_nodes : string list;  (** Node local ids whose execution fails. *)
  fail_imports : int list;  (** Zero-based import ordinals that fail. *)
  reuse_output_storage : bool;
  async_polls : int;  (** Completion polls performed before [run] returns. *)
}

val default_config : string -> config

type counters = {
  mutable prepared_open : int;
  mutable values_live : int;
  mutable prepares : int;
  mutable imports : int;
  mutable exports : int;
  mutable runs : int;
  mutable polls : int;
  mutable executed_nodes : string list;  (** Newest first. *)
  mutable observed_inputs : (string * int64 list) list list;
      (** Per run, newest first: (input local id, elements). *)
}

type error =
  | Closed
  | Import_mismatch of string
  | Input_inventory of string
  | Node_failed of string
  | Refused_operator of string
  | Resource_exhausted
  | Unsupported_graph of string
  | Value_released

val pp_error : Format.formatter -> error -> unit

type t =
  (module Inference_compare_core.Runner.GRAPH_RUNNER
     with type Graph.t = Graph_desc.t
      and type Error.t = error)

val create : config -> t * counters
