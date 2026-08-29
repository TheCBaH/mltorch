(* The all-sample top-5 inference report: open a .pt2, run every [.pt] sample in
   a release's [images/] directory, and compare each local top-5 against the
   [results.json] shipped alongside it.

   Both runners share this. What differs between them is one function -- how a
   loaded tensor becomes ranked class probabilities -- so that is what [run]
   takes as [~infer]. Everything the evaluator does NOT determine (argument
   parsing, label lookup, sample discovery, report formatting, the strict
   verdict) lives here exactly once.

   The evaluator's own error type is a parameter rather than a fixed union: the
   ATen path fails with [Interp.error] and the pure path with a union of
   [Native_interp.error] and [Native_predict.error], and neither library should
   have to know the other exists. *)

type mode =
  | Cram
      (** probabilities dropped when the ranking matches. They differ in their
          low-order digits across systems, so no fixed rounding is exact-match
          safe; on a match the ranking is the whole assertion. *)
  | Natural  (** probabilities at natural precision, for reading *)

(* [mode] is a variant and stays here; the records below get their own modules
   per the record-namespace convention. *)

module Paths : sig
  type t = {
    pt2 : string;
    images_dir : string;
    synsets : string;
    metadata : string;
    results : string;
    inputs : string option;
    expected : string option;
    outputs : string option;
  }
end

module Options : sig
  type t = {
    mode : mode;
    strict : bool;
        (** Fail the run when any sample's top-5 disagrees with its reference.
            Off by default, so the interpreter cram tests -- whose goldens are
            what make a mismatch visible there -- keep their exit-0 contract.
            `make jsoo.pt2.run` turns it on, which is what makes that command a
            gate rather than a report. *)
  }
end

module Mismatch : sig
  type t = { sample : string; local : int list; reference : int list }
end

module Prediction : sig
  type t = {
    rank : int;
    class_index : int;
    label : string;
    probability : float;
  }
  (** One [results.json] entry. All four fields are load-bearing: [class_index]
      is what the ranking comparison comes down to, and the other three are what
      the reference block prints when it disagrees. *)
end

type 'eval error =
  [ `Eval of 'eval
  | `Expected_decode of string
  | `Mismatch of Mismatch.t  (** strict mode only *)
  | `No_reference of string
  | Pt2_archive.error
  | `Results_decode of string ]

val pp_error :
  (Format.formatter -> 'eval -> unit) -> Format.formatter -> 'eval error -> unit

val parse_argv : string array -> (Paths.t * Options.t, string) result
(** Exactly four positional paths -- model, preprocessed tensor map, top-5
    reference, and full-output tensor map -- then zero or more flags. *)

val compare_ranking :
  sample:string ->
  local:(int * float) list ->
  reference:int list ->
  Mismatch.t option
(** [None] when the local class indices equal the reference sequence. *)

val strict_verdict : Mismatch.t list -> (unit, Mismatch.t) result
(** The first mismatch in sample order, so a strict failure names the earliest
    sample rather than whichever one happened to be last. *)

val report :
  label:(int -> string) ->
  reference:(string -> Prediction.t list option) ->
  samples:string list ->
  infer:(string -> ((int * float) list, 'eval error) Err.t) ->
  Options.t ->
  (unit, 'eval error) Err.t
(** The per-sample loop, with every host effect injected -- no filesystem, no
    archive, no clock. That is what lets the strict path be tested with
    synthetic samples: a regression that stops calling {!strict_verdict} is
    invisible to a test of {!strict_verdict} alone, and equally invisible to a
    fixture run whose references all match.

    [infer] returns an ALREADY-CLASSIFIED error, and [report] propagates it
    untouched. It must not re-wrap: its caller's closure performs the per-sample
    [Pt2_archive.load_pt] too, and a blanket [`Eval] here would report a corrupt
    [.pt] file as an evaluator failure.

    [samples] are [results.json]-style keys ("images/<file>.pt"), ascending.
    [reference] returning [None] is [`No_reference]. *)

val run :
  now:(unit -> float) ->
  infer:(Pt2_archive.t -> Pt2_tensor.t -> ((int * float) list, 'eval) Err.t) ->
  Paths.t ->
  Options.t ->
  (unit, 'eval error) Err.t
(** The host shell over {!report}: opens the archive and the producer's tensor
    maps, checks their lexical key agreement, and classifies evaluator failures
    under [`Eval].

    [~now] is the clock. It is a parameter rather than [Unix.gettimeofday]
    because [unix] must not enter the js_of_ocaml closure; the ATen runner
    passes [Unix.gettimeofday] and the pure one passes [Sys.time]. Timings go to
    stderr, never stdout, since they are non-deterministic and stdout is
    compared exactly by the cram tests. *)
