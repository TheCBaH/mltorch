(** Deterministic, structural evidence for a Region program's output ownership.
    This is a diagnostic traversal: production execution uses
    [Region_execution.materialize], not this representation. *)

type entry = { key : Vec6.coord; outputs : Vec6.coord list }

type coverage = {
  total : int;
  keys : int;
  visited : int;
  duplicates : int;
  missing : int;
}

type t = {
  program : Region_program.t;
  entries : entry list;
  coverage : coverage;
}

type error =
  [ Region_partition.error
  | `Coverage of coverage
  | `Ownership of Vec6.coord * Vec6.coord ]

val collect : Region_program.t -> output_shape:Vec6.shape -> (t, error) Err.t
(** Enumerates canonical keys and their owned outputs, and independently counts
    visits in a bounded output-domain table. *)

val summarize : output_shape:Vec6.shape -> entry list -> coverage
(** Recomputes coverage from entries, independently of tensor stores. Exposed
    for trace mutation tests. Entries must contain in-bounds output coordinates.
*)

val pp_error : Format.formatter -> [< error ] -> unit
val pp : Format.formatter -> t -> unit
