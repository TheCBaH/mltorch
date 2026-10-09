(** Ceilings checked before the work or allocation they guard. They narrow what
    a platform can hold, never raise it: the defaults are far below the
    js_of_ocaml string and bigstring ceilings, and a host may only lower them.
    The values are chosen from the formats (a rank-six Native shape, the
    producer's 64 KiB inline cap), not from the models measured, which are
    recorded as observations in the design. *)

type t = {
  max_allocation_bytes : int64;
      (** One buffer this library allocates (a widened or generated value). At
          most the js_of_ocaml ceiling, so narrowing to [int] is in range on
          both backends. *)
  max_captures : int;
  max_checkpoint_files : int;
  max_document_bytes : int;
  max_inline_bytes : int;
  max_prepared_bytes : int64;
      (** Mapped sources plus every buffer allocated, together. *)
  max_rank : int;
  max_source_bytes : int64;  (** One source file. *)
  max_tensor_bytes : int64;
}

val default : t
val pp : t Fmt.t

type which =
  | Allocation_bytes
  | Captures
  | Checkpoint_files
  | Document_bytes
  | Inline_bytes
  | Prepared_bytes
  | Rank
  | Source_bytes
  | Tensor_bytes
      (** A ceiling by name, alphabetically; what an over-limit error reports.
      *)

val pp_which : which Fmt.t
