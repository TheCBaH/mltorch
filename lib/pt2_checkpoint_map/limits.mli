(** Ceilings checked before the work or allocation they guard. They narrow what
    a platform can hold, never raise it: the defaults are far below the
    js_of_ocaml string and bigstring ceilings, and a host may only lower them.
    The values are chosen from the formats (a rank-six Native shape, the
    producer's 64 KiB inline cap), not from the models measured, which are
    recorded as observations in the design. *)

type t = {
  max_captures : int;
  max_checkpoint_files : int;
  max_document_bytes : int;
  max_inline_bytes : int;
  max_rank : int;
  max_tensor_bytes : int64;
}

val default : t
val pp : t Fmt.t

type which =
  | Captures
  | Checkpoint_files
  | Document_bytes
  | Inline_bytes
  | Rank
  | Tensor_bytes
      (** A ceiling by name, alphabetically; what an over-limit error reports.
      *)

val pp_which : which Fmt.t
