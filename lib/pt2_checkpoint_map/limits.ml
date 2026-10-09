type t = {
  max_allocation_bytes : int64;
  max_captures : int;
  max_checkpoint_files : int;
  max_document_bytes : int;
  max_inline_bytes : int;
  max_prepared_bytes : int64;
  max_rank : int;
  max_source_bytes : int64;
  max_tensor_bytes : int64;
}

(* js_of_ocaml's measured [Sys.max_string_length] less a margin, the same
   ceiling as [Pt2_archive.max_file_bytes]: a platform limit, not a model's. *)
let platform_bytes = 0x7000_0000L

let default =
  {
    max_allocation_bytes = platform_bytes;
    max_captures = 100_000;
    max_checkpoint_files = 64;
    max_document_bytes = 0x1000000;
    max_inline_bytes = 64 * 1024;
    max_prepared_bytes = Int64.shift_left 1L 33;
    max_rank = 8;
    max_source_bytes = platform_bytes;
    max_tensor_bytes = Int64.shift_left 1L 40;
  }

let pp ppf t =
  Fmt.pf ppf
    "allocation<=%Ld captures<=%d files<=%d document<=%d inline<=%d \
     prepared<=%Ld rank<=%d source<=%Ld tensor<=%Ld"
    t.max_allocation_bytes t.max_captures t.max_checkpoint_files
    t.max_document_bytes t.max_inline_bytes t.max_prepared_bytes t.max_rank
    t.max_source_bytes t.max_tensor_bytes

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

let pp_which ppf w =
  Fmt.string ppf
    (match w with
    | Allocation_bytes -> "buffer size"
    | Captures -> "capture count"
    | Checkpoint_files -> "checkpoint file count"
    | Document_bytes -> "document size"
    | Inline_bytes -> "inline value size"
    | Prepared_bytes -> "prepared bytes"
    | Rank -> "tensor rank"
    | Source_bytes -> "source file size"
    | Tensor_bytes -> "tensor byte size")
