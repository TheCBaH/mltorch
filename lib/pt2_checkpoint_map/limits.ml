type t = {
  max_captures : int;
  max_checkpoint_files : int;
  max_document_bytes : int;
  max_inline_bytes : int;
  max_rank : int;
  max_tensor_bytes : int64;
}

let default =
  {
    max_captures = 100_000;
    max_checkpoint_files = 64;
    max_document_bytes = 0x1000000;
    max_inline_bytes = 64 * 1024;
    max_rank = 8;
    max_tensor_bytes = Int64.shift_left 1L 40;
  }

let pp ppf t =
  Fmt.pf ppf
    "captures<=%d files<=%d document<=%d inline<=%d rank<=%d tensor<=%Ld"
    t.max_captures t.max_checkpoint_files t.max_document_bytes
    t.max_inline_bytes t.max_rank t.max_tensor_bytes

type which =
  | Captures
  | Checkpoint_files
  | Document_bytes
  | Inline_bytes
  | Rank
  | Tensor_bytes

let pp_which ppf w =
  Fmt.string ppf
    (match w with
    | Captures -> "capture count"
    | Checkpoint_files -> "checkpoint file count"
    | Document_bytes -> "document size"
    | Inline_bytes -> "inline value size"
    | Rank -> "tensor rank"
    | Tensor_bytes -> "tensor byte size")
