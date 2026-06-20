(* OCaml encoding of c10::Layout (torch/headeronly/core/Layout.h). *)
type t =
  | Strided
  | Sparse
  | SparseCsr
  | Mkldnn
  | SparseCsc
  | SparseBsr
  | SparseBsc
  | Jagged

let to_int = function
  | Strided -> 0
  | Sparse -> 1
  | SparseCsr -> 2
  | Mkldnn -> 3
  | SparseCsc -> 4
  | SparseBsr -> 5
  | SparseBsc -> 6
  | Jagged -> 7

let of_int = function
  | 0 -> Some Strided
  | 1 -> Some Sparse
  | 2 -> Some SparseCsr
  | 3 -> Some Mkldnn
  | 4 -> Some SparseCsc
  | 5 -> Some SparseBsr
  | 6 -> Some SparseBsc
  | 7 -> Some Jagged
  | _ -> None
