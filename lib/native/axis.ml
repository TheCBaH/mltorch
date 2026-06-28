(* The six fixed tensor axes, in their canonical order N T D H W C (C innermost /
   channels-last). Every tensor in the engine is logically 6D over these; lower-
   rank tensors embed with size-1 axes. See .ai/native_tensor_design.md §1. *)

type t = N | T | D | H | W | C

let equal = ( = )

(* Canonical order; also the order [Vec6] lays components out (C last). *)
let all = [ N; T; D; H; W; C ]

(* Position in the canonical order, 0..5 — the index into a 6-component vector. *)
let to_int = function N -> 0 | T -> 1 | D -> 2 | H -> 3 | W -> 4 | C -> 5

let to_string = function
  | N -> "N"
  | T -> "T"
  | D -> "D"
  | H -> "H"
  | W -> "W"
  | C -> "C"

let pp fmt a = Format.pp_print_string fmt (to_string a)

let of_string = function
  | "N" -> Some N
  | "T" -> Some T
  | "D" -> Some D
  | "H" -> Some H
  | "W" -> Some W
  | "C" -> Some C
  | _ -> None

let jsont : t Jsont.t =
  Jsont.map ~kind:"axis"
    ~dec:(fun s ->
      match of_string s with
      | Some a -> a
      | None -> Jsont.Error.msgf Jsont.Meta.none "axis: unknown %S" s)
    ~enc:to_string Jsont.string
