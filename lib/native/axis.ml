(* The six fixed tensor axes, in their canonical order N T D H W C (C innermost /
   channels-last). Every tensor in the engine is logically 6D over these; lower-
   rank tensors embed with size-1 axes. See .ai/native_tensor_design.md §1.

   A MANIFEST alias of [Expr.Axis.t], not a conversion. The expression language
   owns its own frame (it must: it cannot depend on [native]), and the two have
   to agree. An equation is checked by the compiler — adding, dropping or
   reordering a constructor on either side fails to build — where a conversion
   table can silently map D to H and every test still passes, because both
   directions use the same wrong table. It also costs nothing on [Vec6.get]'s
   innermost path, which is called once per axis per output coordinate.

   [jsont], [of_string] and [to_string] stay HERE rather than moving: they are a
   wire format, and pulling them into [lib/expr] would cost that library a
   [jsont] dependency it has no other use for. *)

type t = Expr.Axis.t = N | T | D | H | W | C

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
