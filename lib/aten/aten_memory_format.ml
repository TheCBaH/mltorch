(* OCaml encoding of c10::MemoryFormat (torch/headeronly/core/MemoryFormat.h). *)
type t = ChannelsLast | ChannelsLast3d | Contiguous | Preserve

(** This conversion table deliberately follows c10's numeric ABI; do not
    alphabetize its branches. *)
let to_int = function
  | Contiguous -> 0
  | Preserve -> 1
  | ChannelsLast -> 2
  | ChannelsLast3d -> 3

let of_int = function
  | 0 -> Some Contiguous
  | 1 -> Some Preserve
  | 2 -> Some ChannelsLast
  | 3 -> Some ChannelsLast3d
  | _ -> None
