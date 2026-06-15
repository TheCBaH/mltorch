(* OCaml encoding of c10::MemoryFormat (torch/headeronly/core/MemoryFormat.h). *)
type t = Contiguous | Preserve | ChannelsLast | ChannelsLast3d

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
