(* A symbolic input: an *abstract* tensor — everything an op/analysis needs to
   build and reason about an expression EXCEPT the bulk data. A tensor is
   "specified" symbolically by declaring its interface (like a function
   parameter's type). [shape] drives index bounds/broadcast; [fmt] decides
   what a Load emits (plain read / dequant / f16 decode); [quant] is
   compile-time metadata (per-channel scale enters an expr via the C index).
   A later [binding] id -> Tensor.packed supplies real data for evaluation /
   constant-folding. See .ai/native_symbolic_language.md §2.4. *)

type t = {
  id : int; (* binding key; the source a Load refers to *)
  name : string; (* the value name, for debug / pp *)
  shape : Vec6.shape; (* extents drive bounds & broadcast (static for now) *)
  fmt : Payload.packed_fmt;
      (* which storage format — existential over the GADT *)
  quant : Quant.t option; (* Some iff fmt is quantized *)
}

let counter = ref 0

let create ~name ~shape ~fmt ?quant () =
  incr counter;
  { id = !counter; name; shape; fmt; quant }

let pp fmt s = Format.pp_print_string fmt s.name
