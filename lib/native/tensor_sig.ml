(* A symbolic input: an *abstract* tensor — everything an op/analysis needs to
   build and reason about an expression EXCEPT the bulk data. A tensor is
   "specified" symbolically by declaring its interface (like a function
   parameter's type). [shape] drives index bounds/broadcast; [fmt] decides
   what a Load emits (plain read / dequant / f16 decode); [quant] is
   compile-time metadata (per-channel scale enters an expr via the C index).
   A later [binding] id -> Tensor.packed supplies real data for evaluation /
   constant-folding. See .ai/native_symbolic_language.md §2.4. *)

type t = {
  id : Tensor_id.t; (* binding key; the source a Load refers to *)
  shape : Vec6.shape; (* extents drive bounds & broadcast (static for now) *)
  fmt : Payload.packed_fmt;
      (* which storage format — existential over the GADT *)
  quant : Quant.t option; (* Some iff fmt is quantized *)
}

let create ~id ~name:_ ~shape ~fmt ?quant () = { id; shape; fmt; quant }
let pp fmt s = Tensor_id.pp fmt s.id

let jsont : t Jsont.t =
  Jsont.map ~kind:"tensor_sig"
    ~dec:(fun json ->
      let ms = Json_util.req_obj json "tensor_sig" in
      let get k c = Json_util.req_field ms k c "tensor_sig" in
      let id = get "id" Tensor_id.jsont in
      let shape = get "shape" Vec6.shape_jsont in
      let fmt = get "fmt" Payload.packed_fmt_jsont in
      let quant = Json_util.opt_field ms "quant" Quant.jsont in
      { id; shape; fmt; quant })
    ~enc:(fun sg ->
      let base =
        [
          ("fmt", Json_util.enc Payload.packed_fmt_jsont sg.fmt);
          ("id", Json_util.enc Tensor_id.jsont sg.id);
          ("shape", Json_util.enc Vec6.shape_jsont sg.shape);
        ]
      in
      let quant_kv =
        match sg.quant with
        | None -> []
        | Some q -> [ ("quant", Json_util.enc Quant.jsont q) ]
      in
      Json_util.jobj (base @ quant_kv))
    Jsont.json
