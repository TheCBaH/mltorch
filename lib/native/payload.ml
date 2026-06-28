(* The element-format GADT and the typed payload. ['elt] is the *storage* cell
   (what [data.{i}] returns), ['ba] its Bigarray kind, and ['q] a phantom tag
   saying whether turning a cell into a real value needs quantization metadata
   (`Quant) or is fixed by the format (`Real). The quantization GADT is indexed by
   the same tag, so a `Real payload cannot carry quant params and a `Quant one
   cannot omit them. See .ai/native_tensor_design.md §2. *)

type ('elt, 'ba, 'q) fmt =
  | F32 : (float, Bigarray.float32_elt, [ `Real ]) fmt
  | I64 : (int64, Bigarray.int64_elt, [ `Real ]) fmt
  | I32 : (int32, Bigarray.int32_elt, [ `Real ]) fmt
  | F16 :
      ( int,
        Bigarray.int16_unsigned_elt,
        [ `Real ] )
      fmt (* IEEE half, raw bits *)
  | BF16 :
      (int, Bigarray.int16_unsigned_elt, [ `Real ]) fmt (* bfloat16, raw bits *)
  | I16 : (int, Bigarray.int16_signed_elt, [ `Quant ]) fmt
  | I8 : (int, Bigarray.int8_signed_elt, [ `Quant ]) fmt

type _ quantization =
  | No_quant : [ `Real ] quantization
  | Quant : Quant.t -> [ `Quant ] quantization

type ('elt, 'ba, 'q) payload = {
  fmt : ('elt, 'ba, 'q) fmt;
  quant : 'q quantization;
  data : ('elt, 'ba, Bigarray.c_layout) Bigarray.Array1.t;
}

(* The concrete format, hidden — recovered by matching [fmt]. *)
type packed_fmt = Fmt : ('elt, 'ba, 'q) fmt -> packed_fmt

let fmt_name : type e b q. (e, b, q) fmt -> string = function
  | F32 -> "f32"
  | I64 -> "i64"
  | I32 -> "i32"
  | F16 -> "f16"
  | BF16 -> "bf16"
  | I16 -> "i16"
  | I8 -> "i8"

let pp_fmt fmt f = Format.pp_print_string fmt (fmt_name f)

(* Integer storage range of a quantized format (for re-quantising on store). *)
let qrange : type e b q. (e, b, q) fmt -> int * int = function
  | I8 -> (-128, 127)
  | I16 -> (-32768, 32767)
  | _ -> invalid_arg "Payload.qrange: not a quantized format"

(* Decode storage cell [i] to the compute domain (float); [c] is the channel
   coordinate, used by per-channel dequant. *)
let get_float : type e b q. (e, b, q) payload -> c:int -> i:int -> float =
 fun p ~c ~i ->
  match p.fmt with
  | F32 -> p.data.{i}
  | I64 -> Int64.to_float p.data.{i}
  | I32 -> Int32.to_float p.data.{i}
  | F16 -> Half.Half.to_float p.data.{i}
  | BF16 -> Half.Bf16.to_float p.data.{i}
  | I16 -> (
      match p.quant with Quant qz -> Quant.dequantize qz ~c ~q:p.data.{i})
  | I8 -> (
      match p.quant with Quant qz -> Quant.dequantize qz ~c ~q:p.data.{i})

(* Encode a float into storage cell [i] (round/quantise per format). *)
let set_float : type e b q. (e, b, q) payload -> c:int -> i:int -> float -> unit
    =
 fun p ~c ~i x ->
  match p.fmt with
  | F32 -> p.data.{i} <- x
  | I64 -> p.data.{i} <- Int64.of_float x
  | I32 -> p.data.{i} <- Int32.of_float x
  | F16 -> p.data.{i} <- Half.Half.of_float x
  | BF16 -> p.data.{i} <- Half.Bf16.of_float x
  | I16 -> (
      let qmin, qmax = qrange p.fmt in
      match p.quant with
      | Quant qz -> p.data.{i} <- Quant.quantize qz ~c ~qmin ~qmax x)
  | I8 -> (
      let qmin, qmax = qrange p.fmt in
      match p.quant with
      | Quant qz -> p.data.{i} <- Quant.quantize qz ~c ~qmin ~qmax x)

let pp : type e b q. Format.formatter -> (e, b, q) payload -> unit =
 fun fmt p ->
  match p.quant with
  | No_quant -> pp_fmt fmt p.fmt
  | Quant qz -> Format.fprintf fmt "%a[%a]" pp_fmt p.fmt Quant.pp qz

(* ---- JSON codecs ---------------------------------------------------------- *)

let packed_fmt_jsont : packed_fmt Jsont.t =
  Jsont.map ~kind:"fmt"
    ~dec:(fun s ->
      match s with
      | "bf16" -> Fmt BF16
      | "f16" -> Fmt F16
      | "f32" -> Fmt F32
      | "i16" -> Fmt I16
      | "i32" -> Fmt I32
      | "i64" -> Fmt I64
      | "i8" -> Fmt I8
      | _ -> Jsont.Error.msgf Jsont.Meta.none "fmt: unknown %S" s)
    ~enc:(fun (Fmt f) -> fmt_name f)
    Jsont.string

(* Encodes the bulk data of a payload as the union {"Array":[...]} or
   {"None":null}.  [numel] is the number of elements; [iter_floats] yields
   them in storage order.  If [max_elts] is [Some n] and numel > n, the None
   branch is taken. *)
let enc_data_union ~(max_elts : int option) ~numel ~iter_floats : Jsont.json =
  let omit = match max_elts with Some m -> numel > m | None -> false in
  if omit then Json_util.single ~case:"None" Json_util.jnull
  else
    let buf = Array.make numel 0.0 in
    iter_floats (fun i f -> buf.(i) <- f);
    Json_util.single ~case:"Array"
      (Json_util.jarr
         (Array.to_list (Array.map (Json_util.enc Json_util.f32_jsont) buf)))
