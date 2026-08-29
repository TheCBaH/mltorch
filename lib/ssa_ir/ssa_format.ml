(* A buffer's storage format. How a load decodes it to a working type is the
   access's own explicit [Ssa_op.Decode], never implied by the buffer, and a
   store names the encode it applies. A quantized format carries its parameters
   as data, so a decode needs nothing the buffer's declaration does not hold. *)

(* [real = scale * (q - zero_point)], per tensor or per channel along C. *)
type quant =
  | Per_channel of { scale : float array; zero_point : int array }
  | Per_tensor of { scale : float; zero_point : int }

type t =
  | Bf16
  | Bool
  | F16
  | F32
  | F64
  | I16 of quant
  | I32
  | I64
  | I8 of quant

(* The format without its parameters: what an access's decode or encode names. *)
module Family = struct
  type t = Bf16 | Bool | F16 | F32 | F64 | I16 | I32 | I64 | I8

  let name = function
    | Bf16 -> "bf16"
    | Bool -> "bool"
    | F16 -> "f16"
    | F32 -> "f32"
    | F64 -> "f64"
    | I16 -> "i16"
    | I32 -> "i32"
    | I64 -> "i64"
    | I8 -> "i8"
end

let family = function
  | Bf16 -> Family.Bf16
  | Bool -> Family.Bool
  | F16 -> Family.F16
  | F32 -> Family.F32
  | F64 -> Family.F64
  | I16 _ -> Family.I16
  | I32 -> Family.I32
  | I64 -> Family.I64
  | I8 _ -> Family.I8

let name t = Family.name (family t)
let quant = function I16 q | I8 q -> Some q | _ -> None

(* A flat element offset cannot name the channel a per-channel decode needs. *)
let per_channel = function
  | I16 (Per_channel _) | I8 (Per_channel _) -> true
  | Bf16 | Bool | F16 | F32 | F64 | I32 | I64
  | I16 (Per_tensor _)
  | I8 (Per_tensor _) ->
      false

(* The channels a per-channel format holds parameters for. *)
let channels = function
  | I16 (Per_channel { scale; _ }) | I8 (Per_channel { scale; _ }) ->
      Some (Array.length scale)
  | _ -> None

(* Whether a format's cells are floats, integers or int64s in the interpreter. *)
type cells = Float_cells | Int_cells | Int64_cells

let cells = function
  | Bool | F32 | F64 -> Float_cells
  | Bf16 | F16 | I16 _ | I32 | I8 _ -> Int_cells
  | I64 -> Int64_cells

(* The quantized value of a cell. *)
let dequantize q ~channel ~cell =
  let scale, zero_point =
    match q with
    | Per_tensor { scale; zero_point } -> (scale, zero_point)
    | Per_channel { scale; zero_point } ->
        (scale.(channel), zero_point.(channel))
  in
  scale *. float_of_int (cell - zero_point)
