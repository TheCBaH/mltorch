open Ssa_ir

(* A tensor signature as the buffer declaration an SSA program holds, and back.
   Quantization parameters are copied out of the (abstract) [Quant.t] through
   its own accessors, so a program holds no reference into a signature. *)

let quant (q : Quant.t) : Ssa_format.quant =
  match Quant.channel_count q with
  | None ->
      let scale, zero_point = Quant.params q ~c:(Dim.index 0) in
      Ssa_format.Per_tensor { scale; zero_point }
  | Some n ->
      let params = Array.init n (fun i -> Quant.params q ~c:(Dim.index i)) in
      Ssa_format.Per_channel
        { scale = Array.map fst params; zero_point = Array.map snd params }

(* The format of a signature, or [None] for a quantized format declared without
   its parameters. *)
let format (sg : Tensor_sig.t) : Ssa_format.t option =
  let (Payload.Fmt f) = sg.Tensor_sig.fmt in
  let quantized make =
    Option.map (fun q -> make (quant q)) sg.Tensor_sig.quant
  in
  match f with
  | Payload.BF16 -> Some Ssa_format.Bf16
  | Payload.Bool -> Some Ssa_format.Bool
  | Payload.F16 -> Some Ssa_format.F16
  | Payload.F32 -> Some Ssa_format.F32
  | Payload.F64 -> Some Ssa_format.F64
  | Payload.I16 -> quantized (fun q -> Ssa_format.I16 q)
  | Payload.I32 -> Some Ssa_format.I32
  | Payload.I64 -> Some Ssa_format.I64
  | Payload.I8 -> quantized (fun q -> Ssa_format.I8 q)

let format_name (sg : Tensor_sig.t) =
  let (Payload.Fmt f) = sg.Tensor_sig.fmt in
  Payload.fmt_name f

let extents (sg : Tensor_sig.t) =
  Expr.Coord.of_fn (fun a ->
      Int64.of_int (Dim.to_int (Vec6.get sg.Tensor_sig.shape a)))

(* The declaration of a signature's buffer, if its format is representable. *)
let buffer (sg : Tensor_sig.t) role =
  Option.map
    (fun format ->
      {
        Ssa_buffer.id = Ssa_id.Buffer.of_int (Tensor_id.to_int sg.Tensor_sig.id);
        extents = extents sg;
        format;
        role;
      })
    (format sg)

let shape (b : Ssa_buffer.t) =
  let e a = Int64.to_int (Expr.Coord.get b.Ssa_buffer.extents a) in
  Vec6.shape ~n:(e Expr.Axis.N) ~t:(e Expr.Axis.T) ~d:(e Expr.Axis.D)
    ~h:(e Expr.Axis.H) ~w:(e Expr.Axis.W) ~c:(e Expr.Axis.C)

let to_quant : Ssa_format.quant -> Quant.t = function
  | Ssa_format.Per_tensor { scale; zero_point } ->
      Quant.per_tensor ~scale ~zero_point
  | Ssa_format.Per_channel { scale; zero_point } ->
      Err.or_raise ~pp_error:Quant.pp_error
        (Quant.per_channel ~scale ~zero_point)

(* The signature a buffer declaration stands for. *)
let signature (b : Ssa_buffer.t) : Tensor_sig.t =
  let id = Tensor_id.of_int (b.Ssa_buffer.id :> int) in
  let fmt, quant =
    match b.Ssa_buffer.format with
    | Ssa_format.Bf16 -> (Payload.Fmt Payload.BF16, None)
    | Ssa_format.Bool -> (Payload.Fmt Payload.Bool, None)
    | Ssa_format.F16 -> (Payload.Fmt Payload.F16, None)
    | Ssa_format.F32 -> (Payload.Fmt Payload.F32, None)
    | Ssa_format.F64 -> (Payload.Fmt Payload.F64, None)
    | Ssa_format.I16 q -> (Payload.Fmt Payload.I16, Some (to_quant q))
    | Ssa_format.I32 -> (Payload.Fmt Payload.I32, None)
    | Ssa_format.I64 -> (Payload.Fmt Payload.I64, None)
    | Ssa_format.I8 q -> (Payload.Fmt Payload.I8, Some (to_quant q))
  in
  Tensor_sig.create ~id ~name:"" ~shape:(shape b) ~fmt ?quant ()
