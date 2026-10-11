let version = 1
let magic = "MLCJ"
let max_body = 1 lsl 28
let ( let* ) = Result.bind

type request =
  | Hello
  | Prepare of { handle : int; engine : string; graph : string }
  | Run of { handle : int; inputs : (int * Tensor.packed) list }
  | Release of int

type response =
  | Welcome of { engines : string list; runtime : string }
  | Prepared
  | Refused of string
  | Outputs of Tensor.packed list
  | Failed of string
  | Released

(* ---- writing ---- *)

let u8 b n = Buffer.add_uint8 b n
let u32 b n = Buffer.add_int32_le b (Int32.of_int n)

let str b s =
  u32 b (String.length s);
  Buffer.add_string b s

let header kind =
  let b = Buffer.create 64 in
  Buffer.add_string b magic;
  u32 b version;
  u8 b kind;
  b

let tag_of : Tensor.packed -> (int, string) result =
 fun (Tensor.Tensor t) ->
  match t.Tensor.payload.Payload.fmt with
  | Payload.F32 -> Ok 1
  | Payload.F64 -> Ok 2
  | Payload.I32 -> Ok 3
  | Payload.I64 -> Ok 4
  | Payload.Bool -> Ok 5
  | _ -> Error "the wire carries f32, f64, i32, i64 and bool tensors"

let axes = Axis.all

let tensor b (Tensor.Tensor t as packed) =
  let tag = match tag_of packed with Ok k -> k | Error m -> invalid_arg m in
  u8 b tag;
  List.iter (fun a -> u32 b (Vec6.get t.Tensor.shape a :> int)) axes;
  let p = t.Tensor.payload in
  let n = Bigarray.Array1.dim p.Payload.data in
  let w = Payload.cell_bytes p.Payload.fmt in
  Buffer.add_int64_le b (Int64.of_int (n * w));
  let cell : type e b q. (e, b, q) Payload.fmt -> e -> unit =
   fun fmt v ->
    match fmt with
    | Payload.F32 -> Buffer.add_int32_le b (Int32.bits_of_float v)
    | Payload.F64 -> Buffer.add_int64_le b (Int64.bits_of_float v)
    | Payload.I32 -> Buffer.add_int32_le b v
    | Payload.I64 -> Buffer.add_int64_le b v
    | Payload.Bool -> Buffer.add_uint8 b (if v = 0 then 0 else 1)
    | _ -> invalid_arg "wire format"
  in
  for i = 0 to n - 1 do
    cell p.Payload.fmt (Bigarray.Array1.unsafe_get p.Payload.data i)
  done

let encode_request = function
  | Hello -> Buffer.contents (header 1)
  | Prepare { handle; engine; graph } ->
      let b = header 2 in
      u32 b handle;
      str b engine;
      str b graph;
      Buffer.contents b
  | Run { handle; inputs } ->
      let b = header 3 in
      u32 b handle;
      u32 b (List.length inputs);
      List.iter
        (fun (id, t) ->
          u32 b id;
          tensor b t)
        inputs;
      Buffer.contents b
  | Release h ->
      let b = header 4 in
      u32 b h;
      Buffer.contents b

let encode_response = function
  | Welcome { engines; runtime } ->
      let b = header 1 in
      u32 b (List.length engines);
      List.iter (str b) engines;
      str b runtime;
      Buffer.contents b
  | Prepared -> Buffer.contents (header 2)
  | Refused m ->
      let b = header 3 in
      str b m;
      Buffer.contents b
  | Outputs ts ->
      let b = header 4 in
      u32 b (List.length ts);
      List.iter (tensor b) ts;
      Buffer.contents b
  | Failed m ->
      let b = header 5 in
      str b m;
      Buffer.contents b
  | Released -> Buffer.contents (header 6)

let frame body =
  let b = Buffer.create (String.length body + 4) in
  u32 b (String.length body);
  Buffer.add_string b body;
  Buffer.contents b

let frame_length s =
  if String.length s <> 4 then Error "a length prefix is 4 bytes"
  else
    let n = Int32.to_int (String.get_int32_le s 0) in
    if n < 0 || n > max_body then
      Error
        (Printf.sprintf "message of %d bytes is over the %d limit" n max_body)
    else Ok n

(* ---- reading ---- *)

type cursor = { s : string; mutable at : int }

let need c n =
  if n < 0 || c.at + n > String.length c.s then Error "message is truncated"
  else Ok ()

let get_u8 c =
  let* () = need c 1 in
  let v = String.get_uint8 c.s c.at in
  c.at <- c.at + 1;
  Ok v

(* A count or extent: non-negative and below 2^31. *)
let get_u32 c =
  let* () = need c 4 in
  let v = Int32.to_int (String.get_int32_le c.s c.at) in
  c.at <- c.at + 4;
  if v < 0 then Error "a count is out of range" else Ok v

let get_str c =
  let* n = get_u32 c in
  let* () = need c n in
  let v = String.sub c.s c.at n in
  c.at <- c.at + n;
  Ok v

let rec repeat n f acc =
  if n = 0 then Ok (List.rev acc)
  else
    let* x = f () in
    repeat (n - 1) f (x :: acc)

let make : type e b.
    shape:Vec6.shape ->
    (e, b, [ `Real ]) Payload.fmt ->
    int ->
    (int -> e) ->
    Tensor.packed =
 fun ~shape fmt n get ->
  let kind : (e, b) Bigarray.kind =
    match fmt with
    | Payload.F32 -> Bigarray.float32
    | Payload.F64 -> Bigarray.float64
    | Payload.I32 -> Bigarray.int32
    | Payload.I64 -> Bigarray.int64
    | Payload.Bool -> Bigarray.int8_unsigned
    | Payload.BF16 -> Bigarray.int16_unsigned
    | Payload.F16 -> Bigarray.int16_unsigned
  in
  let data = Bigarray.Array1.create kind Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.unsafe_set data i (get i)
  done;
  Tensor.Tensor
    { Tensor.shape; payload = { Payload.fmt; quant = Payload.No_quant; data } }

let get_tensor c =
  let* tag = get_u8 c in
  let* extents = repeat 6 (fun () -> get_u32 c) [] in
  let* shape =
    match extents with
    | [ n; t; d; h; w; cc ] -> Ok (Vec6.shape ~n ~t ~d ~h ~w ~c:cc)
    | _ -> Error "a tensor has six extents"
  in
  let* () = need c 8 in
  let bytes64 = String.get_int64_le c.s c.at in
  c.at <- c.at + 8;
  let* cells =
    (* the count each extent product allows, bounded before it is narrowed *)
    match
      Err.payload (Vec6.numel_bounded ~limit:(Int64.of_int max_body) shape)
    with
    | Ok k -> Ok (Int64.to_int k)
    | Error _ -> Error "a tensor has too many elements"
  in
  let width = match tag with 1 | 3 -> 4 | 2 | 4 -> 8 | 5 -> 1 | _ -> 0 in
  if width = 0 then Error "unknown tensor format"
  else if Int64.compare bytes64 (Int64.of_int (cells * width)) <> 0 then
    Error "a tensor's byte count does not match its extents"
  else
    let* () = need c (cells * width) in
    let at = c.at and s = c.s in
    c.at <- c.at + (cells * width);
    Ok
      (match tag with
      | 1 ->
          make ~shape Payload.F32 cells (fun i ->
              Int32.float_of_bits (String.get_int32_le s (at + (4 * i))))
      | 2 ->
          make ~shape Payload.F64 cells (fun i ->
              Int64.float_of_bits (String.get_int64_le s (at + (8 * i))))
      | 3 ->
          make ~shape Payload.I32 cells (fun i ->
              String.get_int32_le s (at + (4 * i)))
      | 4 ->
          make ~shape Payload.I64 cells (fun i ->
              String.get_int64_le s (at + (8 * i)))
      | _ ->
          make ~shape Payload.Bool cells (fun i ->
              if String.get_uint8 s (at + i) = 0 then 0 else 1))

let open_message s =
  let c = { s; at = 0 } in
  let* () = need c 8 in
  if String.sub s 0 4 <> magic then Error "not a comparison message"
  else
    let v = Int32.to_int (String.get_int32_le s 4) in
    if v <> version then
      Error
        (Printf.sprintf "protocol version %d, this side speaks %d" v version)
    else (
      c.at <- 8;
      let* kind = get_u8 c in
      Ok (c, kind))

let finish c v =
  if c.at <> String.length c.s then Error "trailing bytes after the message"
  else Ok v

let decode_request s =
  let* c, kind = open_message s in
  match kind with
  | 1 -> finish c Hello
  | 2 ->
      let* handle = get_u32 c in
      let* engine = get_str c in
      let* graph = get_str c in
      finish c (Prepare { handle; engine; graph })
  | 3 ->
      let* handle = get_u32 c in
      let* n = get_u32 c in
      let* inputs =
        repeat n
          (fun () ->
            let* id = get_u32 c in
            let* t = get_tensor c in
            Ok (id, t))
          []
      in
      finish c (Run { handle; inputs })
  | 4 ->
      let* h = get_u32 c in
      finish c (Release h)
  | k -> Error (Printf.sprintf "unknown request kind %d" k)

let decode_response s =
  let* c, kind = open_message s in
  match kind with
  | 1 ->
      let* n = get_u32 c in
      let* engines = repeat n (fun () -> get_str c) [] in
      let* runtime = get_str c in
      finish c (Welcome { engines; runtime })
  | 2 -> finish c Prepared
  | 3 ->
      let* m = get_str c in
      finish c (Refused m)
  | 4 ->
      let* n = get_u32 c in
      let* ts = repeat n (fun () -> get_tensor c) [] in
      finish c (Outputs ts)
  | 5 ->
      let* m = get_str c in
      finish c (Failed m)
  | 6 -> finish c Released
  | k -> Error (Printf.sprintf "unknown response kind %d" k)
