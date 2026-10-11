module W = Inference_compare_js_wire.Wire

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:2

let tensor : type e b.
    (e, b, [ `Real ]) Payload.fmt ->
    (e, b) Bigarray.kind ->
    e list ->
    Tensor.packed =
 fun fmt kind cells ->
  let data =
    Bigarray.Array1.create kind Bigarray.c_layout (List.length cells)
  in
  List.iteri (fun i v -> Bigarray.Array1.set data i v) cells;
  Tensor.Tensor
    { Tensor.shape; payload = { Payload.fmt; quant = Payload.No_quant; data } }

let i64 = tensor Payload.I64 Bigarray.int64
let bool_ = tensor Payload.Bool Bigarray.int8_unsigned
let f32 = tensor Payload.F32 Bigarray.float32

(* Beyond 2^53 an integer stored as a float is not itself. *)
let big = Int64.add (Int64.shift_left 1L 53) 1L
let odd_nan = Int32.float_of_bits 0x7fc00001l

let cells (Tensor.Tensor t) =
  let p = t.Tensor.payload in
  let n = Bigarray.Array1.dim p.Payload.data in
  let b = Buffer.create 16 in
  let one : type e b q. (e, b, q) Payload.fmt -> e -> unit =
   fun fmt v ->
    match fmt with
    | Payload.I64 -> Buffer.add_string b (Int64.to_string v ^ " ")
    | Payload.Bool -> Buffer.add_string b (string_of_int v ^ " ")
    | Payload.F32 ->
        Buffer.add_string b (Printf.sprintf "%08lx " (Int32.bits_of_float v))
    | _ -> Buffer.add_string b "?"
  in
  for i = 0 to n - 1 do
    one p.Payload.fmt (Bigarray.Array1.get p.Payload.data i)
  done;
  String.trim (Buffer.contents b)

let ts =
  [
    i64 [ big; -1L; Int64.max_int; Int64.min_int ];
    bool_ [ 0; 1; 1; 0 ];
    f32 [ odd_nan; -0.; 1.5; infinity ];
  ]

let%expect_test "tensors cross the wire exactly" =
  (match W.decode_response (W.encode_response (W.Outputs ts)) with
  | Ok (W.Outputs back) -> List.iter (fun t -> print_endline (cells t)) back
  | Ok _ -> print_endline "other response"
  | Error m -> print_endline m);
  [%expect
    {|
    9007199254740993 -1 9223372036854775807 -9223372036854775808
    0 1 1 0
    7fc00001 80000000 3fc00000 7f800000 |}]

let%expect_test "requests round-trip" =
  let show r =
    match W.decode_request (W.encode_request r) with
    | Ok W.Hello -> print_endline "hello"
    | Ok (W.Prepare { handle; engine; graph }) ->
        Printf.printf "prepare %d %s %s\n" handle engine graph
    | Ok (W.Run { handle; inputs }) ->
        Printf.printf "run %d: %s\n" handle
          (String.concat "; "
             (List.map
                (fun (id, t) -> Printf.sprintf "t%d = %s" id (cells t))
                inputs))
    | Ok (W.Release h) -> Printf.printf "release %d\n" h
    | Error m -> print_endline m
  in
  show W.Hello;
  show (W.Prepare { handle = 7; engine = "loop-js"; graph = "{\"g\":1}" });
  show (W.Run { handle = 7; inputs = [ (3, i64 [ big; 0L; 0L; 0L ]) ] });
  show (W.Release 7);
  [%expect
    {|
    hello
    prepare 7 loop-js {"g":1}
    run 7: t3 = 9007199254740993 0 0 0
    release 7 |}]

let%expect_test "a damaged message is rejected whole" =
  let good = W.encode_response (W.Outputs [ bool_ [ 1; 0; 1; 0 ] ]) in
  let show label s =
    match W.decode_response s with
    | Ok _ -> Printf.printf "%s: accepted\n" label
    | Error m -> Printf.printf "%s: %s\n" label m
  in
  show "good" good;
  show "truncated" (String.sub good 0 (String.length good - 1));
  show "trailing" (good ^ "x");
  show "wrong magic" ("XXXX" ^ String.sub good 4 (String.length good - 4));
  let other_version = Bytes.of_string good in
  Bytes.set_int32_le other_version 4 2l;
  show "other version" (Bytes.to_string other_version);
  let kind = Bytes.of_string good in
  Bytes.set_uint8 kind 8 99;
  show "unknown kind" (Bytes.to_string kind);
  (* the tensor's byte count lies about its extents *)
  let count = Bytes.of_string good in
  Bytes.set_int64_le count (9 + 4 + 1 + 24) 3L;
  show "byte count" (Bytes.to_string count);
  show "empty" "";
  [%expect
    {|
    good: accepted
    truncated: message is truncated
    trailing: trailing bytes after the message
    wrong magic: not a comparison message
    other version: protocol version 2, this side speaks 1
    unknown kind: unknown response kind 99
    byte count: a tensor's byte count does not match its extents
    empty: message is truncated |}]

let%expect_test "a length prefix over the limit is refused" =
  let prefix n =
    let b = Bytes.create 4 in
    Bytes.set_int32_le b 0 n;
    Bytes.to_string b
  in
  List.iter
    (fun n ->
      match W.frame_length (prefix n) with
      | Ok k -> Printf.printf "%ld -> %d\n" n k
      | Error m -> Printf.printf "%ld -> %s\n" n m)
    [ 12l; -1l; Int32.max_int ];
  Printf.printf "framed: %d\n" (String.length (W.frame "abc"));
  [%expect
    {|
    12 -> 12
    -1 -> message of -1 bytes is over the 268435456 limit
    2147483647 -> message of 2147483647 bytes is over the 268435456 limit
    framed: 7 |}]

(* A tensor the wire does not carry is refused, never narrowed. *)
let%expect_test "an unsupported format is refused when sent" =
  let half =
    let data =
      Bigarray.Array1.create Bigarray.int16_unsigned Bigarray.c_layout 4
    in
    Tensor.Tensor
      {
        Tensor.shape;
        payload = { Payload.fmt = Payload.F16; quant = Payload.No_quant; data };
      }
  in
  (try ignore (W.encode_response (W.Outputs [ half ]))
   with Invalid_argument m -> print_endline m);
  [%expect {| the wire carries f32, f64, i32, i64 and bool tensors |}]
